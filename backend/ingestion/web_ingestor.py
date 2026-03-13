import httpx
import logging
import uuid
import re
import html
import unicodedata
from dataclasses import dataclass, field
from typing import List, Tuple, Dict
from bs4 import BeautifulSoup, Tag
import trafilatura
from trafilatura.sitemaps import sitemap_search
from readability import Document
from datetime import datetime
from urllib.parse import urljoin, urlparse

# ─────────────────────────────────────────────
# LOGGING
# ─────────────────────────────────────────────
logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO)


# ─────────────────────────────────────────────
# DATA CONTRACT (UNCHANGED)
# ─────────────────────────────────────────────
@dataclass
class NormalizedContentBlock:
    block_id: str
    block_type: str
    text: str
    heading_path: List[str]
    source_url: str
    dom_path: str
    confidence_score: float
    metadata: Dict[str, any] = field(default_factory=dict)


# ─────────────────────────────────────────────
# INGESTOR
# ─────────────────────────────────────────────
class WebIngestor:

    REMOVE_TAGS = {
        "nav", "footer", "aside", "header",
        "script", "style", "noscript",
        "iframe", "form", "svg", "canvas"
    }

    NOISE_RE = re.compile(
        r"(comment|cookie|popup|banner|sidebar|social|share|advert|promo)",
        re.IGNORECASE
    )

    def __init__(self, timeout: int = 30):
        self.timeout = timeout
        self.headers = {
            "User-Agent": "NotebookLM-Lite/1.0",
            "Accept": "text/html"
        }

    # ─────────────────────────────────────────────
    # PUBLIC ENTRY
    # ─────────────────────────────────────────────
    def ingest(self, url: str) -> List[NormalizedContentBlock]:
        try:
            raw_html, final_url = self._fetch(url)
            clean_html = self._sanitize(raw_html)
            extracted, method, metrics = self._extract(clean_html, final_url)
            blocks = self._reconstruct(extracted, final_url, method, metrics)
            return blocks
        except Exception as e:
            logger.error(f"Ingestion failed: {e}", exc_info=True)
            return [
                NormalizedContentBlock(
                    block_id=str(uuid.uuid4()),
                    block_type="error",
                    text=str(e),
                    heading_path=[],
                    source_url=url,
                    dom_path="root",
                    confidence_score=0.0,
                    metadata={"status": "partial_failure"}
                )
            ]

    def crawl(self, base_url: str, max_pages: int = 15) -> List[NormalizedContentBlock]:
        """
        Depth-1 Domain-Specific Crawler.
        1. Ingest base_url
        2. Discover internal links
        3. Ingest discovered links up to max_pages
        """
        logger.info(f"Starting Depth-1 crawl for: {base_url}")
        
        # 1. Ingest base
        all_blocks = self.ingest(base_url)
        
        # If the first ingest was an error, stop
        if any(b.block_type == "error" for b in all_blocks):
            return all_blocks

        # 2. Discover links from the raw HTML of the base_url
        try:
            raw_html, _ = self._fetch(base_url)
            links = self._discover_internal_links(raw_html, base_url)
        except Exception as e:
            logger.error(f"Link discovery failed: {e}")
            return all_blocks

        # 3. Ingest discovered links
        visited = {base_url}
        count = 1
        for link in links:
            if count >= max_pages:
                break
            if link in visited:
                continue
            
            logger.info(f"Crawling ({count}/{max_pages}): {link}")
            page_blocks = self.ingest(link)
            # Filter out error blocks from sub-pages to keep noise low
            all_blocks.extend([b for b in page_blocks if b.block_type != "error"])
            
            visited.add(link)
            count += 1

        return all_blocks

    def ingest_sitemap(self, sitemap_url: str, max_pages: int = 30) -> List[NormalizedContentBlock]:
        """
        Parses a sitemap and ingests all URLs found.
        """
        logger.info(f"Ingesting sitemap: {sitemap_url}")
        urls = sitemap_search(sitemap_url)
        
        if not urls:
            logger.warning("No URLs found in sitemap.")
            return []

        all_blocks = []
        count = 0
        for url in urls:
            if count >= max_pages:
                break
            
            logger.info(f"Sitemap progress ({count+1}/{len(urls)}): {url}")
            page_blocks = self.ingest(url)
            all_blocks.extend([b for b in page_blocks if b.block_type != "error"])
            
            count += 1
            
        return all_blocks

    def _discover_internal_links(self, html_text: str, base_url: str) -> List[str]:
        soup = BeautifulSoup(html_text, "html.parser")
        domain = urlparse(base_url).netloc
        links = set()
        
        for a in soup.find_all("a", href=True):
            href = a["href"]
            full_url = urljoin(base_url, href)
            parsed = urlparse(full_url)
            
            # Stay on same domain and use http(s)
            if parsed.netloc == domain and parsed.scheme in ["http", "https"]:
                # Strip fragments
                clean_url = full_url.split("#")[0].rstrip("/")
                if clean_url != base_url.rstrip("/"):
                    links.add(clean_url)
                    
        return sorted(list(links))

    # ─────────────────────────────────────────────
    # STAGE 1: FETCH
    # ─────────────────────────────────────────────
    def _fetch(self, url: str) -> Tuple[str, str]:
        if not url.startswith(("http://", "https://")):
            raise ValueError("Invalid URL scheme")

        with httpx.Client(
            headers=self.headers,
            timeout=self.timeout,
            follow_redirects=True
        ) as client:
            resp = client.get(url)
            resp.raise_for_status()

            ct = resp.headers.get("Content-Type", "")
            if "text/html" not in ct.lower():
                raise ValueError("Non-HTML content rejected")

            return resp.text, str(resp.url)

    # ─────────────────────────────────────────────
    # STAGE 2: SANITIZATION
    # ─────────────────────────────────────────────
    def _sanitize(self, html_text: str) -> str:
        soup = BeautifulSoup(html_text, "html.parser")

        # 1. Remove hard-noise tags
        for tag in self.REMOVE_TAGS:
            for el in soup.find_all(tag):
                el.decompose()

        # 2. Heuristic noise removal
        # IMPORTANT: iterate over a SNAPSHOT, not live DOM
        all_elements = list(soup.find_all(True))

        for el in all_elements:
            # If element has already been removed, skip
            if not isinstance(el, Tag):
                continue
            if el.attrs is None:
                continue

            el_id = el.attrs.get("id")
            if el_id and self.NOISE_RE.search(el_id):
                el.decompose()
                continue

            el_class = el.attrs.get("class")
            if el_class:
                class_str = " ".join(el_class) if isinstance(el_class, list) else str(el_class)
                if self.NOISE_RE.search(class_str):
                    el.decompose()
                    continue

        return str(soup)

    # ─────────────────────────────────────────────
    # STAGE 3: EXTRACTION
    # ─────────────────────────────────────────────
    def _extract(self, html_text: str, url: str) -> Tuple[str, str, Dict]:
        extracted = trafilatura.extract(
            html_text,
            url=url,
            include_tables=True,
            include_formatting=True,
            output_format="xml"
        )

        html_len = len(html_text)
        text_len = len(extracted) if extracted else 0
        density = text_len / max(html_len, 1)

        if extracted and density > 0.02 and text_len > 500:
            return extracted, "trafilatura", {
                "density": density,
                "text_length": text_len
            }

        doc = Document(html_text)
        summary = doc.summary(html_partial=True)
        text_len = len(summary)

        if text_len < 300:
            raise ValueError("Extraction quality insufficient")

        return summary, "readability", {
            "density": text_len / max(html_len, 1),
            "text_length": text_len
        }

    # ─────────────────────────────────────────────
    # STAGE 4–6: RECONSTRUCTION + NORMALIZATION + SCORING
    # ─────────────────────────────────────────────
    def _reconstruct(
        self,
        content: str,
        url: str,
        method: str,
        metrics: Dict
    ) -> List[NormalizedContentBlock]:

        soup = BeautifulSoup(content, "html.parser")
        blocks: List[NormalizedContentBlock] = []

        heading_stack: List[Tuple[int, str]] = []

        for el in soup.find_all(
            ["h1", "h2", "h3", "h4", "h5", "h6",
             "p", "pre", "code", "ul", "ol", "table"],
            recursive=True
        ):
            text = el.get_text(" ", strip=True)
            if not text and el.name != "table":
                continue

            block_type = "paragraph"

            # HEADING
            if el.name.startswith("h"):
                level = int(el.name[1])
                while heading_stack and heading_stack[-1][0] >= level:
                    heading_stack.pop()
                heading_stack.append((level, text))
                block_type = "heading"

            # CODE
            elif el.name in {"pre", "code"}:
                block_type = "code_block"

            # LIST
            elif el.name in {"ul", "ol"}:
                items = [
                    li.get_text(" ", strip=True)
                    for li in el.find_all("li", recursive=False)
                ]
                text = "\n".join(f"- {i}" for i in items if i)
                block_type = "list"

            # TABLE
            elif el.name == "table":
                text = self._table_to_md(el)
                block_type = "table"

            text = self._normalize(text)
            if not text:
                continue

            dom_path = self._dom_path(el)

            confidence = self._score(
                block_type,
                text,
                dom_path,
                method,
                metrics
            )

            blocks.append(
                NormalizedContentBlock(
                    block_id=str(uuid.uuid4()),
                    block_type=block_type,
                    text=text,
                    heading_path=[h[1] for h in heading_stack],
                    source_url=url,
                    dom_path=dom_path,
                    confidence_score=confidence,
                    metadata={
                        "method": method,
                        "ingested_at": datetime.utcnow().isoformat()
                    }
                )
            )

        return blocks

    # ─────────────────────────────────────────────
    # HELPERS
    # ─────────────────────────────────────────────
    def _normalize(self, text: str) -> str:
        text = unicodedata.normalize("NFKC", text)
        text = html.unescape(text)
        text = text.replace("\xa0", " ")
        text = re.sub(r"[ \t]+", " ", text)
        return text.strip()

    def _dom_path(self, el: Tag) -> str:
        parts = []
        while el and el.name != "[document]":
            siblings = (
                el.find_previous_siblings(el.name)
                if el.parent else []
            )
            idx = len(siblings)
            parts.append(f"{el.name}[{idx}]")
            el = el.parent
        return "/".join(reversed(parts))

    def _table_to_md(self, table: Tag) -> str:
        rows = []
        for tr in table.find_all("tr"):
            cols = [
                c.get_text(" ", strip=True)
                for c in tr.find_all(["th", "td"])
            ]
            if cols:
                rows.append("| " + " | ".join(cols) + " |")
        return "\n".join(rows)

    def _score(
        self,
        block_type: str,
        text: str,
        dom_path: str,
        method: str,
        metrics: Dict
    ) -> float:

        score = 1.0

        if method == "readability":
            score *= 0.9

        if block_type == "paragraph" and len(text) < 80:
            score *= 0.75

        if block_type == "code_block":
            score *= 1.05

        depth = dom_path.count("/")
        if depth > 10:
            score *= 0.85

        score *= min(1.0, metrics["density"] * 10)

        return round(max(0.1, min(score, 1.0)), 2)
