import logging
from typing import Any, Dict, List, Optional

from qdrant_client.models import FieldCondition, Filter, MatchAny, MatchValue, PointStruct

from infrastructure import get_infra, qdrant

_LOG = logging.getLogger("search_repository")


class SearchRepository:
    def __init__(self, collection_name: str):
        self.collection_name = collection_name

    async def collection_exists(self) -> bool:
        async with get_infra().qdrant_semaphore:
            collections_res = await qdrant.get_collections()
        return any(c.name == self.collection_name for c in collections_res.collections)

    async def validate_collection_dimensions(self, required_dims: int) -> bool:
        """Checks if the existing collection's vector size matches the model's dimensions."""
        if not await self.collection_exists():
            return True # No collection yet, so it's 'valid' for creation
            
        async with get_infra().qdrant_semaphore:
            info = await qdrant.get_collection(self.collection_name)
            # Check the 'default' vector size
            if hasattr(info.config.params.vectors, 'size'):
                return info.config.params.vectors.size == required_dims
            # For multi-vector setups
            elif isinstance(info.config.params.vectors, dict):
                return info.config.params.vectors.get("default", {}).size == required_dims
        return True

    async def ensure_collection(self, vectors_config: Any):
        if await self.collection_exists():
            return

        async with get_infra().qdrant_semaphore:
            await qdrant.create_collection(
                collection_name=self.collection_name,
                vectors_config=vectors_config,
            )
        _LOG.info("Qdrant collection ready: %s", self.collection_name)

    async def recreate_collection(self, vectors_config: Any):
        async with get_infra().qdrant_semaphore:
            try:
                await qdrant.delete_collection(self.collection_name)
            except Exception:
                pass
            await qdrant.create_collection(
                collection_name=self.collection_name,
                vectors_config=vectors_config,
            )
        _LOG.info("Qdrant collection recreated: %s", self.collection_name)

    async def ensure_payload_indexes(self, index_specs: Dict[str, Any]):
        for field_name, field_schema in index_specs.items():
            try:
                async with get_infra().qdrant_semaphore:
                    await qdrant.create_payload_index(
                        collection_name=self.collection_name,
                        field_name=field_name,
                        field_schema=field_schema,
                        wait=True,
                    )
                _LOG.info("Qdrant payload index ready for %s.%s", self.collection_name, field_name)
            except Exception as exc:
                _LOG.warning(
                    "Failed to create payload index for %s.%s: %s",
                    self.collection_name,
                    field_name,
                    exc,
                )

    @staticmethod
    def _build_filter(
        *,
        exact_matches: Optional[Dict[str, Any]] = None,
        match_any: Optional[Dict[str, List[Any]]] = None,
    ) -> Optional[Filter]:
        must_conditions = []

        if exact_matches:
            must_conditions.extend(
                FieldCondition(key=key, match=MatchValue(value=value))
                for key, value in exact_matches.items()
            )

        if match_any:
            must_conditions.extend(
                FieldCondition(key=key, match=MatchAny(any=values))
                for key, values in match_any.items()
            )

        if not must_conditions:
            return None

        return Filter(must=must_conditions)

    async def dense_search(self, query_vector: List[float], batch_id: str, top_k: int = 50) -> Dict[str, float]:
        async with get_infra().qdrant_semaphore:
            results = await qdrant.search(
                collection_name=self.collection_name,
                query_vector=("default", query_vector),
                query_filter=self._build_filter(exact_matches={"batch_id": batch_id}),
                limit=top_k,
                with_payload=["chunk_id"],
            )

        dense_scores: Dict[str, float] = {}
        for point in results:
            payload = point.payload or {}
            chunk_id = payload.get("chunk_id")
            if chunk_id:
                dense_scores[str(chunk_id)] = float(point.score)
        return dense_scores

    async def search_by_vector(
        self,
        query_vector: List[float],
        *,
        exact_matches: Dict[str, Any],
        limit: int = 1,
        score_threshold: float | None = None,
        with_payload: Any = True,
    ):
        async with get_infra().qdrant_semaphore:
            return await qdrant.search(
                collection_name=self.collection_name,
                query_vector=("default", query_vector),
                query_filter=self._build_filter(exact_matches=exact_matches),
                limit=limit,
                score_threshold=score_threshold,
                with_payload=with_payload,
            )

    async def fetch_chunk_details(self, chunk_ids: List[str], batch_id: Optional[str] = None) -> List[Dict]:
        if not chunk_ids:
            return []

        points = await self.scroll_payloads(
            exact_matches={"batch_id": batch_id} if batch_id else None,
            match_any={"chunk_id": chunk_ids},
            limit=len(chunk_ids),
        )

        payload_lookup = {}
        for point in points:
            payload = point.payload or {}
            chunk_id = payload.get("chunk_id")
            if chunk_id:
                payload_lookup[str(chunk_id)] = payload

        return [payload_lookup[cid] for cid in chunk_ids if cid in payload_lookup]

    async def upsert_point(self, point_id: str, vector: List[float], payload: Dict[str, Any]):
        await self.upsert_points(
            [
                PointStruct(
                    id=point_id,
                    vector={"default": vector},
                    payload=payload,
                )
            ]
        )

    async def upsert_points(self, points: List[PointStruct]):
        if not points:
            return

        async with get_infra().qdrant_semaphore:
            await qdrant.upsert(
                collection_name=self.collection_name,
                points=points,
            )

    async def scroll_payloads(
        self,
        *,
        exact_matches: Optional[Dict[str, Any]] = None,
        match_any: Optional[Dict[str, List[Any]]] = None,
        limit: Optional[int] = None,
        page_size: int = 256,
        with_payload: Any = True,
    ):
        items = []
        offset = None
        scroll_filter = self._build_filter(exact_matches=exact_matches, match_any=match_any)

        while True:
            current_limit = page_size if limit is None else min(page_size, max(1, limit - len(items)))
            if current_limit <= 0:
                break

            async with get_infra().qdrant_semaphore:
                points, next_offset = await qdrant.scroll(
                    collection_name=self.collection_name,
                    scroll_filter=scroll_filter,
                    limit=current_limit,
                    offset=offset,
                    with_payload=with_payload,
                )

            items.extend(points)

            if next_offset is None:
                break
            if limit is not None and len(items) >= limit:
                break

            offset = next_offset

        return items[:limit] if limit is not None else items

    async def fetch_batch_chunks(self, batch_id: str, *, page_size: int = 512) -> List[Dict]:
        points = await self.scroll_payloads(
            exact_matches={"batch_id": batch_id},
            page_size=page_size,
            with_payload=True,
        )
        return [point.payload or {} for point in points]

    async def list_documents(self, batch_id: Optional[str] = None, *, limit: int = 100) -> List[Dict[str, Any]]:
        points = await self.scroll_payloads(
            exact_matches={"batch_id": batch_id} if batch_id else None,
            limit=min(max(limit * 20, 200), 5000),
            page_size=min(max(limit * 2, 100), 1000),
            with_payload=True,
        )

        seen: Dict[str, Dict[str, Any]] = {}
        for point in points:
            payload = point.payload or {}
            doc_id = payload.get("document_id")
            if not doc_id or doc_id in seen:
                continue

            seen[doc_id] = {
                "document_id": doc_id,
                "filename": payload.get("metadata", {}).get("filename", "Unknown"),
                "type": payload.get("type", "persistent"),
                "is_active": True,
                "ingested_at": payload.get("metadata", {}).get("ingested_at"),
            }

            if len(seen) >= limit:
                break

        return list(seen.values())

    async def get_document_payload(self, document_id: str) -> Optional[Dict[str, Any]]:
        points = await self.scroll_payloads(
            exact_matches={"document_id": document_id},
            limit=1,
            with_payload=True,
        )
        if not points:
            return None
        return points[0].payload or {}

    async def delete_payloads(
        self,
        *,
        exact_matches: Optional[Dict[str, Any]] = None,
        match_any: Optional[Dict[str, List[Any]]] = None,
    ):
        points_selector = self._build_filter(exact_matches=exact_matches, match_any=match_any)
        if points_selector is None:
            raise ValueError("delete_payloads requires at least one filter condition")

        async with get_infra().qdrant_semaphore:
            await qdrant.delete(
                collection_name=self.collection_name,
                points_selector=points_selector,
            )
