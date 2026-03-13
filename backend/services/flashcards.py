import json
import asyncio
from typing import List, Optional, Callable, Dict, Awaitable
from pydantic import BaseModel, Field
from services.llm import call_gemini_async
from json_utils import safe_json_load
from services.retrieval import retrieve_candidates
from retrieval.query_rewriter import rewrite_query_ensemble
from retrieval.bm25_index import BM25ChunkIndex

# 1. Strict Data Contract
class Flashcard(BaseModel):
    topic: str = Field(..., description="The high-level concept bucket (e.g., 'Neural Networks', 'History').")
    question: str = Field(..., description="The query for recall.")
    answer: str = Field(..., description="The verified fact or explanation.")
    card_type: str = Field(..., description="Type of recall: 'definition', 'contrast', 'cause-effect', 'list'.")
    citations: List[str] = Field(..., description="List of chunk_ids that provide evidence for this card.")
    confidence_score: float = Field(..., description="0.0 to 1.0 score of how well the evidence supports the answer.")

class FlashcardSet(BaseModel):
    cards: List[Flashcard]

# 2. Generation Service
async def generate_flashcards(
    bm25: Optional[BM25ChunkIndex],
    graph: Dict,
    chunk_fetcher: Callable,
    cohort_id: str,
    limit: int = 15,
    dense_retrieve_fn: Optional[Callable] = None,
    call_gemini_fn: Optional[Callable] = None,
    topics: Optional[List[str]] = None,
    complexity: str = "Undergrad"
) -> List[dict]:
    """
    Generates retrieval-grounded flashcards (Async/Stateless).
    """
    llm_fn = call_gemini_fn or call_gemini_async
    
    complexity_map = {
        "5-Year-Old": "Use very simple words and basic concepts only.",
        "High School": "Use clear, standard language suitable for teens.",
        "Undergrad": "Use academic terminology and standard depth.",
        "PhD Expert": "Use highly technical language and focus on advanced nuances."
    }
    c_instr = complexity_map.get(complexity, complexity_map["Undergrad"])

    all_final_cards = []

    if topics and dense_retrieve_fn:
        for topic in topics:
            print(f"   [Flashcards] Generating up to {limit} cards for topic: '{topic}' at {complexity} level...")
            
            rewrite_result = await rewrite_query_ensemble(topic, llm_fn, 3, 2, use_llm=True)
            rewrites = rewrite_result.get("rewrites", [{"query": topic, "weight": 1.0}])
            
            candidates = await retrieve_candidates(
                query=topic,
                rewrites=rewrites,
                bm25=bm25,
                graph=graph,
                chunk_fetcher=chunk_fetcher,
                dense_fn=dense_retrieve_fn,
                cohort_id=cohort_id,
                max_candidates=15,
                min_dense_score=0.30 # AC-35: Enforce Chat-level precision
            )

            
            if not candidates:
                print(f"   [Flashcards] No context found for topic: '{topic}'. Skipping.")
                continue

            context_text = " ".join([
                f"<CHUNK id='{c.get('doc_id') or c.get('chunk_id')}'>\n{c['text']}\n</CHUNK>"
                for c in candidates
            ])

            prompt = f"""
            You are an Expert Knowledge Extraction Engine.
            Your Goal: Create a set of high-quality Flashcards for the specific topic: "{topic}".
            
            COMPLEXITY LEVEL: {complexity} ({c_instr})

            STRICT CONSTRAINTS:
            1. TOPIC-FOCUS: All cards must strictly relate to "{topic}".
            2. EVIDENCE-LOCKED: Every answer MUST be derived *exclusively* from the provided CHUNK sources.
            3. CITATION-REQUIRED: You must cite the 'chunk_id' for every fact used.
            4. SCHEMA-LOCKED: Output must be a JSON object matching the defined schema.
            5. QUALITY-GATE: 
               - If a chunk is too vague, IGNORE it. 
               - If you cannot form a clear Question/Answer pair with >0.8 confidence, DO NOT create a card.
               
            SCHEMA (JSON):
            {{
              "cards": [
                {{
                  "topic": "{topic}",
                  "question": "string", 
                  "answer": "string",
                  "card_type": "definition" | "contrast" | "cause-effect" | "list",
                  "citations": ["chunk_id_1", "chunk_id_2"],
                  "confidence_score": float (0.0-1.0)
                }}
              ]
            }}

            SOURCES:
            {context_text}
            
            Generate {limit} flashcards for "{topic}" now.
            """

            try:
                response_text = await llm_fn(prompt)
                data = safe_json_load(response_text)
                flashcard_set = FlashcardSet(**data)
                
                valid_cards = [
                    card.dict() for card in flashcard_set.cards 
                    if card.confidence_score >= 0.7
                ]
                
                for vc in valid_cards:
                    vc["topic"] = topic
                    
                all_final_cards.extend(valid_cards)
                print(f"   [Flashcards] Successfully generated {len(valid_cards)} cards for '{topic}'.")
            except Exception as e:
                print(f"   [Flashcards] Failed for topic '{topic}': {e}")
                
        return all_final_cards
    return []