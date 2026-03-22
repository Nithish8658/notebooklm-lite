from typing import List, Callable, Dict, Optional, Awaitable
import json
import uuid
import asyncio
from pydantic import BaseModel, Field
from services.llm import call_gemini_async
from services.embedding_runtime import EmbeddingRequestContext
from services.phase_logging import PhaseTrace
from retrieval.query_rewriter import rewrite_query_ensemble
from services.retrieval import retrieve_candidates
from json_utils import safe_json_load
from retrieval.bm25_index import BM25ChunkIndex

# --- Data Models ---

class QuizOption(BaseModel):
    id: str = Field(..., description="Option identifier (A, B, C, D)")
    text: str = Field(..., description="The text of the option")

class QuizQuestion(BaseModel):
    id: str = Field(default_factory=lambda: str(uuid.uuid4()), description="Unique ID for the question")
    question: str = Field(..., description="The question text")
    options: List[QuizOption] = Field(..., description="List of 4 options")
    correct_option_id: str = Field(..., description="The ID of the correct option (A, B, C, or D)")
    explanation: str = Field(..., description="Explanation of why the answer is correct")
    citation: Optional[str] = Field(None, description="Chunk ID source")

class Quiz(BaseModel):
    id: str = Field(default_factory=lambda: str(uuid.uuid4()), description="Unique ID for the quiz set")
    topic: str = Field(..., description="The topic this quiz covers")
    questions: List[QuizQuestion]

# --- Service Logic ---

async def generate_quiz_for_topic(
    topic: str, 
    bm25: Optional[BM25ChunkIndex],
    graph: Dict,
    chunk_fetcher: Callable,
    dense_retrieve_fn: Callable, 
    call_gemini_fn: Optional[Callable[[str], Awaitable[str]]] = None,
    batch_id: str = "default_batch",
    complexity: str = "Undergrad"
) -> Optional[Quiz]:
    """
    Generates a quiz (set of MCQs) for a specific topic (Async/Stateless).
    """
    llm_fn = call_gemini_fn or call_gemini_async
    trace = PhaseTrace(
        "QUIZ-TOPIC",
        f"batch={batch_id}|topic={topic}|complexity={complexity}",
    )
    trace.log("Starting quiz generation")
    
    complexity_map = {
        "5-Year-Old": "Create very simple, fundamental questions. Avoid any complex terms.",
        "High School": "Create challenging questions for a high school student. Clear concepts.",
        "Undergrad": "Create college-level questions with academic depth.",
        "PhD Expert": "Create extremely difficult, nuanced questions that test deep expertise and subtle details."
    }
    c_instr = complexity_map.get(complexity, complexity_map["Undergrad"])

    # 1. Retrieve Context
    try:
        embedding_context = EmbeddingRequestContext()
        rewrite_start = trace.start_phase("Query Rewriting")
        rewrite_result = await rewrite_query_ensemble(
            topic,
            llm_fn,
            3,
            2,
            use_llm=True,
            embedding_context=embedding_context,
        )
        trace.complete_phase(
            "Query Rewriting",
            rewrite_start,
            detail=f"rewrites={len(rewrite_result.get('rewrites', []))}",
        )
        rewrites = rewrite_result.get("rewrites", [{"query": topic, "weight": 1.0}])
        
        retrieval_start = trace.start_phase("Retrieval & Reranking")
        candidates = await retrieve_candidates(
            query=topic,
            rewrites=rewrites,
            bm25=bm25,
            graph=graph,
            chunk_fetcher=chunk_fetcher,
            dense_fn=lambda q, batch_id, top_k=50: dense_retrieve_fn(
                q,
                batch_id=batch_id,
                top_k=top_k,
                embedding_context=embedding_context,
            ),
            batch_id=batch_id,
            dense_top_k=50,
            max_candidates=6,
            min_dense_score=0.30,
            correlation_id=f"quiz_{topic[:10]}"
        )
        trace.complete_phase(
            "Retrieval & Reranking",
            retrieval_start,
            detail=f"candidates={len(candidates)}",
        )

    except Exception as e:
        trace.failure(detail=f"retrieval_error={e}")
        return None

    if not candidates:
        trace.log("No relevant context found")
        return None

    context_text = "\n\n".join([
        f"<CHUNK id='{c['doc_id']}'>\n{c['text']}\n</CHUNK>"
        for c in candidates
    ])

    # 2. Prompt Gemini
    prompt = f"""
    You are an expert examiner. Create a Multiple Choice Quiz (MCQ) for the topic: "{topic}" only if the chunk sources are directly or sematically related.
    
    LEVEL: {complexity} ({c_instr})

    RULES

    1. Question Generation
    Create **10–15 challenging questions** for the specified LEVEL using only the provided CHUNKS related to the topic.
    2. Context Restriction
    All questions, options, answers, and explanations must be derived strictly from the CHUNKS. Do not use external knowledge or assumptions.
    3. Confidence Gate
    If the CHUNKS do not provide enough information to form a reliable question with ≥0.8 confidence, return: NO_RELEVANT_CONTEXT
    4. Question Format
    Each question must include **exactly four options**: A, B, C, D.
    5. Explanation
    Provide a clear explanation for the correct answer based only on the CHUNKS.
    6. Chunk Filtering
    Ignore vague or unrelated chunks. If no valid questions can be formed after filtering, return: NO_RELEVANT_CONTEXT
       
    OUTPUT FORMAT (Strict JSON):
    {{
      "questions": [
        {{
          "question": "Question text here?",
          "options": [
            {{ "id": "A", "text": "Option A text" }},
            {{ "id": "B", "text": "Option B text" }},
            {{ "id": "C", "text": "Option C text" }},
            {{ "id": "D", "text": "Option D text" }}
          ],
          "correct_option_id": "B",
          "explanation": "Why B is correct...",
          "citation": "chunk_id_source"
        }}
      ]
    }}

    SOURCE TEXT:
    {context_text}
    
    GENERATE JSON NOW:
    """

    try:
        llm_start = trace.start_phase("LLM Quiz Generation")
        response_text = await llm_fn(prompt)
        trace.complete_phase("LLM Quiz Generation", llm_start)

        parse_start = trace.start_phase("Response Parsing & Validation")
        data = safe_json_load(response_text)
        
        questions_data = data.get("questions", [])
        valid_questions = []
        
        for q in questions_data:
            try:
                if len(q["options"]) != 4: continue
                if q["correct_option_id"] not in ["A", "B", "C", "D"]: continue
                valid_questions.append(QuizQuestion(**q))
            except Exception as e:
                trace.log("Skipping invalid question", detail=str(e))
                continue
        
        if not valid_questions:
            trace.complete_phase("Response Parsing & Validation", parse_start, detail="valid_questions=0")
            trace.log("No valid questions generated")
            return None

        trace.complete_phase(
            "Response Parsing & Validation",
            parse_start,
            detail=f"valid_questions={len(valid_questions)}",
        )
        trace.success(detail=f"generated_questions={len(valid_questions)}")
        return Quiz(topic=topic, questions=valid_questions)

    except Exception as e:
        trace.failure(detail=f"generation_error={e}")
        return None
