import os
from dotenv import load_dotenv
from langchain_groq import ChatGroq
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_classic.chains import create_retrieval_chain
from langchain_classic.chains.combine_documents import create_stuff_documents_chain
from src.retriever import build_or_load_vector_store

# Load the Groq API key from your .env file
load_dotenv()


# ─────────────────────────────────────────────
# STEP 1: Query Normalization Chain
# Translates messy Roman Urdu → clean English
# Used ONLY for retrieval, never shown as answer
# ─────────────────────────────────────────────

NORMALIZATION_SYSTEM_PROMPT = (
    "You are a precise translation engine. "
    "Your ONLY job is to translate the user's query from Roman Urdu (or code-mixed Roman Urdu/English) "
    "into a clean, formal English query suitable for searching a university policy document. "
    "STRICT RULES: "
    "1. Output ONLY the translated English query. No explanations, no preamble, no punctuation changes. "
    "2. Preserve all specific terms like GPA, CGPA, HoD, probation, semester exactly as-is. "
    "3. If the query is already in English, return it unchanged. "
    "4. Do NOT answer the question. Translate only."
)

normalization_prompt = ChatPromptTemplate.from_messages([
    ("system", NORMALIZATION_SYSTEM_PROMPT),
    ("human", "{raw_query}"),
])


# ─────────────────────────────────────────────
# STEP 3: Generation Prompt
# Same guardrails as baseline — do not weaken
# ─────────────────────────────────────────────

GENERATION_SYSTEM_PROMPT = (
    "You are a helpful, professional, and formal university assistant at GIKI. "
    "Use the following retrieved context to answer the user's question. "
    "IMPORTANT RULES: "
    "1. You MUST answer the question in polite, standard Pakistani Roman Urdu. "
    "2. STRICTLY PROHIBITED HINDI WORDS: Do NOT use words like 'yadi', 'anusar', 'avashyakta', 'kintu', 'parantu', or 'bhai sahab'. "
    "3. PREFERRED URDU WORDS: Use words like 'agar' (if), 'ke mutabiq' (according to), 'zaroorat' (need), 'lekin' (but), and 'talib-e-ilm' or 'student' instead. "
    "4. Do NOT answer in pure English or standard Arabic/Urdu script. "
    "5. CRITICAL HALLUCINATION RULE: If the answer is not contained in the provided context, "
    "your ENTIRE response MUST be ONLY this one sentence and absolutely nothing else: "
    "'Maaf kijiye, mujhe university ki policy mein iski maloomat nahi mili.' "
    "Do NOT add 'lekin', 'magar', or any additional information after this sentence. "
    "Do NOT use your own knowledge. Do NOT elaborate. STOP after that sentence. "
    "Adding ANYTHING after the apology sentence is a CRITICAL FAILURE.\n\n"
    "Context:\n{context}"
)

generation_prompt = ChatPromptTemplate.from_messages([
    ("system", GENERATION_SYSTEM_PROMPT),
    ("human", "{input}"),
])


# ─────────────────────────────────────────────
# Main Function: Two-Step Advanced RAG Pipeline
# Returns both the answer AND the English query
# so app.py can display the normalization step
# ─────────────────────────────────────────────

def advanced_rag_query(raw_query: str) -> dict:
    """
    Runs the two-step Advanced RAG pipeline.

    Args:
        raw_query: The raw, messy Roman Urdu query from the user.

    Returns:
        A dict with:
            - "answer":         Final Roman Urdu response for the user.
            - "expanded_query": The English translation used for retrieval.
    """

    # Initialize LLM (shared across both steps)
    llm = ChatGroq(model="llama-3.1-8b-instant", temperature=0)

    # ── Step 1: Normalize the query ──────────────────────────────────────
    normalization_chain = normalization_prompt | llm | StrOutputParser()
    english_query = normalization_chain.invoke({"raw_query": raw_query})
    english_query = english_query.strip()
    print(f"[Normalization] '{raw_query}' → '{english_query}'")

    # ── Step 2: Retrieve using the clean English query ───────────────────
    db = build_or_load_vector_store()
    retriever = db.as_retriever(search_kwargs={"k": 3})

    # Manually invoke retriever with the English translation
    retrieved_docs = retriever.invoke(english_query)

    # ── Step 3: Generate response in Roman Urdu ──────────────────────────
    # IMPORTANT: We pass the ORIGINAL raw_query as `input` so the LLM
    # responds in Roman Urdu, not English.
    question_answer_chain = create_stuff_documents_chain(llm, generation_prompt)

    # Build context string from retrieved docs for the stuff chain
    context_text = "\n\n".join([doc.page_content for doc in retrieved_docs])

    answer = question_answer_chain.invoke({
        "input": raw_query,
        "context": retrieved_docs,
    })

    return {
        "answer": answer,
        "expanded_query": english_query,
    }


# ─────────────────────────────────────────────
# Quick test when running this file directly
# ─────────────────────────────────────────────

if __name__ == "__main__":
    test_cases = [
        "grading policy kia hai aur fail hone pe kya hota hai?",
        "probation pe kab dala jata hai student ko?",
        "HoD se permission leni hoti hai kya?",
    ]

    for query in test_cases:
        print(f"\n{'='*60}")
        print(f"User Query  : {query}")
        result = advanced_rag_query(query)
        print(f"English Used: {result['expanded_query']}")
        print(f"Answer      : {result['answer']}")