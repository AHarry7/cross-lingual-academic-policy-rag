import os
from dotenv import load_dotenv
from langchain_groq import ChatGroq
from langchain_core.prompts import ChatPromptTemplate
from langchain_classic.chains import create_retrieval_chain
from langchain_classic.chains.combine_documents import create_stuff_documents_chain
from src.retriever import build_or_load_vector_store

# Load the Groq API key from your .env file
load_dotenv()

def get_rag_chain():
    # 1. Initialize the LLM (Using Llama-3 via Groq for extreme speed)
    # temperature=0 ensures the model doesn't hallucinate or get overly creative
    llm = ChatGroq(
        model="llama-3.1-8b-instant", 
        temperature=0
    )

    # 2. Get the Retriever from our previous script
    db = build_or_load_vector_store()
    # We tell it to fetch the top 3 most relevant chunks
    retriever = db.as_retriever(search_kwargs={"k": 3})

    # 3. Create the System Prompt (This is where the magic happens)
    system_prompt = (
        "You are a helpful, professional, and formal university assistant at GIKI. "
        "Use the following retrieved context to answer the user's question. "
        "IMPORTANT RULES: "
        "1. You MUST answer the question in polite, standard Pakistani Roman Urdu. "
        "2. STRICTLY PROHIBITED HINDI WORDS: Do NOT use words like 'yadi', 'anusar', 'avashyakta', 'kintu', 'parantu', or 'bhai sahab'. "
        "3. PREFERRED URDU WORDS: Use words like 'agar' (if), 'ke mutabiq' (according to), 'zaroorat' (need), 'lekin' (but), and 'talib-e-ilm' or 'student' instead. "
        "4. Do NOT answer in pure English or standard Arabic/Urdu script. "
        "5. If the answer is not contained in the provided context, DO NOT guess. "
        "Just say: 'Maaf kijiye, mujhe university ki policy mein iski maloomat nahi mili' (Sorry, I couldn't find this info).\n\n"
        "Context:\n{context}"
    )

    prompt = ChatPromptTemplate.from_messages([
        ("system", system_prompt),
        ("human", "{input}"),
    ])

    # 4. Chain it all together
    question_answer_chain = create_stuff_documents_chain(llm, prompt)
    rag_chain = create_retrieval_chain(retriever, question_answer_chain)

    return rag_chain

# Let's test the full pipeline!
if __name__ == "__main__":
    print("Initializing the RAG Chain...")
    rag_chain = get_rag_chain()
    
    # You can change this to test different questions
    test_question = "grading policy kia hai aur fail hone pe kya hota hai?"
    
    print(f"\nUser: {test_question}")
    print("Thinking...\n")
    
    # Run the chain
    response = rag_chain.invoke({"input": test_question})
    
    print(f"Assistant: {response['answer']}")