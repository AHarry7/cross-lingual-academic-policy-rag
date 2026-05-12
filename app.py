import streamlit as st
from src.generator import get_rag_chain
from src.advanced_generator import advanced_rag_query
from src.hybrid_generator import hybrid_rag_query

# ─────────────────────────────────────────────
# 1. Page Configuration
# ─────────────────────────────────────────────
st.set_page_config(page_title="GIKI Student Assistant", page_icon="🎓", layout="centered")

st.title("🎓 GIKI Student Assistant")
st.markdown("Mujhse university policies aur rules ke baray mein Roman Urdu mein puchiye!")


# ─────────────────────────────────────────────
# 2. Sidebar — Pipeline Selector
# ─────────────────────────────────────────────
with st.sidebar:
    st.header("⚙️ Pipeline Settings")
    selected_pipeline = st.radio(
        "Retrieval Pipeline:",
        options=["Baseline", "Advanced", "Hybrid"],
        index=2,                  # Default to Hybrid (most capable)
        help=(
            "**Baseline**: Direct cross-lingual retrieval. Raw Roman Urdu query → ChromaDB.\n\n"
            "**Advanced**: Query normalization + dense retrieval. Roman Urdu → English → ChromaDB.\n\n"
            "**Hybrid**: Query normalization + BM25 + ChromaDB fused via Reciprocal Rank Fusion."
        ),
    )

    st.divider()

    # Pipeline capability badges
    if selected_pipeline == "Baseline":
        st.caption("🟡 No query normalization")
        st.caption("🟡 Dense retrieval only")
        st.caption("🟡 Zero-shot cross-lingual")
    elif selected_pipeline == "Advanced":
        st.caption("🟢 LLM query normalization")
        st.caption("🟡 Dense retrieval only")
        st.caption("🟢 English search query")
    else:
        st.caption("🟢 LLM query normalization")
        st.caption("🟢 BM25 + Dense + RRF fusion")
        st.caption("🟢 English search query")


# ─────────────────────────────────────────────
# 3. Cache baseline chain independently
#    Advanced and Hybrid are stateless functions
#    so they do not need caching here.
# ─────────────────────────────────────────────
@st.cache_resource
def load_baseline_chain():
    return get_rag_chain()


# ─────────────────────────────────────────────
# 4. Chat History — clear when pipeline switches
#    Bleeding history across pipelines would make
#    the ablation demo confusing and misleading.
# ─────────────────────────────────────────────
if "active_pipeline" not in st.session_state:
    st.session_state.active_pipeline = selected_pipeline
    st.session_state.messages = []

if st.session_state.active_pipeline != selected_pipeline:
    st.session_state.active_pipeline = selected_pipeline
    st.session_state.messages = []
    st.rerun()


# ─────────────────────────────────────────────
# 5. Replay Full Chat History
# ─────────────────────────────────────────────
for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        if message["role"] == "assistant" and message.get("expanded_query"):
            st.caption(f"🔍 *Searched database for: {message['expanded_query']}*")
        st.markdown(message["content"])


# ─────────────────────────────────────────────
# 6. Chat Input
# ─────────────────────────────────────────────
user_query = st.chat_input("Apna sawal likhiye (e.g., grading policy kia hai?)...")

if user_query:

    # Display and save user message
    with st.chat_message("user"):
        st.markdown(user_query)
    st.session_state.messages.append({
        "role": "user",
        "content": user_query,
        "expanded_query": None,
    })

    # Run selected pipeline and display response
    with st.chat_message("assistant"):

        if selected_pipeline == "Baseline":
            with st.spinner("Soch raha hoon..."):
                chain = load_baseline_chain()
                response = chain.invoke({"input": user_query})
                answer = response["answer"]
                expanded_query = None
            st.markdown(answer)

        elif selected_pipeline == "Advanced":
            with st.spinner("Query normalize ho rahi hai aur database search ho raha hai..."):
                result = advanced_rag_query(user_query)
                answer = result["answer"]
                expanded_query = result["expanded_query"]
            st.caption(f"🔍 *Searched database for: {expanded_query}*")
            st.markdown(answer)

        else:  # Hybrid
            with st.spinner("BM25 + Dense search aur RRF fusion ho rahi hai..."):
                result = hybrid_rag_query(user_query)
                answer = result["answer"]
                expanded_query = result["expanded_query"]
            st.caption(f"🔍 *Searched database for: {expanded_query}*")
            st.markdown(answer)

    # Save assistant message with expanded_query for history replay
    st.session_state.messages.append({
        "role": "assistant",
        "content": answer,
        "expanded_query": expanded_query,
    })