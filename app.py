import streamlit as st
from src.advanced_generator import advanced_rag_query

# ─────────────────────────────────────────────
# 1. Page Configuration
# ─────────────────────────────────────────────
st.set_page_config(page_title="GIKI Student Assistant", page_icon="🎓", layout="centered")

st.title("🎓 GIKI Student Assistant")
st.markdown("Mujhse university policies aur rules ke baray mein Roman Urdu mein puchiye!")

# ─────────────────────────────────────────────
# 2. Chat History Initialisation
# Each message now stores three fields:
#   - role:           "user" or "assistant"
#   - content:        the visible text
#   - expanded_query: English translation (assistant messages only)
# ─────────────────────────────────────────────
if "messages" not in st.session_state:
    st.session_state.messages = []

# ─────────────────────────────────────────────
# 3. Replay Full Chat History on Page Reload
# ─────────────────────────────────────────────
for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        # For assistant messages, show the retrieval caption first
        if message["role"] == "assistant" and message.get("expanded_query"):
            st.caption(f"🔍 *Searched database for: {message['expanded_query']}*")
        st.markdown(message["content"])

# ─────────────────────────────────────────────
# 4. Chat Input
# ─────────────────────────────────────────────
user_query = st.chat_input("Apna sawal likhiye (e.g., grading policy kia hai?)...")

if user_query:

    # Display and save the user's message
    with st.chat_message("user"):
        st.markdown(user_query)
    st.session_state.messages.append({
        "role": "user",
        "content": user_query,
        "expanded_query": None,
    })

    # Display the assistant's two-step response
    with st.chat_message("assistant"):
        with st.spinner("Query normalize ho rahi hai aur database search ho raha hai..."):
            result = advanced_rag_query(user_query)
            answer = result["answer"]
            expanded_query = result["expanded_query"]

        # Show the normalised English query as a subtle caption
        st.caption(f"🔍 *Searched database for: {expanded_query}*")

        # Show the final Roman Urdu answer
        st.markdown(answer)

    # Save assistant message WITH expanded_query so history renders correctly
    st.session_state.messages.append({
        "role": "assistant",
        "content": answer,
        "expanded_query": expanded_query,
    })