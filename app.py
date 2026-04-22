import streamlit as st
from src.generator import get_rag_chain

# 1. Configure the Web Page
st.set_page_config(page_title="GIKI Student Assistant", page_icon="🎓", layout="centered")

st.title("🎓 GIKI Student Assistant")
st.markdown("Mujhse university policies aur rules ke baray mein Roman Urdu mein puchiye!")

# 2. Load the backend logic (Cached so it doesn't reload the models every time)
@st.cache_resource
def load_chain():
    return get_rag_chain()

rag_chain = load_chain()

# 3. Setup Chat History
if "messages" not in st.session_state:
    st.session_state.messages = []

# Display previous messages when the page reloads
for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

# 4. The Chat Input box
user_query = st.chat_input("Apna sawal likhiye (e.g., grading policy kia hai?)...")

if user_query:
    # Display the user's message on the screen
    with st.chat_message("user"):
        st.markdown(user_query)
    
    # Save it to history
    st.session_state.messages.append({"role": "user", "content": user_query})

    # Display the Assistant's thinking process and answer
    with st.chat_message("assistant"):
        with st.spinner("Soch raha hoon..."):
            # Call your LLM backend
            response = rag_chain.invoke({"input": user_query})
            answer = response["answer"]
            st.markdown(answer)
            
    # Save the assistant's answer to history
    st.session_state.messages.append({"role": "assistant", "content": answer})