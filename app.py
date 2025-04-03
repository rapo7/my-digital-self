import streamlit as st
import time
from typing import List, Dict, Any, Literal
from langchain.embeddings import HuggingFaceEmbeddings
from langchain.vectorstores import FAISS
from langchain.chains import RetrievalQA
from langchain.llms import Together
import os
from dotenv import load_dotenv

# Set page configuration
st.set_page_config(
    page_title="RaviGPT",
    page_icon="✨",
    layout="centered",
    initial_sidebar_state="collapsed"
)

if os.getenv('ENVIRONMENT') == 'development':
    load_dotenv()

# Apply custom CSS for styling
st.markdown("""
<style>
    .stApp {
        background-color: #191622;
        color: white;
    }
    .stButton>button {
        background-color: #2A2438;
        color: white;
        border-radius: 10px;
        border: none;
        padding: 10px 15px;
        transition: background-color 0.3s;
    }
    .stButton>button:hover {
        background-color: #362f4a;
    }
    .stButton>button:active {
        background-color: #7B61FF;
    }
    .active-category {
        background-color: #7B61FF !important;
    }
    div[data-testid="stChatMessage"] {
        background-color: #373440;
        border-radius: 10px;
        padding: 10px;
        margin-bottom: 10px;
    }
    div[data-testid="stChatMessage"][data-user="user"] {
        background-color: #7B61FF;
    }
    .suggestion-btn {
        background-color: rgba(55, 52, 64, 0.5);
        color: white;
        border-radius: 20px;
        border: none;
        padding: 5px 15px;
        margin: 5px;
        cursor: pointer;
        font-size: 14px;
    }
    .suggestion-btn:hover {
        background-color: rgba(123, 97, 255, 0.3);
    }
    div[data-testid="stForm"] {
        background-color: #2A2438;
        border-radius: 10px;
        padding: 10px;
    }
    div[data-testid="stChatInput"] input {
        background-color: #2A2438;
        color: white;
        border-radius: 10px;
    }
    div.row-widget.stRadio > div {
        display: flex;
        flex-direction: row;
    }
    div.row-widget.stRadio > div[role="radiogroup"] > label {
        background-color: #2A2438;
        color: white;
        border-radius: 10px;
        padding: 10px 15px;
        margin-right: 10px;
        cursor: pointer;
    }
    div.row-widget.stRadio > div[role="radiogroup"] > label[data-baseweb="radio"] > div:first-child div {
        background-color: #7B61FF;
    }
</style>
""", unsafe_allow_html=True)

# Define message history in session state if it doesn't exist
if "messages" not in st.session_state:
    st.session_state.messages = []

# Define active category in session state
if "active_category" not in st.session_state:
    st.session_state.active_category = "about"

# Define suggestion categories and questions
suggestion_categories = {
    "about": [
        "What is your educational background?",
        "How did you get started in software engineering?",
        "What programming languages do you know?",
        "What are your strongest technical skills?",
        "Where are you currently working?"
    ],
    "experience": [
        "What companies have you worked for?",
        "What was your most challenging project?",
        "What is your leadership experience?",
        "Tell me about your software engineering experience.",
        "What industries have you worked in?"
    ],
    "projects": [
        "What are your most impressive projects?",
        "Do you have any open source contributions?",
        "What technologies do you use in your projects?",
        "Can you share your GitHub?",
        "What project are you most proud of?"
    ],
    "interests": [
        "What do you like to do outside of work?",
        "What are your hobbies?",
        "What tech topics are you passionate about?",
        "What are you learning right now?",
        "What technology excites you the most?"
    ]
}

# Define category icons (using emojis as we can't use React icons directly)
category_icons = {
    "about": "✨",
    "experience": "💻",
    "projects": "📚",
    "interests": "💡"
}

# Initialize RAG components
if 'vectorstore' not in st.session_state:
    embeddings = HuggingFaceEmbeddings()
    st.session_state.vectorstore = FAISS.from_texts(
        ["Ravi is a software engineer with experience in Python, JavaScript, and React.",
         "Ravi has worked on full-stack applications and is currently learning AI and ML.",
         "Ravi's educational background includes a degree in Computer Science."],
        embeddings
    )

if 'qa_chain' not in st.session_state:
    llm = Together(
        model="togethercomputer/llama-2-7b-chat",
        temperature=0,
        together_api_key=os.getenv('TOGETHER_API_KEY')
    )
    st.session_state.qa_chain = RetrievalQA.from_chain_type(
        llm=llm,
        chain_type="stuff",
        retriever=st.session_state.vectorstore.as_retriever()
    )

def get_ravi_response(user_message: str) -> str:
    """
    This function simulates RaviGPT's response to user questions.
    In a real application, this would connect to a backend service or AI model.
    """
    # Simple response for now - this can be expanded later
    return st.session_state.qa_chain({"question": user_message})

def main():
    # Display header
    st.title("RaviGPT")
    st.subheader("Ask me anything about Ravi, Software Engineer")
    
    # Display chat messages from history
    for message in st.session_state.messages:
        with st.chat_message(message["role"]):
            st.markdown(message["content"])
    
    # If there are no messages, show categories and suggestions
    if not st.session_state.messages:
        # Create category buttons
        cols = st.columns(4)
        categories = list(suggestion_categories.keys())
        
        for i, category in enumerate(categories):
            with cols[i]:
                if st.button(
                    f"{category_icons[category]} {category.capitalize()}", 
                    key=f"btn_{category}",
                    use_container_width=True,
                    type="primary" if st.session_state.active_category == category else "secondary"
                ):
                    st.session_state.active_category = category
                    st.rerun()
        st.write("### Suggested questions:")
        
        for suggestion in suggestion_categories[st.session_state.active_category]:
            suggestion_col = st.columns([1, 6])[1]  # For indentation
            if suggestion_col.button(suggestion, key=f"suggestion_{suggestion}", use_container_width=True):
                st.session_state.messages.append({"role": "user", "content": suggestion})
                # Simulate typing with a spinner
                with st.spinner("RaviGPT is thinking..."):
                    time.sleep(1.5)  # Simulate thinking time
                
                # Add assistant response
                response = get_ravi_response(suggestion)
                st.session_state.messages.append({"role": "assistant", "content": response})
                st.rerun()
    
    # Chat input
    if prompt := st.chat_input("Ask me anything about Ravi..."):
        # Add user message to chat history
        st.session_state.messages.append({"role": "user", "content": prompt})
        
        # Display user message immediately
        with st.chat_message("user"):
            st.markdown(prompt)
        
        # Simulate assistant thinking with a spinner
        with st.spinner("RaviGPT is thinking..."):
            time.sleep(1.5)  # Simulate thinking time
        
        # Display assistant response
        with st.chat_message("assistant"):
            response = get_ravi_response(prompt)
            st.markdown(response)
        
        # Add assistant response to chat history
        st.session_state.messages.append({"role": "assistant", "content": response})

if __name__ == "__main__":
    main()
