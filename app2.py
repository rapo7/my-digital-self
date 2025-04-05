import streamlit as st
from together import Together
import chromadb
import json
from chromadb.utils.embedding_functions import SentenceTransformerEmbeddingFunction
import os

# Disable tokenizer parallelism warnings
os.environ["TOKENIZERS_PARALLELISM"] = "false"

# Default system prompt
DEFAULT_SYSTEM_PROMPT = """I am Ravi, a Software Engineer with a Master's Degree in Computer Science specializing in Machine Learning. 
My background includes:
- Bachelor's in Electronics and Communication Engineering
- Experience at Oracle and GW Law's Office of Instructional Technology
- Expertise in Python, JavaScript, cloud technologies, and automation
- Strong focus on building efficient developer tools and workflows

and You are a digital version of me so try to Answer questions professionally while maintaining a helpful, knowledgeable tone. 
When possible, provide structured responses with clear sections and bullet points.
"""

# Set page configuration
st.set_page_config(page_title="How can I help you?", page_icon="🤖", layout="centered")

# Custom CSS for styling
st.markdown(
    """
    <style>
    body {
        background-color: #1e1e2f;
        color: #ffffff;
    }
    .title {
        font-size: 2.5rem;
        font-weight: bold;
        text-align: center;
        margin-bottom: 20px;
    }
    .category-container {
        display: flex;
        justify-content: center;
        flex-wrap: wrap;
        gap: 10px;
        margin-bottom: 20px;
    }
    .category-button {
        background-color: #2c2c3e;
        color: #ffffff;
        border: none;
        padding: 10px 20px;
        border-radius: 5px;
        cursor: pointer;
        font-size: 1rem;
    }
    .category-button:hover, .category-button.active {
        background-color: #44445a;
    }
    .question-container {
        display: flex;
        flex-direction: column;
        gap: 10px;
        margin: 20px 0;
    }
    .question-button {
        background-color: #2c2c3e;
        color: #ffffff;
        border: none;
        padding: 10px 20px;
        border-radius: 5px;
        cursor: pointer;
        font-size: 1rem;
        text-align: left;
    }
    .question-button:hover {
        background-color: #44445a;
    }
    .chat-container {
        margin-top: 30px;
    }
    .stTextInput > div > div > input {
        background-color: #2c2c3e;
        color: white;
    }
    </style>
    """,
    unsafe_allow_html=True,
)

# Title
st.markdown('<div class="title">Ravi\'s Digital Self</div>', unsafe_allow_html=True)

# Initialize session state
if "messages" not in st.session_state:
    st.session_state.messages = []
if "selected_category" not in st.session_state:
    st.session_state.selected_category = "Basic"
if "user_input" not in st.session_state:
    st.session_state.user_input = ""
if "chroma_client" not in st.session_state:
    # Initialize ChromaDB client
    st.session_state.chroma_client = chromadb.Client()

    # Create or get collection
    embedding_function = SentenceTransformerEmbeddingFunction(
        model_name="all-MiniLM-L6-v2"
    )
    st.session_state.collection = st.session_state.chroma_client.create_collection(
        name="knowledge_base", embedding_function=embedding_function
    )

    # Load knowledge base
    try:
        with open("knowledge_base.jsonl", "r") as f:
            for idx, line in enumerate(f):
                data = json.loads(line)
                # Store both question and answer as documents
                question = data["messages"][0]["content"]
                answer = data["messages"][1]["content"]
                st.session_state.collection.add(
                    documents=[f"Q: {question}\nA: {answer}"],
                    metadatas=[{"question": question}],
                    ids=[str(idx)],
                )
    except FileNotFoundError:
        st.warning("knowledge_base.jsonl not found")

# Initialize Together client
together_client = Together(api_key=st.secrets["TOGETHER_API_KEY"])

# Categories and questions
categories = {
    "Basic": [
        "What is your educational background?",
        "How did you get started in software engineering?",
        "What programming languages do you know?",
        "What are your strongest technical skills?",
        "How to contact you?",
    ],
    "Work": [
        "Where are you currently working?",
        "What companies have you worked for?",
        "What was your most challenging project?",
        "What was your Current project?",
        "What is your leadership experience?",
    ],
    "Skills": [
        "Tell me about your software engineering experience.",
        "What industries have you worked in?",
        "What are your most impressive projects?",
        "Do you have any open source contributions?",
        "What technologies do you use in your projects?",
    ],
    "Hobbies": [
        "What are your hobbies?",
        "What do you like to do outside of work?",
        "What project are you most proud of?",
        "What are you learning right now?",
        "Can you share your GitHub?",
    ],
}

# Function to select category
def select_category(category):
    st.session_state.selected_category = category

# Function to set question and trigger chat
def handle_question_click(question):
    # Add user message to history
    st.session_state.messages.append({"role": "user", "content": question})
    
    # Perform semantic search with top 3 matches
    similar_results = st.session_state.collection.query(
        query_texts=[question], n_results=3
    )
    context = (
        "\n".join([doc for doc in similar_results["documents"][0] if "A: " in doc])
        if similar_results["documents"]
        else ""
    )

    response = together_client.chat.completions.create(
        model="meta-llama/Meta-Llama-3-8B-Instruct-Turbo",
        messages=[
            {"role": "system", "content": DEFAULT_SYSTEM_PROMPT},
            {"role": "user", "content": question},
            {
                "role": "system",
                "content": f"Please use the following information to craft a helpful and accurate response:\n{context}"
                if context
                else f"No additional information is available. Please provide a concise and professional response.",
            },
        ],
    )
    assistant_response = response.choices[0].message.content

    # Store assistant response
    st.session_state.messages.append(
        {"role": "assistant", "content": assistant_response}
    )
    
    # Rerun to update the UI
    st.rerun()

chat_input = st.chat_input("Type your message here...")

# And update the chat input handler similarly:
if user_input := chat_input:
    # Store last input to prevent duplicate submissions
    st.session_state.last_input = user_input
    
    # Add user message to history
    st.session_state.messages.append({"role": "user", "content": user_input})
    
    # Perform semantic search with top 3 matches
    similar_results = st.session_state.collection.query(
        query_texts=[user_input], n_results=3
    )
    context = (
        "\n".join([doc for doc in similar_results["documents"][0] if "A: " in doc])
        if similar_results["documents"]
        else ""
    )

    response = together_client.chat.completions.create(
        model="meta-llama/Meta-Llama-3-8B-Instruct-Turbo",
        messages=[
            {"role": "system", "content": DEFAULT_SYSTEM_PROMPT},
            {"role": "user", "content": user_input},
            {
                "role": "system",
                "content": f"Please use the following information to craft a helpful and accurate response:\n{context}"
                if context
                else f"No additional information is available. Please provide a concise and professional response.",
            },
        ],
    )
    assistant_response = response.choices[0].message.content

    # Store assistant response
    st.session_state.messages.append(
        {"role": "assistant", "content": assistant_response}
    )
    
    # Clear input after submission
    st.session_state.user_input = ""
    
    # Rerun to update the UI
    st.rerun()
# Category buttons
st.markdown('<div class="category-container">', unsafe_allow_html=True)
cols = st.columns(len(categories))
for i, category in enumerate(categories):
    with cols[i]:
        active_class = "active" if st.session_state.selected_category == category else ""
        if st.button(category, key=f"cat_{category}", 
                    use_container_width=True):
            select_category(category)
st.markdown('</div>', unsafe_allow_html=True)

# Display questions for selected category
st.markdown('<div class="question-container">', unsafe_allow_html=True)
for question in categories[st.session_state.selected_category]:
    if st.button(question, key=f"q_{question}", use_container_width=True):
        handle_question_click(question)
st.markdown('</div>', unsafe_allow_html=True)

# Display chat messages
st.markdown('<div class="chat-container">', unsafe_allow_html=True)
for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.write(message["content"])
st.markdown('</div>', unsafe_allow_html=True)

# Single chat input at the bottom
if user_input := chat_input:
    # Store last input to prevent duplicate submissions
    st.session_state.last_input = user_input
    
    # Add user message to history
    st.session_state.messages.append({"role": "user", "content": user_input})
    
    # Perform semantic search with top 3 matches
    similar_results = st.session_state.collection.query(
        query_texts=[user_input], n_results=3
    )
    context = (
        "\n".join([doc for doc in similar_results["documents"][0] if "A: " in doc])
        if similar_results["documents"]
        else ""
    )

    # Generate response using LLM
    response = together_client.chat.completions.create(
        model="meta-llama/Meta-Llama-3-8B-Instruct-Turbo",
        messages=[
            {"role": "system", "content": DEFAULT_SYSTEM_PROMPT},
            {"role": "user", "content": user_input},
            {
                "role": "system",
                "content": f"Please use the following information to craft a helpful and accurate response:\n{context}"
                if context
                else f"No additional information is available. Please provide a concise and professional response.",
            },
        ],
    )
    assistant_response = response.choices[0].message.content

    # Store assistant response
    st.session_state.messages.append(
        {"role": "assistant", "content": assistant_response}
    )
    
    # Rerun to update the UI
    st.rerun()
