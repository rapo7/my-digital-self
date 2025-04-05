import streamlit as st
from together import Together
import chromadb
import json
from chromadb.utils.embedding_functions import SentenceTransformerEmbeddingFunction

# Add this at the top of app.py
import os

os.environ["TOKENIZERS_PARALLELISM"] = "false"

DEFAULT_SYSTEM_PROMPT = """You are Ravi, a Software Engineer with a Master's Degree in Computer Science specializing in Machine Learning. 
Your background includes:
- Bachelor's in Electronics and Communication Engineering
- Experience at Oracle and GW Law's Office of Instructional Technology
- Expertise in Python, JavaScript, cloud technologies, and automation
- Strong focus on building efficient developer tools and workflows

Answer questions professionally while maintaining a helpful, knowledgeable tone. 
When possible, provide structured responses with clear sections and bullet points.
"""

# Initialize session state
if "messages" not in st.session_state:
    st.session_state.messages = []
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

# App title
st.title("RaviGPT - RAG with Together AI")

# Display chat messages
for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.write(message["content"])

# Chat input
user_input = st.chat_input("Ask me anything...")
if user_input:
    # Add user message to history and display it
    st.session_state.messages.append({"role": "user", "content": user_input})
    with st.chat_message("user"):
        st.write(user_input)

    # First check for exact question match
    exact_match = None
    results = st.session_state.collection.query(
        query_texts=[user_input],
        n_results=1,
        where={"question": {"$eq": user_input}},  # Correct filter syntax
    )
    print(results)

    if results["documents"] and len(results["documents"][0]) > 0:
        # Extract just the answer portion
        full_doc = results["documents"][0][0]
        if "A: " in full_doc:
            exact_match = full_doc.split("A: ")[1].strip()

    if exact_match:
        # Use exact answer from knowledge base
        assistant_response = exact_match
    else:
        # Fall back to semantic search with top 3 matches
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
                *st.session_state.messages,
                {
                    "role": "system",
                    "content": f"Use this context if relevant:\n{context}",
                },
            ],
        )
        assistant_response = response.choices[0].message.content

    # Store and display assistant response
    st.session_state.messages.append(
        {"role": "assistant", "content": assistant_response}
    )
    with st.chat_message("assistant"):
        st.write(assistant_response)
