import streamlit as st
import chromadb
from sentence_transformers import SentenceTransformer
from groq import Groq
import os

# ---------------------------------------------------
# PAGE CONFIG
# ---------------------------------------------------

st.set_page_config(
    page_title="NPCI Audit AI",
    layout="wide"
)

# ---------------------------------------------------
# GROQ CLIENT
# ---------------------------------------------------

client = Groq(
    api_key=st.secrets["GROQ_API_KEY"]
)

# ---------------------------------------------------
# SIDEBAR
# ---------------------------------------------------

st.sidebar.title("NPCI Audit AI")

domain = st.sidebar.selectbox(
    "Select Domain",
    [
        "UPI",
        "TPAP",
        "Mapper",
        "Fraud Monitoring",
        "Cybersecurity",
        "BBPS",
        "PPI"
    ]
)

st.sidebar.markdown("---")

st.sidebar.subheader("Latest Circulars")

st.sidebar.write("• Dynamic retrieval enabled")

# ---------------------------------------------------
# MAIN TITLE
# ---------------------------------------------------

st.title("NPCI / RBI Audit AI")

st.markdown(
    "AI-powered internal audit assistant for NPCI and RBI regulations"
)

# ---------------------------------------------------
# EMBEDDING MODEL
# ---------------------------------------------------

@st.cache_resource
def load_embedding_model():

    return SentenceTransformer(
        "all-MiniLM-L6-v2"
    )

embedding_model = load_embedding_model()

# ---------------------------------------------------
# LOAD COLLECTION
# ---------------------------------------------------

@st.cache_resource
def load_collection():

    client_db = chromadb.PersistentClient(
        path="./db"
    )

    collection_name = "npci_audit"

    existing = [
        col.name
        for col in client_db.list_collections()
    ]

    # -----------------------------------
    # LOAD EXISTING COLLECTION
    # -----------------------------------

    if collection_name in existing:

        return client_db.get_collection(
            name=collection_name
        )

    # -----------------------------------
    # CREATE NEW COLLECTION
    # -----------------------------------

    collection = client_db.create_collection(
        name=collection_name
    )

    folder_path = "chunks"

    documents = []
    ids = []

    if os.path.exists(folder_path):

        files = os.listdir(folder_path)

        counter = 0

        for file in files:

            if file.endswith(".txt"):

                file_path = os.path.join(
                    folder_path,
                    file
                )

                try:

                    with open(
                        file_path,
                        "r",
                        encoding="utf-8"
                    ) as f:

                        text = f.read()

                    if len(text.strip()) > 50:

                        circular_name = file.replace(
                            ".txt",
                            ""
                        )

                        structured_text = f"""
CIRCULAR: {circular_name}

SOURCE FILE: {file}

REGULATORY EXCERPT:
{text[:1200]}
"""

                        documents.append(
                            structured_text
                        )

                        ids.append(
                            f"doc_{counter}"
                        )

                        counter += 1

                except Exception:

                    pass

    # -----------------------------------
    # CREATE EMBEDDINGS
    # -----------------------------------

    if len(documents) > 0:

        embeddings = embedding_model.encode(
            documents
        ).tolist()

        collection.add(
            documents=documents,
            embeddings=embeddings,
            ids=ids
        )

    return collection

# ---------------------------------------------------
# INITIALIZE COLLECTION
# ---------------------------------------------------

collection = load_collection()

# ---------------------------------------------------
# USER INPUT
# ---------------------------------------------------

query = st.text_input(
    "Enter Audit Control",
    placeholder="Example: UPI mapper verification"
)

search = st.button("Search")

# ---------------------------------------------------
# SEARCH LOGIC
# ---------------------------------------------------

if search and query:

    with st.spinner("Searching regulations..."):

        query_embedding = embedding_model.encode(
            query
        ).tolist()

        results = collection.query(
            query_embeddings=[query_embedding],
            n_results=2
        )

        documents = results["documents"][0]
        st.write("DEBUG RETRIEVED DOCUMENTS")
        st.write(documents)

        trimmed_docs = []

        for doc in documents:

            trimmed_docs.append(
                doc[:1500]
            )

        context = "\n\n".join(trimmed_docs)

        # -----------------------------------
        # PROMPT
        # -----------------------------------

        prompt = f"""
You are an NPCI regulatory audit assistant.

STRICT RULES:

1. Use ONLY information explicitly present in retrieved excerpts.
2. NEVER mention RBI policies unless explicitly written.
3. NEVER invent circular numbers.
4. NEVER invent years.
5. NEVER infer missing controls.
6. If information is unavailable, say:
   "Not explicitly mentioned in retrieved excerpts."

USER QUERY:
{query}

RETRIEVED EXCERPTS:
{context}

TASK:

1. Identify exact circulars retrieved.
2. Explain ONLY what retrieved excerpts say.
3. Mention audit testing areas ONLY from retrieved excerpts.
4. Quote exact lines wherever possible.
5. Mention evidence required.
6. Mention missing information if unavailable.

FORMAT:

# Retrieved Circulars

# Regulatory Requirement

# Audit Testing Areas

# Evidence Required

# Exact Regulatory Quotes

# Missing Information / Gaps
"""

        # -----------------------------------
        # GROQ RESPONSE
        # -----------------------------------

        response = client.chat.completions.create(
            model="llama-3.1-8b-instant",
            messages=[
                {
                    "role": "user",
                    "content": prompt
                }
            ]
        )

        answer = response.choices[0].message.content

    # -----------------------------------
    # DISPLAY RETRIEVED CLAUSES FIRST
    # -----------------------------------

    st.subheader(
        "Retrieved Regulatory Clauses"
    )

    for i, doc in enumerate(trimmed_docs):

        st.info(
            f"Clause {i+1}\n\n{doc}"
        )

    # -----------------------------------
    # DISPLAY AI RESPONSE
    # -----------------------------------

    st.success(
        "Relevant regulations identified"
    )

    st.subheader("AI Audit Response")

    st.markdown(answer)