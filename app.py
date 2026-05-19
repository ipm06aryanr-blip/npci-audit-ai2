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

st.sidebar.write("• UPI-OC-170")
st.sidebar.write("• UPI-OC-171")
st.sidebar.write("• UPI-OC-172")

# ---------------------------------------------------
# MAIN TITLE
# ---------------------------------------------------

st.title("NPCI / RBI Audit AI")

st.markdown(
    "AI-powered internal audit assistant for NPCI and RBI regulations"
)

# ---------------------------------------------------
# LOAD EMBEDDING MODEL
# ---------------------------------------------------

@st.cache_resource
def load_embedding_model():

    model = SentenceTransformer(
        "all-MiniLM-L6-v2"
    )

    return model

embedding_model = load_embedding_model()

# ---------------------------------------------------
# LOAD / CREATE CHROMADB
# ---------------------------------------------------

@st.cache_resource
def load_collection():

    client_db = chromadb.PersistentClient(
        path="./db"
    )

    collection_name = "npci_audit"

    existing_collections = client_db.list_collections()

    existing_names = [
        col.name
        for col in existing_collections
    ]

    # ---------------------------------------------------
    # IF COLLECTION EXISTS
    # ---------------------------------------------------

    if collection_name in existing_names:

        collection = client_db.get_collection(
            name=collection_name
        )

        return collection

    # ---------------------------------------------------
    # CREATE COLLECTION
    # ---------------------------------------------------

    collection = client_db.create_collection(
        name=collection_name
    )

    folder_path = "chunks"

    documents = []
    ids = []

    # ---------------------------------------------------
    # READ ALL CHUNK FILES
    # ---------------------------------------------------

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

                        # Skip empty docs
                        if len(text.strip()) > 50:

                            documents.append(
                                text[:4000]
                            )

                            ids.append(
                                f"doc_{counter}"
                            )

                            counter += 1

                except Exception as e:

                    st.warning(
                        f"Could not read {file}"
                    )

    # ---------------------------------------------------
    # CREATE EMBEDDINGS
    # ---------------------------------------------------

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
# LOAD COLLECTION
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

        # ---------------------------------------------------
        # QUERY EMBEDDING
        # ---------------------------------------------------

        query_embedding = embedding_model.encode(
            query
        ).tolist()

        # ---------------------------------------------------
        # VECTOR SEARCH
        # ---------------------------------------------------

        results = collection.query(
            query_embeddings=[query_embedding],
            n_results=5
        )

        documents = results["documents"][0]

        # ---------------------------------------------------
        # TRIM DOCUMENTS
        # ---------------------------------------------------

        trimmed_docs = []

        for doc in documents:

            trimmed_docs.append(
                doc[:1500]
            )

        context = "\n\n".join(trimmed_docs)

        # ---------------------------------------------------
        # PROMPT
        # ---------------------------------------------------

        prompt = f"""
You are an expert NPCI and RBI internal audit assistant.

STRICT RULES:
- Answer ONLY from provided excerpts
- Do NOT hallucinate
- Mention circular number wherever possible
- Mention policy year wherever possible
- Focus on internal audit testing
- Be detailed and structured
- Quote relevant policy lines
- Mention evidence required

USER QUERY:
{query}

REGULATORY EXCERPTS:
{context}

Generate response in EXACT structure below:

# Applicable Policy / Circular

Mention:
- Circular Number
- Year
- Topic

# What the Policy Says

Summarize the regulatory expectation.

# Areas to be Tested

Mention detailed audit testing areas.

# Audit Procedures / Fieldwork

Provide audit procedures.

# Evidence Required

Mention logs, reports, screenshots, approvals,
configurations, monitoring evidence, etc.

# Direct Regulatory Quotes

Quote exact relevant lines.

# Risk if Non-Compliant

Mention:
- Regulatory risk
- Fraud risk
- Compliance risk
- Operational risk
- Reputational risk
"""

        # ---------------------------------------------------
        # GROQ RESPONSE
        # ---------------------------------------------------

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

    # ---------------------------------------------------
    # DISPLAY RESULTS
    # ---------------------------------------------------

    st.success(
        "Relevant regulations identified"
    )

    st.subheader("AI Audit Response")

    st.markdown(answer)

    st.subheader(
        "Retrieved Regulatory Clauses"
    )

    for i, doc in enumerate(trimmed_docs):

        st.info(
            f"Clause {i+1}\n\n{doc}"
        )