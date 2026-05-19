import streamlit as st
import chromadb
from sentence_transformers import SentenceTransformer
from groq import Groq

# -----------------------------------
# PAGE CONFIG
# -----------------------------------

st.set_page_config(
    page_title="NPCI Audit AI",
    layout="wide"
)

# -----------------------------------
# GROQ CLIENT
# -----------------------------------

import streamlit as st
import chromadb
from sentence_transformers import SentenceTransformer
from groq import Groq
from dotenv import load_dotenv
import os

load_dotenv()

client = Groq(
    api_key=st.secrets["GROQ_API_KEY"])


# -----------------------------------
# SIDEBAR
# -----------------------------------

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

# -----------------------------------
# MAIN TITLE
# -----------------------------------

st.title("NPCI / RBI Audit AI")

st.markdown(
    "AI-powered internal audit assistant for NPCI and RBI regulations"
)

# -----------------------------------
# LOAD EMBEDDING MODEL
# -----------------------------------

@st.cache_resource
def load_embedding_model():

    return SentenceTransformer(
        'all-MiniLM-L6-v2'
    )

embedding_model = load_embedding_model()

# -----------------------------------
# LOAD CHROMADB
# -----------------------------------

@st.cache_resource
def load_collection():

    client_db = chromadb.PersistentClient(
        path="../db"
    )

    return client_db.get_collection(
        name="npci_audit"
    )

collection = load_collection()

# -----------------------------------
# USER INPUT
# -----------------------------------

query = st.text_input(
    "Enter Audit Control",
    placeholder="Example: UPI mapper verification"
)

search = st.button("Search")

# -----------------------------------
# SEARCH LOGIC
# -----------------------------------

if search and query:

    with st.spinner("Searching regulations..."):

        # Query embedding
        query_embedding = embedding_model.encode(
            query
        ).tolist()

        # Vector search
        results = collection.query(
            query_embeddings=[query_embedding],
            n_results=2
        )

        documents = results["documents"][0]

        # Trim for speed
        trimmed_docs = [
            doc[:1200]
            for doc in documents
        ]

        context = "\n\n".join(trimmed_docs)

        # -----------------------------------
        # PROMPT
        # -----------------------------------

        prompt = f"""
You are an expert NPCI and RBI internal audit assistant.

STRICT RULES:
- Answer ONLY from provided excerpts.
- Do NOT use outside knowledge.
- Be highly structured.
- Mention circular/policy number wherever possible.
- Mention year if available.
- Quote exact lines from policy.
- Focus on internal audit testing.

USER QUERY:
{query}

REGULATORY EXCERPTS:
{context}

Generate output in EXACT format below:

# Applicable Policy / Circular

Mention:
- Policy/Circular Number
- Year
- Topic

# What the Policy Says

Summarize regulatory expectation in simple audit language.

# Areas to be Tested

Provide specific internal audit testing areas.

# Audit Procedures / Fieldwork

Provide detailed fieldwork steps.

# Evidence Required

Mention exact evidence/documents/logs required.

# Direct Regulatory Quotes

Quote exact lines from the provided excerpts.

Mention:
- Circular Number
- Relevant clause

# Risk if Non-Compliant

Mention operational, regulatory, compliance, fraud, and reputational risks.
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
    # DISPLAY
    # -----------------------------------

    st.success("Relevant regulations identified")

    st.subheader("AI Audit Response")

    st.markdown(answer)

    st.subheader("Retrieved Regulatory Clauses")

    for i, doc in enumerate(trimmed_docs):

        st.info(
            f"Clause {i+1}\n\n{doc}"
        )