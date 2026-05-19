import streamlit as st
from groq import Groq
import os
import shutil

from whoosh.index import create_in, open_dir
from whoosh.fields import Schema, TEXT, ID
from whoosh.qparser import MultifieldParser

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
        "PPI",
        "KYC",
        "Cashback"
    ]
)

st.sidebar.markdown("---")

st.sidebar.success(
    "BM25 Retrieval Enabled"
)

# ---------------------------------------------------
# REBUILD INDEX BUTTON
# ---------------------------------------------------

if st.sidebar.button("Rebuild Search Index"):

    if os.path.exists("indexdir"):

        shutil.rmtree("indexdir")

    st.sidebar.success(
        "Search index deleted. Refresh app."
    )

# ---------------------------------------------------
# MAIN TITLE
# ---------------------------------------------------

st.title("NPCI / RBI Audit AI")

st.markdown(
    "AI-powered internal audit assistant for NPCI and RBI regulations"
)

# ---------------------------------------------------
# CREATE / LOAD SEARCH INDEX
# ---------------------------------------------------

@st.cache_resource
def create_search_index():

    schema = Schema(
        title=ID(stored=True),
        content=TEXT(stored=True)
    )

    # ---------------------------------------------------
    # CREATE INDEX
    # ---------------------------------------------------

    if not os.path.exists("indexdir"):

        os.mkdir("indexdir")

        ix = create_in(
            "indexdir",
            schema
        )

        writer = ix.writer()

        chunk_folder = "chunks"

        if os.path.exists(chunk_folder):

            files = os.listdir(chunk_folder)

            for file in files:

                if file.endswith(".txt"):

                    file_path = os.path.join(
                        chunk_folder,
                        file
                    )

                    try:

                        with open(
                            file_path,
                            "r",
                            encoding="utf-8"
                        ) as f:

                            text = f.read()

                        # -----------------------------------
                        # TOPIC DETECTION
                        # -----------------------------------

                        filename_lower = file.lower()

                        topic = "General"

                        if "kyc" in filename_lower:
                            topic = "KYC"

                        elif "cashback" in filename_lower:
                            topic = "Cashback"

                        elif "fraud" in filename_lower:
                            topic = "Fraud Monitoring"

                        elif "tpap" in filename_lower:
                            topic = "TPAP"

                        elif "mapper" in filename_lower:
                            topic = "UPI Mapper"

                        elif "microatm" in filename_lower:
                            topic = "MicroATM"

                        elif "upi" in filename_lower:
                            topic = "UPI"

                        structured_text = f"""
TOPIC: {topic}

CIRCULAR: {file}

REGULATORY EXCERPT:
{text}
"""

                        writer.add_document(
                            title=file,
                            content=structured_text
                        )

                    except Exception:

                        pass

        writer.commit()

    # ---------------------------------------------------
    # LOAD INDEX
    # ---------------------------------------------------

    return open_dir("indexdir")

# ---------------------------------------------------
# INITIALIZE INDEX
# ---------------------------------------------------

ix = create_search_index()

# ---------------------------------------------------
# USER INPUT
# ---------------------------------------------------

query = st.text_input(
    "Enter Audit Control",
    placeholder="Example: KYC, Cashback regulation, TPAP market share"
)

search = st.button("Search")

# ---------------------------------------------------
# SEARCH LOGIC
# ---------------------------------------------------

if search and query:

    with st.spinner("Searching regulations..."):

        documents = []

        with ix.searcher() as searcher:

            parser = MultifieldParser(
                ["content", "title"],
                schema=ix.schema
            )

            myquery = parser.parse(query)

            results = searcher.search(
                myquery,
                limit=5
            )

            for r in results:

                documents.append(
                    r["content"]
                )

        # ---------------------------------------------------
        # NO RESULTS
        # ---------------------------------------------------

        if len(documents) == 0:

            st.error(
                "No relevant regulatory circular found."
            )

            st.stop()

        # ---------------------------------------------------
        # SHOW RETRIEVED CLAUSES
        # ---------------------------------------------------

        st.subheader(
            "Retrieved Regulatory Clauses"
        )

        trimmed_docs = []

        for i, doc in enumerate(documents):

            trimmed_doc = doc[:2000]

            trimmed_docs.append(
                trimmed_doc
            )

            st.info(
                f"Clause {i+1}\n\n{trimmed_doc}"
            )

        # ---------------------------------------------------
        # CREATE CONTEXT
        # ---------------------------------------------------

        context = "\n\n".join(trimmed_docs)

        # ---------------------------------------------------
        # PROMPT
        # ---------------------------------------------------

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
    # FINAL OUTPUT
    # ---------------------------------------------------

    st.success(
        "Relevant regulations identified"
    )

    st.subheader(
        "AI Audit Response"
    )

    st.markdown(answer)