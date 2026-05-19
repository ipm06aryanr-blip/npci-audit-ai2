import streamlit as st
from groq import Groq
import os
import fitz
import pytesseract
from PIL import Image
from whoosh.index import create_in, open_dir
from whoosh.fields import Schema, TEXT, ID
from whoosh.qparser import MultifieldParser
import shutil

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

if st.sidebar.button("Rebuild Search Index"):

    if os.path.exists("indexdir"):

        shutil.rmtree("indexdir")

    st.sidebar.success(
        "Index deleted. Refresh app."
    )

# ---------------------------------------------------
# TITLE
# ---------------------------------------------------

st.title("NPCI / RBI Audit AI")

st.markdown(
    "AI-powered internal audit assistant for NPCI and RBI regulations"
)

# ---------------------------------------------------
# PDF TEXT EXTRACTION
# ---------------------------------------------------

def extract_pdf_text(pdf_path):

    text = ""

    try:

        doc = fitz.open(pdf_path)

        for page in doc:

            page_text = page.get_text()

            # ---------------------------------------
            # NORMAL TEXT EXISTS
            # ---------------------------------------

            if page_text.strip():

                text += page_text

            # ---------------------------------------
            # OCR FOR IMAGE PDFs
            # ---------------------------------------

            else:

                pix = page.get_pixmap()

                img_path = "temp_page.png"

                pix.save(img_path)

                image = Image.open(img_path)

                ocr_text = pytesseract.image_to_string(image)

                text += ocr_text

        return text

    except Exception:

        return ""

# ---------------------------------------------------
# CREATE SEARCH INDEX
# ---------------------------------------------------

@st.cache_resource
def create_search_index():

    schema = Schema(
        title=ID(stored=True),
        content=TEXT(stored=True)
    )

    # -----------------------------------------------
    # CREATE INDEX
    # -----------------------------------------------

    if not os.path.exists("indexdir"):

        os.mkdir("indexdir")

        ix = create_in(
            "indexdir",
            schema
        )

        writer = ix.writer()

        folder = "documents"

        files = os.listdir(folder)

        for file in files:

            if file.endswith(".pdf"):

                file_path = os.path.join(
                    folder,
                    file
                )

                st.write(
                    f"Processing: {file}"
                )

                extracted_text = extract_pdf_text(
                    file_path
                )

                if len(extracted_text.strip()) > 100:

                    writer.add_document(
                        title=file,
                        content=extracted_text
                    )

        writer.commit()

    # -----------------------------------------------
    # LOAD INDEX
    # -----------------------------------------------

    return open_dir("indexdir")

# ---------------------------------------------------
# LOAD INDEX
# ---------------------------------------------------

ix = create_search_index()

# ---------------------------------------------------
# USER QUERY
# ---------------------------------------------------

query = st.text_input(
    "Enter Audit Control"
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
                    f"""
CIRCULAR:
{r['title']}

REGULATORY EXCERPT:
{r['content'][:4000]}
"""
                )

        # -------------------------------------------
        # NO RESULTS
        # -------------------------------------------

        if len(documents) == 0:

            st.error(
                "No relevant circular found."
            )

            st.stop()

        # -------------------------------------------
        # SHOW RETRIEVED DOCUMENTS
        # -------------------------------------------

        st.subheader(
            "Retrieved Regulatory Clauses"
        )

        for i, doc in enumerate(documents):

            st.info(
                f"Clause {i+1}\n\n{doc}"
            )

        # -------------------------------------------
        # CONTEXT
        # -------------------------------------------

        context = "\n\n".join(documents)

        # -------------------------------------------
        # PROMPT
        # -------------------------------------------

        prompt = f"""
You are an NPCI audit assistant.

STRICT RULES:

1. ONLY use retrieved excerpts.
2. NEVER hallucinate circulars.
3. NEVER mention RBI unless explicitly present.
4. If info missing, say:
   "Not explicitly mentioned."

USER QUERY:
{query}

RETRIEVED EXCERPTS:
{context}

Generate:

# Retrieved Circulars

# Regulatory Requirement

# Audit Testing Areas

# Evidence Required

# Exact Regulatory Quotes

# Compliance Risks

# Missing Information
"""

        # -------------------------------------------
        # GROQ RESPONSE
        # -------------------------------------------

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

    # ------------------------------------------------
    # OUTPUT
    # ------------------------------------------------

    st.success(
        "Relevant regulations identified"
    )

    st.subheader(
        "AI Audit Response"
    )

    st.markdown(answer)