import os

from dotenv import load_dotenv

from fastembed import TextEmbedding

from langchain_text_splitters import RecursiveCharacterTextSplitter

from langchain_community.vectorstores import FAISS

from langchain_core.embeddings import Embeddings

from langchain_groq import ChatGroq


load_dotenv()


# ==========================================
# FastEmbed Wrapper
# ==========================================

class FastEmbedEmbeddings(Embeddings):

    def __init__(self):

        self.model = TextEmbedding(
            model_name="sentence-transformers/all-MiniLM-L6-v2"
        )


    def embed_documents(self, texts):

        return list(
            self.model.embed(texts)
        )


    def embed_query(self, text):

        return list(
            self.model.embed([text])
        )[0]


# ==========================================
# Initialize Embeddings
# ==========================================

embeddings = FastEmbedEmbeddings()


# ==========================================
# Vector Database
# ==========================================

vector_db = None


# ==========================================
# Upload Logs to FAISS
# ==========================================

def upload_logs(logs: str):

    global vector_db

    splitter = RecursiveCharacterTextSplitter(
        chunk_size=300,
        chunk_overlap=50
    )

    chunks = splitter.split_text(logs)

    if not chunks:
        return 0

    vector_db = FAISS.from_texts(
        texts=chunks,
        embedding=embeddings
    )

    return len(chunks)


# ==========================================
# Analyze Logs using RAG
# ==========================================

def analyze_logs(query: str, k: int = 5):

    global vector_db

    if vector_db is None:

        return None


    # --------------------------------------
    # Retrieve relevant chunks
    # --------------------------------------

    docs = vector_db.similarity_search(
        query,
        k=k
    )


    context = "\n\n".join(
        doc.page_content
        for doc in docs
    )


    # --------------------------------------
    # Groq LLM
    # --------------------------------------

    llm = ChatGroq(

        model="openai/gpt-oss-20b",

        temperature=0.2,

        api_key=os.getenv("GROQ_API_KEY")
    )


    # --------------------------------------
    # RCA Prompt
    # --------------------------------------

    prompt = f"""
You are an expert DevOps, Site Reliability Engineer (SRE),
and Root Cause Analysis specialist.

Analyze the provided system logs carefully.

IMPORTANT RULES:

- Use ONLY the evidence available in the logs.
- Do NOT invent errors that are not present.
- If there is insufficient evidence, clearly mention it.
- Identify patterns, failures, warnings, and possible dependencies.
- Focus on determining the most probable root cause.

LOGS:

{context}


Provide the analysis in exactly this format:

ROOT CAUSE:
Explain the most probable root cause.

SEVERITY:
Low / Medium / High / Critical

ESTIMATED IMPACT:
Estimate the possible system impact percentage.

EVIDENCE:
Mention the important log messages supporting the conclusion.

FAILURE POINT:
Explain where the failure most likely occurred.

RECOMMENDED FIX:
Provide clear step-by-step resolution actions.

PREVENTION:
Provide preventive measures to avoid this issue again.
"""


    response = llm.invoke(prompt)


    return {

        "answer": response.content,

        "retrieved_chunks": [

            {
                "rank": index + 1,

                "chunk": doc.page_content

            }

            for index, doc in enumerate(docs)

        ]

    }


# ==========================================
# Complete Other Logs Pipeline
# ==========================================

def analyze_other_logs(logs: str):

    # Step 1: Store logs in FAISS

    chunks = upload_logs(logs)


    if chunks == 0:

        return {

            "success": False,

            "message": "No logs were provided.",

            "chunks": 0

        }


    # Step 2: Automatically perform RCA

    query = """
Analyze these logs and perform a complete Root Cause Analysis.
Identify the root cause, severity, evidence, estimated impact,
failure point, recommended fix, and preventive measures.
"""


    result = analyze_logs(
        query=query,
        k=5
    )


    return {

        "success": True,

        "chunks": chunks,

        "rca": result["answer"],

        "retrieved_chunks": result["retrieved_chunks"]

    }