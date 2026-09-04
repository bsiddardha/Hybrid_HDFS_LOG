from fastapi import FastAPI

from fastapi.responses import FileResponse

from pydantic import BaseModel

from rag import analyze_other_logs

from hdfs.hdfs_service import analyze_hdfs_log


# ==========================================
# FastAPI Application
# ==========================================

app = FastAPI(
    title="LogSense AI - Intelligent Log Analyzer"
)


# ==========================================
# Request Model
# ==========================================

class AnalyzeRequest(BaseModel):

    log_type: str

    log: str


# ==========================================
# Home Page
# ==========================================

@app.get("/")
def home():

    return FileResponse(
        "frontend/index.html"
    )


# ==========================================
# Health Check
# ==========================================

@app.get("/health")
def health():

    return {

        "status": "healthy",

        "service": "LogSense AI"

    }


# ==========================================
# Main Analysis API
# ==========================================

@app.post("/analyze")
def analyze(request: AnalyzeRequest):

    log_type = request.log_type.lower().strip()

    log = request.log.strip()


    # --------------------------------------
    # Validate Input
    # --------------------------------------

    if not log:

        return {

            "success": False,

            "message": "Please provide log data."

        }


    # ======================================
    # HDFS PIPELINE
    # ======================================

    if log_type == "hdfs":

        result = analyze_hdfs_log(log)

        return {

            "pipeline": "HDFS Transformer",

            **result

        }


    # ======================================
    # OTHER LOGS PIPELINE
    # ======================================

    elif log_type == "other":

        result = analyze_other_logs(log)

        return {

            "pipeline": "RAG",

            **result

        }


    # ======================================
    # Invalid Log Type
    # ======================================

    else:

        return {

            "success": False,

            "message": (
                "Invalid log type. "
                "Please select 'hdfs' or 'other'."
            )

        }


# ==========================================
# Run Application
# ==========================================

if __name__ == "__main__":

    import uvicorn

    uvicorn.run(

        "main:app",

        host="0.0.0.0",

        port=8000,

        reload=True

    )