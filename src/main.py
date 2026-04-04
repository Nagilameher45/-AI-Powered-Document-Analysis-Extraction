import os
import json
import httpx
import uvicorn
import sys
from fastapi import FastAPI, Header, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import HTMLResponse
from pydantic import BaseModel
from typing import List, Optional

# --- Initialization ---
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.append(BASE_DIR)

app = FastAPI(title="AI Document Analysis API")

# --- CORS Setup ---
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# --- Environment Variables (IMPORTANT) ---
GEMINI_API_KEY = os.getenv("GEMINI_API_KEY")
X_API_KEY = os.getenv("X_API_KEY", "sk_track2_987654321")

# --- Data Schemas ---
class DocumentRequest(BaseModel):
    fileName: str
    fileType: str
    fileBase64: str

class Entities(BaseModel):
    names: List[str]
    dates: List[str]
    organizations: List[str]
    amounts: List[str]

class AnalysisResponse(BaseModel):
    status: str
    fileName: str
    summary: str
    entities: Entities
    sentiment: str

# --- AI Logic ---
async def extract_data(base64_data: str, file_type: str, file_name: str):
    
    if not GEMINI_API_KEY:
        raise Exception("Missing GEMINI_API_KEY")

    url = f"https://generativelanguage.googleapis.com/v1/models/gemini-1.5-flash:generateContent?key={GEMINI_API_KEY}"

    system_prompt = """
    Extract:
    1. One-line summary
    2. Entities:
       - names
       - dates
       - organizations
       - amounts
    3. Sentiment (Positive/Neutral/Negative)

    Return ONLY JSON in this format:
    {
      "summary": "",
      "entities": {
        "names": [],
        "dates": [],
        "organizations": [],
        "amounts": []
      },
      "sentiment": ""
    }
    """

    mime_map = {
        "pdf": "application/pdf",
        "docx": "application/vnd.openxmlformats-officedocument.wordprocessingml.document",
        "jpg": "image/jpeg",
        "jpeg": "image/jpeg",
        "png": "image/png"
    }

    mime_type = mime_map.get(file_type.lower(), "application/octet-stream")

    # ✅ FIXED PAYLOAD
    payload = {
        "contents": [
            {
                "role": "user",
                "parts": [
                    {"text": f"{system_prompt}\nFile Name: {file_name}"},
                    {
                        "inline_data": {
                            "mime_type": mime_type,
                            "data": base64_data
                        }
                    }
                ]
            }
        ]
    }

    async with httpx.AsyncClient(timeout=60.0) as client:
        response = await client.post(url, json=payload)

        print("STATUS:", response.status_code)
        print("RESPONSE:", response.text)

        if response.status_code != 200:
            raise Exception(f"AI Service Error: {response.status_code} - {response.text}")

        result = response.json()

        try:
            raw_text = result['candidates'][0]['content']['parts'][0]['text']
            return json.loads(raw_text)
        except Exception:
            raise Exception("Invalid AI response format")

# --- Routes ---
@app.get("/", response_class=HTMLResponse)
async def serve_home():
    try:
        path = os.path.join(BASE_DIR, "index.html")
        with open(path, "r", encoding="utf-8") as f:
            return f.read()
    except Exception:
        return "<h1>API is Online</h1><p>Ready to process documents.</p>"

@app.post("/api/document-analyze", response_model=AnalysisResponse)
async def analyze_document(
    request: DocumentRequest,
    x_api_key: Optional[str] = Header(None, alias="x-api-key")
):
    if x_api_key != X_API_KEY:
        raise HTTPException(status_code=401, detail="Unauthorized")

    try:
        analysis = await extract_data(
            request.fileBase64,
            request.fileType,
            request.fileName
        )

        return {
            "status": "success",
            "fileName": request.fileName,
            "summary": analysis.get("summary", ""),
            "entities": analysis.get("entities", {
                "names": [],
                "dates": [],
                "organizations": [],
                "amounts": []
            }),
            "sentiment": analysis.get("sentiment", "Neutral")
        }

    except Exception as e:
        return {
            "status": "error",
            "fileName": request.fileName,
            "summary": str(e),
            "entities": {
                "names": [],
                "dates": [],
                "organizations": [],
                "amounts": []
            },
            "sentiment": "Neutral"
        }

# --- Run Server ---
if __name__ == "__main__":
    port = int(os.environ.get("PORT", 8000))
    uvicorn.run(app, host="0.0.0.0", port=port)
