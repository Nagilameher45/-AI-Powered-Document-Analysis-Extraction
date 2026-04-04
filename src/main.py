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

# --- CORS Setup (Requirement for Web Browsers) ---
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Configuration from Render Environment Variables
GEMINI_API_KEY = os.getenv("GEMINI_API_KEY", "AIzaSyCWVOcYgdVATQ85oXn0zc_oX0BGkDfD4Ps")
X_API_KEY = os.getenv("X_API_KEY", "sk_track2_987654321")

# --- Data Schemas (Requirement 39 & 41) ---
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

# --- AI Logic (Requirement 4, 8, 9, & 12) ---
async def extract_data(base64_data: str, file_type: str, file_name: str):
    # API Endpoint for Gemini 1.5 Flash
    url = f"https://generativelanguage.googleapis.com/v1beta/models/gemini-1.5-flash:generateContent?key={GEMINI_API_KEY}"
    
    system_prompt = """
    Analyze the document. Extract:
    1. SUMMARY: A concise 1-sentence summary. [cite: 95]
    2. ENTITIES: Names, dates, organizations, and monetary amounts. [cite: 9, 98]
    3. SENTIMENT: Positive, Neutral, or Negative. [cite: 100]
    Return ONLY a JSON object. [cite: 23]
    """
    
    mime_map = {
        "pdf": "application/pdf",
        "docx": "application/vnd.openxmlformats-officedocument.wordprocessingml.document",
        "jpg": "image/jpeg", "jpeg": "image/jpeg", "png": "image/png"
    }
    mime_type = mime_map.get(file_type.lower(), "application/octet-stream")

    payload = {
        "contents": [{
            "parts": [
                {"text": f"File Name: {file_name}"},
                {"inlineData": {"mimeType": mime_type, "data": base64_data}}
            ]
        }],
        "systemInstruction": {"parts": [{"text": system_prompt}]},
        "generationConfig": { "responseMimeType": "application/json" }
    }

    async with httpx.AsyncClient() as client:
        response = await client.post(url, json=payload, timeout=60.0)
        if response.status_code != 200:
            raise Exception(f"AI Service Error: {response.status_code}")
        
        result = response.json()
        raw_text = result['candidates'][0]['content']['parts'][0]['text']
        return json.loads(raw_text)

# --- Routes ---

@app.get("/", response_class=HTMLResponse)
async def serve_home():
    """Serves the dashboard UI (Requirement 15)."""
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
    # Requirement 25: API Authentication
    if x_api_key != X_API_KEY:
        raise HTTPException(status_code=401, detail="Unauthorized")

    try:
        analysis = await extract_data(request.fileBase64, request.fileType, request.fileName)
        return {
            "status": "success",
            "fileName": request.fileName,
            "summary": analysis.get("summary", ""),
            "entities": analysis.get("entities", {"names": [], "dates": [], "organizations": [], "amounts": []}),
            "sentiment": analysis.get("sentiment", "Neutral")
        }
    except Exception as e:
        return {
            "status": "error",
            "fileName": request.fileName,
            "summary": str(e),
            "entities": {"names": [], "dates": [], "organizations": [], "amounts": []},
            "sentiment": "Neutral"
        }

if __name__ == "__main__":
    port = int(os.environ.get("PORT", 8000))
    uvicorn.run(app, host="0.0.0.0", port=port)
