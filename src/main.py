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

# --- Initialization & Path Setup ---
# Ensures the app can find the index.html file in the same directory [cite: 59]
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.append(BASE_DIR)

app = FastAPI(
    title="AI-Powered Document Analysis",
    description="Track 2: AI-Powered Document Analysis & Extraction API",
    version="1.0.0"
)

# --- Middleware ---
# Essential for allowing the web-based frontend to communicate with this API
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

# --- Configuration (Requirement 6 & 12) [cite: 25, 72] ---
# These are retrieved from Render's Environment Variables
GEMINI_API_KEY = os.getenv("GEMINI_API_KEY", "AIzaSyCWVOcYgdVATQ85oXn0zc_oX0BGkDfD4Ps")
X_API_KEY = os.getenv("X_API_KEY", "sk_track2_987654321")

# --- Data Models (Requirement 8 & 9) [cite: 39, 41] ---
class DocumentRequest(BaseModel):. BaseModel
    fileName: str # [cite: 40]
    fileType: str # [cite: 40]
    fileBase64: str # [cite: 40]

class Entities(BaseModel):
    names: List[str] # [cite: 48]
    dates: List[str] # [cite: 49]
    organizations: List[str] # [cite: 50]
    amounts: List[str] # [cite: 51]

class AnalysisResponse(BaseModel):
    status: str # [cite: 44]
    fileName: str # [cite: 45]
    summary: str # [cite: 46]
    entities: Entities # [cite: 47]
    sentiment: str # [cite: 53]

# --- AI Core Logic (Requirement 2 & 12) [cite: 5, 75] ---
async def extract_and_analyze(base64_data: str, file_type: str, file_name: str):
    """
    Uses Gemini-2.5-Flash to perform OCR and extraction in one step.
    This fulfills the requirement for multi-format support and automatic analysis[cite: 3, 6].
    """
    url = f"https://generativelanguage.googleapis.com/v1beta/models/gemini-2.5-flash-preview-09-2025:generateContent?key={GEMINI_API_KEY}"
    
    system_prompt = """
    You are a professional Document Analysis System[cite: 3].
    Analyze the provided content and extract:
    1. SUMMARY: A concise 1-sentence summary[cite: 8, 95].
    2. ENTITIES: Names, dates, organizations, and monetary amounts[cite: 9, 98].
    3. SENTIMENT: 'Positive', 'Neutral', or 'Negative'[cite: 100].
    Return the response strictly as a JSON object[cite: 23].
    """
    
    # Mime Type mapping for Multi-format support (Requirement 2) [cite: 6, 20]
    mime_map = {
        "pdf": "application/pdf",
        "docx": "application/vnd.openxmlformats-officedocument.wordprocessingml.document",
        "jpg": "image/jpeg",
        "jpeg": "image/jpeg",
        "png": "image/png"
    }
    mime_type = mime_map.get(file_type.lower(), "application/octet-stream")

    payload = {
        "contents": [{
            "parts": [
                {"text": f"Analyze this {file_type} document named '{file_name}'."},
                {"inlineData": {"mimeType": mime_type, "data": base64_data}}
            ]
        }],
        "systemInstruction": {"parts": [{"text": system_prompt}]},
        "generationConfig": {
            "responseMimeType": "application/json"
        }
    }

    async with httpx.AsyncClient() as client:
        response = await client.post(url, json=payload, timeout=60.0)
        if response.status_code != 200:
            raise Exception(f"AI Service Error: {response.status_code}")
        
        result = response.json()
        raw_output = result['candidates'][0]['content']['parts'][0]['text']
        return json.loads(raw_output)

# --- Routes ---

@app.get("/", response_class=HTMLResponse)
async def serve_dashboard():
    """
    Serves the dashboard UI (index.html) at the root URL[cite: 15].
    """
    try:
        html_path = os.path.join(BASE_DIR, "index.html")
        with open(html_path, "r", encoding="utf-8") as f:
            return f.read()
    except FileNotFoundError:
        return "<h1>API Live</h1><p>Upload index.html to the src/ folder to see the dashboard.</p>"

@app.post("/api/document-analyze", response_model=AnalysisResponse)
async def analyze_document_endpoint(
    request: DocumentRequest, 
    x_api_key: Optional[str] = Header(None, alias="x-api-key")
):
    """
    Main extraction endpoint (Requirement 5)[cite: 18, 30]. 
    Validates x-api-key (Requirement 6) and returns structured data[cite: 25, 41].
    """
    if x_api_key != X_API_KEY:
        raise HTTPException(status_code=401, detail="Unauthorized: Invalid API Key [cite: 25]")

    try:
        analysis_data = await extract_and_analyze(
            request.fileBase64, 
            request.fileType, 
            request.fileName
        )
        
        return {
            "status": "success",
            "fileName": request.fileName,
            **analysis_data
        }
    except Exception as e:
        return {
            "status": "error",
            "fileName": request.fileName,
            "summary": f"Failed to process: {str(e)}",
            "entities": {"names": [], "dates": [], "organizations": [], "amounts": []},
            "sentiment": "Neutral"
        }

if __name__ == "__main__":
    # Render assigns the port via the PORT environment variable [cite: 73]
    server_port = int(os.environ.get("PORT", 8000))
    uvicorn.run(app, host="0.0.0.0", port=server_port)
