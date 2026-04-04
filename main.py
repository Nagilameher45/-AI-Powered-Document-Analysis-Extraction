import os
import json
import httpx
import uvicorn
import sys
from fastapi import FastAPI, Header, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from typing import List, Optional

# --- Fix for ModuleNotFoundError ---
# This ensures that even if the app is run from inside 'src', 
# or from the root, the imports work correctly.
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

# --- Configuration ---
app = FastAPI(title="AI Document Analysis API")

# Enable CORS for the React Frontend
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

# Environment Variables
GEMINI_API_KEY = os.getenv("GEMINI_API_KEY", "AIzaSyCWVOcYgdVATQ85oXn0zc_oX0BGkDfD4Ps")
X_API_KEY = os.getenv("X_API_KEY", "sk_track2_987654321")

# --- Models ---
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

# --- Extraction Strategy (LLM Integration) ---
async def extract_data_with_ai(base64_data: str, file_type: str, file_name: str):
    """
    Uses Gemini-2.5-Flash to handle OCR and Analysis in one pass.
    """
    url = f"https://generativelanguage.googleapis.com/v1beta/models/gemini-2.5-flash-preview-09-2025:generateContent?key={GEMINI_API_KEY}"
    
    system_prompt = """
    You are an AI Document Processor. Analyze the input file.
    1. If it's an image, perform OCR first.
    2. Provide a concise summary (1-2 sentences).
    3. Extract named entities (names, dates, organizations, monetary amounts).
    4. Determine sentiment (Positive, Negative, or Neutral).
    Return ONLY valid JSON.
    """
    
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
                {"text": f"Process this {file_type} file named {file_name}."},
                {"inlineData": {"mimeType": mime_type, "data": base64_data}}
            ]
        }],
        "systemInstruction": {"parts": [{"text": system_prompt}]},
        "generationConfig": {
            "responseMimeType": "application/json",
            "responseSchema": {
                "type": "OBJECT",
                "properties": {
                    "summary": {"type": "string"},
                    "entities": {
                        "type": "OBJECT",
                        "properties": {
                            "names": {"type": "ARRAY", "items": {"type": "string"}},
                            "dates": {"type": "ARRAY", "items": {"type": "string"}},
                            "organizations": {"type": "ARRAY", "items": {"type": "string"}},
                            "amounts": {"type": "ARRAY", "items": {"type": "string"}}
                        }
                    },
                    "sentiment": {"type": "string", "enum": ["Positive", "Neutral", "Negative"]}
                }
            }
        }
    }

    async with httpx.AsyncClient() as client:
        try:
            response = await client.post(url, json=payload, timeout=60.0)
            if response.status_code != 200:
                raise Exception(f"AI Service Error ({response.status_code})")
            
            result = response.json()
            raw_json = result['candidates'][0]['content']['parts'][0]['text']
            return json.loads(raw_json)
        except Exception as e:
            raise Exception(f"AI Connection Error: {str(e)}")

# --- Routes ---
@app.get("/")
async def root():
    return {"message": "API is online", "docs": "/docs"}

@app.post("/api/document-analyze", response_model=AnalysisResponse)
async def analyze_document(
    request: DocumentRequest, 
    x_api_key: Optional[str] = Header(None, alias="x-api-key")
):
    if x_api_key != X_API_KEY:
        raise HTTPException(status_code=401, detail="Unauthorized")

    try:
        analysis = await extract_data_with_ai(
            request.fileBase64, 
            request.fileType, 
            request.fileName
        )
        return {
            "status": "success",
            "fileName": request.fileName,
            **analysis
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
    # Using 'main:app' instead of just 'app' helps uvicorn locate the module correctly
    uvicorn.run("main:app", host="0.0.0.0", port=port, reload=False)
