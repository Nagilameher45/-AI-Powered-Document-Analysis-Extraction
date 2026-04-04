import os
import json
import base64
import httpx
import uvicorn
import fitz  # PyMuPDF
import io
from PIL import Image
import pytesseract
from docx import Document

from fastapi import FastAPI, Header, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import HTMLResponse
from pydantic import BaseModel
from typing import List, Optional

app = FastAPI(title="FREE AI Document Analyzer")

# --- CORS ---
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# --- API Keys ---
GEMINI_API_KEY = os.getenv("GEMINI_API_KEY")
X_API_KEY = "sk_track2_987654321"

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

# --- TEXT EXTRACTION ---
def extract_text(base64_data, file_type):
    file_bytes = base64.b64decode(base64_data)

    if file_type == "pdf":
        text = ""
        pdf = fitz.open(stream=file_bytes, filetype="pdf")
        for page in pdf:
            text += page.get_text()
        return text

    elif file_type in ["jpg", "jpeg", "png"]:
        image = Image.open(io.BytesIO(file_bytes))
        return pytesseract.image_to_string(image)

    elif file_type == "docx":
        doc = Document(io.BytesIO(file_bytes))
        return "\n".join([p.text for p in doc.paragraphs])

    return ""

# --- AI FUNCTION ---
async def analyze_text(text):
    text = text[:3000]

    url = f"https://generativelanguage.googleapis.com/v1beta/models/gemini-pro:generateContent?key={GEMINI_API_KEY}"

    prompt = f"""
    Analyze the document and return ONLY JSON:

    Text:
    {text}

    Format:
    {{
      "summary": "",
      "entities": {{
        "names": [],
        "dates": [],
        "organizations": [],
        "amounts": []
      }},
      "sentiment": ""
    }}
    """

    payload = {
        "contents": [
            {"parts": [{"text": prompt}]}
        ]
    }

    async with httpx.AsyncClient() as client:
        res = await client.post(url, json=payload)

        if res.status_code != 200:
            raise Exception(res.text)

        data = res.json()
        raw = data['candidates'][0]['content']['parts'][0]['text']

        try:
            return json.loads(raw)
        except:
            return {
                "summary": raw[:200],
                "entities": {"names": [], "dates": [], "organizations": [], "amounts": []},
                "sentiment": "Neutral"
            }

# --- ROUTES ---
@app.get("/", response_class=HTMLResponse)
def home():
    with open("index.html", "r", encoding="utf-8") as f:
        return f.read()

@app.post("/api/document-analyze", response_model=AnalysisResponse)
async def analyze(req: DocumentRequest, x_api_key: Optional[str] = Header(None)):
    
    if x_api_key != X_API_KEY:
        raise HTTPException(status_code=401, detail="Unauthorized")

    text = extract_text(req.fileBase64, req.fileType.lower())

    if not text:
        raise HTTPException(status_code=400, detail="Text extraction failed")

    result = await analyze_text(text)

    return {
        "status": "success",
        "fileName": req.fileName,
        "summary": result.get("summary", ""),
        "entities": result.get("entities", {
            "names": [], "dates": [], "organizations": [], "amounts": []
        }),
        "sentiment": result.get("sentiment", "Neutral")
    }

# --- RUN ---
if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000)
