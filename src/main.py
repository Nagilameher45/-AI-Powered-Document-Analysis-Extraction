import os
import json
import base64
import httpx
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

# ---------------- APP ---------------- #
app = FastAPI(title="AI Document Analyzer")

# ---------------- CORS ---------------- #
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# ---------------- CONFIG ---------------- #
GEMINI_API_KEY = os.getenv("GEMINI_API_KEY")
X_API_KEY = "sk_track2_987654321"

# ---------------- MODELS ---------------- #
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

# ---------------- TEXT EXTRACTION ---------------- #
def extract_text(base64_data, file_type):
    file_bytes = base64.b64decode(base64_data)

    try:
        # -------- PDF -------- #
        if file_type == "pdf":
            text = ""
            pdf = fitz.open(stream=file_bytes, filetype="pdf")
            for page in pdf:
                text += page.get_text()
            return text

        # -------- IMAGE (OCR) -------- #
        elif file_type in ["jpg", "jpeg", "png"]:
            image = Image.open(io.BytesIO(file_bytes)).convert("L")
            text = pytesseract.image_to_string(image, config='--psm 6')
            return text

        # -------- DOCX -------- #
        elif file_type == "docx":
            doc = Document(io.BytesIO(file_bytes))
            return "\n".join([p.text for p in doc.paragraphs])

    except Exception as e:
        print("Extraction Error:", e)

    return ""

# ---------------- AI ANALYSIS ---------------- #
async def analyze_text(text):
    text = text[:4000]

    url = f"https://generativelanguage.googleapis.com/v1beta/models/gemini-pro:generateContent?key={GEMINI_API_KEY}"

    prompt = f"""
    You are an AI document analysis system.

    Analyze the document and STRICTLY return JSON:

    RULES:
    - Summary: 3-5 lines only
    - Extract all names, dates, organizations, monetary amounts
    - Sentiment: Positive / Negative / Neutral

    TEXT:
    {text}

    OUTPUT:
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
        "contents": [{"parts": [{"text": prompt}]}]
    }

    async with httpx.AsyncClient(timeout=30) as client:
        res = await client.post(url, json=payload)

        if res.status_code != 200:
            raise Exception(f"Gemini Error: {res.text}")

        data = res.json()

        try:
            raw = data['candidates'][0]['content']['parts'][0]['text']
            return json.loads(raw)
        except:
            return {
                "summary": "AI parsing failed",
                "entities": {
                    "names": [],
                    "dates": [],
                    "organizations": [],
                    "amounts": []
                },
                "sentiment": "Neutral"
            }

# ---------------- ROUTES ---------------- #

@app.get("/", response_class=HTMLResponse)
def home():
    with open("index.html", "r", encoding="utf-8") as f:
        return f.read()

@app.post("/api/document-analyze", response_model=AnalysisResponse)
async def analyze(req: DocumentRequest, x_api_key: Optional[str] = Header(None)):

    # ---- API KEY CHECK ---- #
    if x_api_key != X_API_KEY:
        raise HTTPException(status_code=401, detail="Unauthorized")

    # ---- TEXT EXTRACTION ---- #
    text = extract_text(req.fileBase64, req.fileType.lower())

    if not text.strip():
        raise HTTPException(status_code=400, detail="Text extraction failed")

    # ---- AI PROCESS ---- #
    result = await analyze_text(text)

    return {
        "status": "success",
        "fileName": req.fileName,
        "summary": result.get("summary", ""),
        "entities": result.get("entities", {
            "names": [],
            "dates": [],
            "organizations": [],
            "amounts": []
        }),
        "sentiment": result.get("sentiment", "Neutral")
    }

# ---------------- RUN ---------------- #
if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
