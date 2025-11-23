from fastapi import FastAPI
from pydantic import BaseModel
from src.inference import summarize

app = FastAPI(title="Legal Policy Summarizer API")

class Request(BaseModel):
    text: str
    mode: str = "paragraph"

@app.post("/summarize")
async def summarize_api(req: Request):
    result = summarize(req.text, req.mode)
    return {"summary": result, "mode": req.mode}
