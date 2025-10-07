import os
import uuid
import shutil
from fastapi import FastAPI, UploadFile, Request
from fastapi.responses import HTMLResponse
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates
import subprocess
import json

app = FastAPI()
app.mount("/static", StaticFiles(directory="static"), name="static")
templates = Jinja2Templates(directory="templates")

DATA_IN = "data/incoming"

os.makedirs(DATA_IN, exist_ok=True)
os.makedirs("outputs", exist_ok=True)

def run_pipeline(call_id):
    """Run pipeline sequentially for a given call_id"""
    subprocess.run(["python", "scripts/transcribe.py", call_id])
    subprocess.run(["python", "scripts/diarize.py", call_id])
    subprocess.run(["python", "scripts/nlp_worker.py", call_id])
    subprocess.run(["python", "scripts/summarize_actions.py", call_id, "--use_openai"])
    subprocess.run(["python", "scripts/score.py", call_id])

@app.get("/", response_class=HTMLResponse)
def index(request: Request):
    return templates.TemplateResponse("index.html", {"request": request})

@app.post("/analyze", response_class=HTMLResponse)
def analyze(request: Request, file: UploadFile):
    
    ext = os.path.splitext(file.filename)[1]
    temp_path = os.path.join(DATA_IN, file.filename)
    with open(temp_path, "wb") as buffer:
        shutil.copyfileobj(file.file, buffer)

    
    call_id = str(uuid.uuid4())
    dest = f"data/processed/{call_id}{ext}"
    shutil.copy(temp_path, dest)
    with open(f"data/processed/{call_id}.meta.json", "w") as f:
        json.dump({"call_id": call_id, "processed_file": dest}, f)

    
    run_pipeline(call_id)

    
    with open(f"outputs/{call_id}.diarized.json") as f:
        diarized = json.load(f)
    with open(f"outputs/{call_id}.scores.json") as f:
        scores = json.load(f)
    with open(f"outputs/{call_id}.summary.json") as f:
        summary_data = json.load(f)

    return templates.TemplateResponse(
        "results.html",
        {
            "request": request,
            "call_id": call_id,
            "segments": diarized["segments"],
            "tele_score": scores["telecaller_score"],
            "cust_score": scores["customer_sentiment_score"],
            "summary": summary_data.get("summary", "No summary generated."),
            "actions": summary_data.get("suggested_actions", [])
        }
    )
