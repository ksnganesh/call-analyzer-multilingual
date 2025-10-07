import os
import shutil
import json
import uuid
from datetime import datetime
import argparse

PROCESSED_DIR = "data/processed"

def ingest(file_path, telecaller_id=None, campaign=None):
    
    os.makedirs(PROCESSED_DIR, exist_ok=True)

    
    call_id = str(uuid.uuid4())

    ext = os.path.splitext(file_path)[1].lower()
    dest_path = os.path.join(PROCESSED_DIR, f"{call_id}{ext}")

    shutil.copy(file_path, dest_path)

    meta = {
        "call_id": call_id,
        "original_file": file_path,
        "processed_file": dest_path,
        "telecaller_id": telecaller_id,
        "campaign": campaign,
        "ingest_time": datetime.utcnow().isoformat() + "Z"
    }

    meta_path = os.path.join(PROCESSED_DIR, f"{call_id}.meta.json")
    with open(meta_path, "w") as f:
        json.dump(meta, f, indent=2)

    print(f"[✓] Ingested file: {file_path}")
    print(f"[+] Call ID: {call_id}")
    return call_id

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Ingest an audio file into pipeline")
    parser.add_argument("file_path", help="Path to audio file")
    parser.add_argument("--telecaller_id", default=None, help="Telecaller ID")
    parser.add_argument("--campaign", default=None, help="Campaign name")
    args = parser.parse_args()

    ingest(args.file_path, args.telecaller_id, args.campaign)