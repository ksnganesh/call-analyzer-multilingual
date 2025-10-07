import os
import json
import argparse
import whisper

PROCESSED_DIR = "data/processed"
OUTPUTS_DIR = "outputs"

def transcribe(call_id, model_size="medium"):
    
    meta_path = os.path.join(PROCESSED_DIR, f"{call_id}.meta.json")
    if not os.path.exists(meta_path):
        raise FileNotFoundError(f"No metadata for call_id {call_id}")

    with open(meta_path, "r") as f:
        meta = json.load(f)

    audio_path = meta["processed_file"]

    
    print(f"[~] Loading Whisper model: {model_size}")
    model = whisper.load_model(model_size)

    
    print(f"[~] Transcribing {audio_path} ...")
    result = model.transcribe(audio_path, verbose=False)

    
    out = {
        "call_id": call_id,
        "language": result.get("language"),
        "full_transcript": result["text"],
        "segments": []
    }

    for seg in result["segments"]:
        out["segments"].append({
            "id": seg["id"],
            "start": seg["start"],
            "end": seg["end"],
            "text": seg["text"].strip()
        })

    
    os.makedirs(OUTPUTS_DIR, exist_ok=True)
    out_path = os.path.join(OUTPUTS_DIR, f"{call_id}.transcript.json")
    with open(out_path, "w") as f:
        json.dump(out, f, indent=2, ensure_ascii=False)

    print(f"[✓] Transcript saved to {out_path}")
    return out_path

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Transcribe audio for given call_id")
    parser.add_argument("call_id", help="Call ID from ingestion step")
    parser.add_argument("--model_size", default="small", help="Whisper model size (tiny/small/medium/large)")
    args = parser.parse_args()

    transcribe(args.call_id, args.model_size)
