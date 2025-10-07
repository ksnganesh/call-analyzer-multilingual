import os
import json
import argparse
from langdetect import detect, DetectorFactory
from better_profanity import profanity
import re
from transformers import pipeline

OUTPUTS_DIR = "outputs"


DetectorFactory.seed = 0


sentiment_model = pipeline(
    "sentiment-analysis",
    model="tabularisai/multilingual-sentiment-analysis"
)

def enrich_segments(call_id):
    diarized_path = os.path.join(OUTPUTS_DIR, f"{call_id}.diarized.json")
    if not os.path.exists(diarized_path):
        raise FileNotFoundError(f"No diarized file for {call_id}")

    with open(diarized_path, "r") as f:
        diarized = json.load(f)

    segments = diarized["segments"]

    profanity.load_censor_words()

    for seg in segments:
        text = seg["text"]

        
        try:
            seg["language"] = detect(text) if text.strip() else "unk"
        except:
            seg["language"] = "unk"

        
        if text.strip():
            res = sentiment_model(text[:512])[0]  
            seg["sentiment_label"] = res["label"]
            seg["sentiment_score"] = float(res["score"])
        else:
            seg["sentiment_label"] = "NEUTRAL"
            seg["sentiment_score"] = 0.0

        
        seg["contains_profanity"] = profanity.contains_profanity(text)

        
        phone_regex = re.compile(r"\b\d{10}\b")
        seg["contains_phone_number"] = bool(phone_regex.search(text))

    out = {
        "call_id": call_id,
        "segments": segments
    }

    out_path = os.path.join(OUTPUTS_DIR, f"{call_id}.nlp.json")
    with open(out_path, "w") as f:
        json.dump(out, f, indent=2, ensure_ascii=False)

    print(f"[✓] NLP analysis saved to {out_path}")
    return out_path


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run NLP analysis on diarized transcript")
    parser.add_argument("call_id", help="Call ID")
    args = parser.parse_args()

    enrich_segments(args.call_id)
