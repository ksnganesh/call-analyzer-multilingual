import os
import json
import argparse
from jiwer import wer

OUTPUTS_DIR = "outputs"

def evaluate_wer(call_id, ref_path):
    
    sys_path = os.path.join(OUTPUTS_DIR, f"{call_id}.transcript.json")
    if not os.path.exists(sys_path):
        raise FileNotFoundError(f"No transcript found for {call_id}")
    with open(sys_path, "r") as f:
        sys = json.load(f)
    sys_text = sys["full_transcript"].lower()

    with open(ref_path, "r") as f:
        ref_text = f.read().strip().lower()

    error = wer(ref_text, sys_text)
    print(f"[WER] {call_id}: {error:.2%}")
    return error

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Evaluate WER against reference")
    parser.add_argument("call_id", help="Call ID")
    parser.add_argument("ref_file", help="Reference transcript file (plain text)")
    args = parser.parse_args()
    evaluate_wer(args.call_id, args.ref_file)
