import os
import json
import argparse
import numpy as np
import librosa
from resemblyzer import VoiceEncoder, preprocess_wav
from sklearn.cluster import AgglomerativeClustering

PROCESSED_DIR = "data/processed"
OUTPUTS_DIR = "outputs"

def diarize(call_id, num_speakers=2, window_size=1.5, hop_size=0.75):
    
    meta_path = os.path.join(PROCESSED_DIR, f"{call_id}.meta.json")
    if not os.path.exists(meta_path):
        raise FileNotFoundError(f"No metadata for call_id {call_id}")

    with open(meta_path, "r") as f:
        meta = json.load(f)

    audio_path = meta["processed_file"]

    
    wav, sr = librosa.load(audio_path, sr=None)
    wav_preprocessed = preprocess_wav(audio_path)

    
    encoder = VoiceEncoder()
    step = int(hop_size * sr)
    size = int(window_size * sr)

    frames = []
    embs = []
    for start in range(0, len(wav) - size, step):
        chunk = wav[start:start + size]
        if len(chunk) == 0:
            continue
        emb = encoder.embed_utterance(chunk)
        embs.append(emb)
        frames.append((start / sr, (start + size) / sr))

    embs = np.array(embs)

    
    clustering = AgglomerativeClustering(n_clusters=num_speakers).fit(embs)
    labels = clustering.labels_

    
    diarization = []
    for (start, end), label in zip(frames, labels):
        diarization.append({
            "start": float(start),
            "end": float(end),
            "speaker": f"SPEAKER_{label}"
        })

    
    transcript_path = os.path.join(OUTPUTS_DIR, f"{call_id}.transcript.json")
    diarized_segments = []
    if os.path.exists(transcript_path):
        with open(transcript_path, "r") as f:
            transcript = json.load(f)

        for seg in transcript["segments"]:
            seg_mid = (seg["start"] + seg["end"]) / 2
            
            speaker = "UNKNOWN"
            for d in diarization:
                if d["start"] <= seg_mid <= d["end"]:
                    speaker = d["speaker"]
                    break
            diarized_segments.append({
                "id": seg["id"],
                "start": seg["start"],
                "end": seg["end"],
                "text": seg["text"],
                "speaker": speaker
            })
    else:
        diarized_segments = diarization

    
    out = {
        "call_id": call_id,
        "segments": diarized_segments,
        "num_speakers": num_speakers
    }

    os.makedirs(OUTPUTS_DIR, exist_ok=True)
    out_path = os.path.join(OUTPUTS_DIR, f"{call_id}.diarized.json")
    with open(out_path, "w") as f:
        json.dump(out, f, indent=2, ensure_ascii=False)

    print(f"[✓] Diarization saved to {out_path}")
    return out_path


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Diarize call audio for given call_id")
    parser.add_argument("call_id", help="Call ID from ingestion step")
    parser.add_argument("--num_speakers", type=int, default=2, help="Expected number of speakers")
    args = parser.parse_args()

    diarize(args.call_id, num_speakers=args.num_speakers)
