import os
import json
import argparse
import statistics

OUTPUTS_DIR = "outputs"

def sentiment_to_num(label, score):
    if label == "POSITIVE":
        return score
    elif label == "NEGATIVE":
        return -score
    else:
        return 0

def compute_scores(call_id):
    nlp_path = os.path.join(OUTPUTS_DIR, f"{call_id}.nlp.json")
    if not os.path.exists(nlp_path):
        raise FileNotFoundError(f"No NLP file for {call_id}")

    with open(nlp_path, "r") as f:
        nlp_data = json.load(f)

    segments = nlp_data["segments"]

    
    telecaller_segments = [s for s in segments if s["speaker"] == "SPEAKER_0"]
    customer_segments   = [s for s in segments if s["speaker"] == "SPEAKER_1"]

    
    tele_score = 0

    
    structure_score = 0
    full_text = " ".join(s["text"].lower() for s in telecaller_segments)
    if any(word in full_text for word in ["hello", "hi", "good morning", "good afternoon"]):
        structure_score += 5
    if "campaign" in full_text or "from" in full_text:
        structure_score += 5
    if "agenda" in full_text or "purpose" in full_text:
        structure_score += 5
    if any(word in full_text for word in ["thank you", "have a nice day", "goodbye"]):
        structure_score += 10
    tele_score += min(structure_score, 25)

    
    polite_phrases = ["please", "thank you", "sorry", "apologize"]
    politeness_hits = sum(full_text.count(p) for p in polite_phrases)
    politeness_score = min(20, politeness_hits * 5)
    tele_score += politeness_score

    
    objection_score = 0
    for cust in customer_segments:
        if cust["sentiment_label"] == "NEGATIVE":
            
            resp = next((t for t in telecaller_segments if t["start"] > cust["end"]), None)
            if resp and resp["sentiment_label"] == "POSITIVE":
                objection_score += 10
    tele_score += min(objection_score, 20)

    
    tc_time = sum(s["end"] - s["start"] for s in telecaller_segments)
    cust_time = sum(s["end"] - s["start"] for s in customer_segments)
    total_time = tc_time + cust_time
    if total_time > 0:
        ratio = tc_time / total_time
        if 0.4 <= ratio <= 0.6:
            tele_score += 20
        elif 0.3 <= ratio <= 0.7:
            tele_score += 15
        else:
            tele_score += 5

    
    compliance_score = 15
    for seg in telecaller_segments:
        if seg.get("contains_profanity"):
            compliance_score -= 5
        if seg.get("contains_phone_number"):
            compliance_score -= 5
    compliance_score = max(0, compliance_score)
    tele_score += compliance_score

    tele_score = min(100, tele_score)

    
    cust_sent_values = [sentiment_to_num(s["sentiment_label"], s["sentiment_score"]) for s in customer_segments]
    if not cust_sent_values:
        cust_sent_score = 50
    else:
        overall_avg = statistics.mean(cust_sent_values)

        
        k = max(1, int(0.2 * len(cust_sent_values)))
        final_avg = statistics.mean(cust_sent_values[-k:])

        
        trend = cust_sent_values[-1] - cust_sent_values[0]

        cust_sent_score = (0.5 * overall_avg + 0.3 * final_avg + 0.2 * trend) * 100
        cust_sent_score = max(0, min(100, cust_sent_score))

    out = {
        "call_id": call_id,
        "telecaller_score": round(tele_score),
        "customer_sentiment_score": round(cust_sent_score),
        "components": {
            "structure": structure_score,
            "politeness": politeness_score,
            "objection_handling": objection_score,
            "talk_dynamics": ratio if total_time > 0 else None,
            "compliance": compliance_score
        }
    }

    out_path = os.path.join(OUTPUTS_DIR, f"{call_id}.scores.json")
    with open(out_path, "w") as f:
        json.dump(out, f, indent=2)

    print(f"[✓] Scores saved to {out_path}")
    return out_path


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Compute call scores")
    parser.add_argument("call_id", help="Call ID")
    args = parser.parse_args()

    compute_scores(args.call_id)
