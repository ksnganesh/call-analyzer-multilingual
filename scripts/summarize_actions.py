import os
import json
import argparse
from transformers import pipeline
import openai
from dotenv import load_dotenv
load_dotenv()
OUTPUTS_DIR = "outputs"


summarizer = pipeline(
    "summarization",
    model="facebook/bart-large-cnn"
)

def llm_action_suggestions(transcript_text, summary, use_openai=False):
    if use_openai:
        
        client = openai.OpenAI(api_key=os.getenv("OPENAI_API_KEY"))
        prompt = f"""
        You are an assistant for call center analysis.
        The following is a call summary:

        {summary}

        The call transcript text was:
        {transcript_text[:2000]}

        Suggest 1-3 concrete follow-up actions for the telecaller
        (short, professional, bullet-point style).
        """
        resp = client.chat.completions.create(
            model="gpt-4o-mini",
            messages=[{"role": "user", "content": prompt}],
            temperature=0.3,
        )
        actions = resp.choices[0].message.content.strip().split("\n")
        actions = [a.strip("-• ") for a in actions if a.strip()]
        return actions[:3]

    else:
        
        if "demo" in summary.lower():
            return ["Schedule a demo"]
        if "price" in summary.lower():
            return ["Send pricing details"]
        return ["Follow up with customer"]

def summarize_and_suggest(call_id, use_openai=False):
    nlp_path = os.path.join(OUTPUTS_DIR, f"{call_id}.nlp.json")
    if not os.path.exists(nlp_path):
        raise FileNotFoundError(f"No NLP file for {call_id}")

    with open(nlp_path, "r") as f:
        nlp_data = json.load(f)

    segments = nlp_data["segments"]
    full_text = " ".join(seg["text"] for seg in segments if seg["text"].strip())

    # Summarize
    if full_text.strip():
        summary = summarizer(
            full_text,
            max_length=120,
            min_length=30,
            do_sample=False
        )[0]["summary_text"]
    else:
        summary = "No content available."

    
    actions = llm_action_suggestions(full_text, summary, use_openai=use_openai)

    out = {
        "call_id": call_id,
        "summary": summary.strip(),
        "suggested_actions": actions
    }

    out_path = os.path.join(OUTPUTS_DIR, f"{call_id}.summary.json")
    with open(out_path, "w") as f:
        json.dump(out, f, indent=2, ensure_ascii=False)

    print(f"[✓] Summary & actions saved to {out_path}")
    return out_path


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Summarize call and suggest follow-ups")
    parser.add_argument("call_id", help="Call ID")
    parser.add_argument("--use_openai", action="store_true", help="Use OpenAI API for actions")
    args = parser.parse_args()

    summarize_and_suggest(args.call_id, use_openai=args.use_openai)


