import os
import json
import tempfile
from flask import Flask, request, jsonify, render_template
from openai import OpenAI
import anthropic

app = Flask(__name__)

openai_client = OpenAI(api_key=os.environ.get("OPENAI_API_KEY"))
anthropic_client = anthropic.Anthropic(api_key=os.environ.get("ANTHROPIC_API_KEY"))

EXTRACTION_PROMPT = """You are an expert meeting analyst. Given a transcript, extract structured information.

Return ONLY valid JSON with this exact structure:
{
  "summary": "2-3 sentence summary of the entire conversation",
  "action_items": [
    {"task": "what needs to be done", "owner": "who is responsible (or unassigned)", "deadline": "deadline if mentioned or null"}
  ],
  "decisions": ["list of decisions made during the conversation"],
  "key_facts": ["important facts, numbers, names, or data points mentioned"],
  "follow_ups": ["questions raised but not answered, topics to revisit"],
  "sentiment": "positive | neutral | negative | mixed",
  "topics": ["main topics discussed"],
  "participants": ["names of people mentioned or speaking"]
}

Transcript:
{transcript}
"""

@app.route("/")
def index():
    return render_template("index.html")

@app.route("/transcribe", methods=["POST"])
def transcribe():
    if "audio" not in request.files:
        return jsonify({"error": "No audio file provided"}), 400
    audio_file = request.files["audio"]
    with tempfile.NamedTemporaryFile(suffix=".mp3", delete=False) as tmp:
        audio_file.save(tmp.name)
        tmp_path = tmp.name
    try:
        with open(tmp_path, "rb") as f:
            transcript_response = openai_client.audio.transcriptions.create(
                model="whisper-1",
                file=f,
                response_format="text"
            )
        transcript = transcript_response
        os.unlink(tmp_path)
        return jsonify({"transcript": transcript})
    except Exception as e:
        os.unlink(tmp_path)
        return jsonify({"error": str(e)}), 500

@app.route("/extract", methods=["POST"])
def extract():
    data = request.json
    transcript = data.get("transcript", "")
    if not transcript:
        return jsonify({"error": "No transcript provided"}), 400
    try:
        message = anthropic_client.messages.create(
            model="claude-opus-4-5",
            max_tokens=2000,
            messages=[{"role": "user", "content": EXTRACTION_PROMPT.format(transcript=transcript)}]
        )
        raw = message.content[0].text.strip()
        if raw.startswith("```"):
            raw = raw.split("```")[1]
            if raw.startswith("json"):
                raw = raw[4:]
        structured = json.loads(raw)
        return jsonify({"structured": structured})
    except json.JSONDecodeError:
        return jsonify({"error": "Failed to parse structured output"}), 500
    except Exception as e:
        return jsonify({"error": str(e)}), 500

@app.route("/process", methods=["POST"])
def process():
    if "audio" not in request.files:
        return jsonify({"error": "No audio file provided"}), 400
    audio_file = request.files["audio"]
    with tempfile.NamedTemporaryFile(suffix=".mp3", delete=False) as tmp:
        audio_file.save(tmp.name)
        tmp_path = tmp.name
    try:
        with open(tmp_path, "rb") as f:
            transcript_response = openai_client.audio.transcriptions.create(
                model="whisper-1",
                file=f,
                response_format="text"
            )
        transcript = transcript_response
        os.unlink(tmp_path)
        message = anthropic_client.messages.create(
            model="claude-opus-4-5",
            max_tokens=2000,
            messages=[{"role": "user", "content": EXTRACTION_PROMPT.format(transcript=transcript)}]
        )
        raw = message.content[0].text.strip()
        if raw.startswith("```"):
            raw = raw.split("```")[1]
            if raw.startswith("json"):
                raw = raw[4:]
        structured = json.loads(raw)
        return jsonify({"transcript": transcript, "structured": structured})
    except Exception as e:
        if os.path.exists(tmp_path):
            os.unlink(tmp_path)
        return jsonify({"error": str(e)}), 500

if __name__ == "__main__":    app.run(debug=True, port=5000)
