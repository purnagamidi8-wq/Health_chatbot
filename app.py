from flask import Flask, render_template, request, jsonify
from dotenv import load_dotenv
from google import genai
from google.genai import types
import os

# --------------------------------------------------
# Load Environment Variables
# --------------------------------------------------

load_dotenv()

API_KEY = os.getenv("GOOGLE_API_KEY")

if not API_KEY:
    raise ValueError(
        "GOOGLE_API_KEY not found. Create a .env file and add:\n"
        "GOOGLE_API_KEY=your_api_key_here"
    )

# --------------------------------------------------
# Flask App
# --------------------------------------------------

app = Flask(__name__)
app.config["MAX_CONTENT_LENGTH"] = 10 * 1024 * 1024

# --------------------------------------------------
# Gemini Client
# --------------------------------------------------

client = genai.Client(api_key=API_KEY)

MODEL_NAME = "gemini-2.5-flash"

# --------------------------------------------------
# System Prompt
# --------------------------------------------------

SYSTEM_PROMPT = """
You are HealthLens, an AI Public Health Disease Awareness Assistant.

Responsibilities:
- Explain diseases and symptoms.
- Describe causes and risk factors.
- Promote disease prevention.
- Encourage healthy lifestyle habits.
- Analyze health-related images for awareness purposes.

Rules:
- Never diagnose diseases.
- Never prescribe medicines.
- Never replace medical professionals.
- Encourage consultation with healthcare professionals.
- For emergencies advise immediate medical attention.
"""

# --------------------------------------------------
# Routes
# --------------------------------------------------

@app.route("/")
def home():
    return render_template("start.html")


@app.route("/chat")
def chatbot():
    return render_template("index.html")


@app.route("/chat", methods=["POST"])
def chat():

    try:

        user_message = request.form.get("message", "").strip()
        image_file = request.files.get("image")

        if not user_message and not image_file:
            return jsonify({
                "response": "Please enter a question or upload an image."
            })

        parts = []

        # Image Processing
        if image_file and image_file.filename:

            image_bytes = image_file.read()

            parts.append(
                types.Part.from_bytes(
                    data=image_bytes,
                    mime_type=image_file.content_type
                )
            )

        # User Message
        if user_message:
            parts.append(
                types.Part.from_text(
                    text=user_message
                )
            )
        else:
            parts.append(
                types.Part.from_text(
                    text="Analyze this image from a public health awareness perspective."
                )
            )

        response = client.models.generate_content(
            model=MODEL_NAME,
            contents=[
                types.Content(
                    role="user",
                    parts=parts
                )
            ],
            config=types.GenerateContentConfig(
                system_instruction=SYSTEM_PROMPT,
                temperature=0.4,
                max_output_tokens=1024
            )
        )

        return jsonify({
            "response": response.text
        })

    except Exception as e:

        print("ERROR:", str(e))

        return jsonify({
            "response": f"Error: {str(e)}"
        })


# --------------------------------------------------
# Run App
# --------------------------------------------------

if __name__ == "__main__":

    print("=" * 60)
    print("HealthLens AI Chatbot Started")
    print("=" * 60)

    print("API Key Loaded:", API_KEY[:8] + "********")

    print("Running on:")
    print("http://127.0.0.1:5000")

    print("=" * 60)

    app.run(
        host="0.0.0.0",
        port=5000,
        debug=True
    )

