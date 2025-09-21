import os
import io
from fastapi import FastAPI, UploadFile, File, HTTPException
from fastapi.responses import StreamingResponse
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
import google.generativeai as genai
from spitch import Spitch

# Initialize FastAPI app
app = FastAPI(title="Naija AI Assistant API", version="1.0.0")

# --- CORS Middleware Configuration ---
# This is the new section that fixes the error.
# It allows your frontend (running on localhost) to make requests to your backend.
origins = [
    "http://localhost:8080",  # For local development with Vite
    "http://localhost:3000",  # For local development with Next.js/Create React App
    # Add the URL of your deployed frontend here once it's live
]

app.add_middleware(
    CORSMiddleware,
    allow_origins=origins,
    allow_credentials=True,
    allow_methods=["*"],  # Allows all HTTP methods
    allow_headers=["*"],  # Allows all headers
)

# --- Configuration ---
# Load API keys from environment variables
spitch_api_key = os.getenv("SPITCH_API_KEY")
gemini_api_key = os.getenv("GEMINI_API_KEY")

if not spitch_api_key:
    raise RuntimeError("SPITCH_API_KEY environment variable not set.")
if not gemini_api_key:
    raise RuntimeError("GEMINI_API_KEY environment variable not set.")

# Initialize Spitch and Gemini clients
try:
    spitch_client = Spitch(api_key=spitch_api_key)
    genai.configure(api_key=gemini_api_key)
except Exception as e:
    raise RuntimeError(f"Failed to initialize API clients: {e}")

# --- Pydantic Models ---
class TextToSpeechRequest(BaseModel):
    text: str
    language: str = "en-NG" # Changed from pcm-NG
    voice: str = "femi"

class ChatRequest(BaseModel):
    text: str
    language: str = "yo-NG"

# --- API Endpoints ---
@app.post("/speech-to-text/", summary="Transcribe Audio to Text")
async def speech_to_text(file: UploadFile = File(...), language: str = "yo-NG"):
    """
    Transcribes audio to text.
    - **file**: The audio file to transcribe.
    - **language**: The language code of the audio (e.g., 'yo-NG', 'ig-NG', 'ha-NG', 'en-NG').
    """
    if not file.content_type.startswith("audio/"):
        raise HTTPException(status_code=400, detail="Invalid file type. Please upload an audio file.")

    try:
        content = await file.read()
        response = spitch_client.speech.transcribe(content=content, language=language)
        return {"text": response.text}
    except Exception as e:
        print(f"Spitch STT Error: {e}")
        raise HTTPException(status_code=500, detail=f"Spitch API error: {e}")

@app.post("/text-to-speech/", summary="Convert Text to Speech")
async def text_to_speech(request: TextToSpeechRequest):
    """
    Converts text to spoken audio.
    """
    try:
        response = spitch_client.speech.generate(
            text=request.text,
            language=request.language,
            voice=request.voice,
        )
        return StreamingResponse(io.BytesIO(response.read()), media_type="audio/wav")
    except Exception as e:
        print(f"Spitch TTS Error: {e}")
        raise HTTPException(status_code=500, detail=f"Spitch API error: {e}")

@app.post("/chat/", summary="Chat with the AI Assistant")
async def chat(request: ChatRequest):
    """
    Receives a text prompt, gets a response from Gemini in a specified Nigerian
    language, and converts that response to speech using Spitch.
    """
    try:
        model = genai.GenerativeModel('gemini-1.5-flash')

        prompt = (
            "You are a helpful and friendly Naija AI assistant. "
            f"Please respond in the {request.language} language to the following message: '{request.text}'"
        )

        gemini_response = await model.generate_content_async(prompt)
        ai_text = gemini_response.text

        voice_map = {
            "ha-NG": "hasan",
            "ig-NG": "ngozi",
        }
        selected_voice = voice_map.get(request.language, "femi") # 'femi' will be used for Yoruba and English

        tts_response = spitch_client.speech.generate(
            text=ai_text,
            language=request.language,
            voice=selected_voice,
        )

        audio_content = tts_response.read()
        return StreamingResponse(io.BytesIO(audio_content), media_type="audio/wav")
    except Exception as e:
        print(f"Chat Endpoint Error: {e}")
        raise HTTPException(status_code=500, detail=f"An error occurred: {e}")

@app.get("/", summary="Health Check", include_in_schema=False)
def read_root():
    """A simple endpoint to confirm that the API is running."""
    return {"status": "ok", "message": "Naija AI Assistant is running!"}

