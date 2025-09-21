import os
import io
from fastapi import FastAPI, UploadFile, File, HTTPException
from fastapi.responses import StreamingResponse
from pydantic import BaseModel
from google import genai
from spitch import Spitch

# Initialize FastAPI app
app = FastAPI(title="Naija AI Assistant API", version="1.0.0")

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
    language: str = "pcm"
    voice: str = "femi"

class ChatRequest(BaseModel):
    text: str
    language: str = "pcm"

# --- API Endpoints ---
@app.post("/speech-to-text/", summary="Transcribe Audio to Text")
async def speech_to_text(file: UploadFile = File(...), language: str = "pcm"):
    if not file.content_type.startswith("audio/"):
        raise HTTPException(status_code=400, detail="Invalid file type. Please upload an audio file.")

    try:
        content = await file.read()
        response = spitch_client.speech.transcribe(content=content, language=language)
        return {"text": response.text}
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Spitch API error: {e}")

@app.post("/text-to-speech/", summary="Convert Text to Speech")
async def text_to_speech(request: TextToSpeechRequest):
    try:
        response = spitch_client.speech.generate(
            text=request.text,
            language=request.language,
            voice=request.voice,
        )
        return StreamingResponse(io.BytesIO(response.read()), media_type="audio/wav")
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Spitch API error: {e}")

@app.post("/chat/", summary="Chat with the AI Assistant")
async def chat(request: ChatRequest):
    try:
        model = genai.GenerativeModel('gemini-1.5-flash')
        prompt = (
            f"You are a helpful and friendly assistant that speaks Nigerian languages. "
            f"Please respond in {request.language} to the following message: '{request.text}'"
        )
        gemini_response = await model.generate_content_async(prompt)
        ai_text = gemini_response.text

        tts_response = spitch_client.speech.generate(
            text=ai_text,
            language=request.language,
            voice="femi",
        )

        audio_content = tts_response.read()
        return StreamingResponse(io.BytesIO(audio_content), media_type="audio/wav")
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"An error occurred: {e}")

@app.get("/", summary="Health Check")
def read_root():
    return {"status": "ok"}
