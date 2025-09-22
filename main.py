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
origins = [
    "http://localhost:8080",  # For local development with Vite
    "http://localhost:3000",  # Common for other local dev servers
    # Add the URL of your deployed frontend here once it's live
]

app.add_middleware(
    CORSMiddleware,
    allow_origins=origins,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# --- Configuration ---
spitch_api_key = os.getenv("SPITCH_API_KEY")
gemini_api_key = os.getenv("GEMINI_API_KEY")

if not spitch_api_key:
    raise RuntimeError("SPITCH_API_KEY environment variable not set.")
if not gemini_api_key:
    raise RuntimeError("GEMINI_API_KEY environment variable not set.")

try:
    spitch_client = Spitch(api_key=spitch_api_key)
    genai.configure(api_key=gemini_api_key)
except Exception as e:
    raise RuntimeError(f"Failed to initialize API clients: {e}")

# --- Pydantic Models ---
class ChatRequest(BaseModel):
    text: str
    language: str

# --- Helper Function to map language codes ---
def get_spitch_language_code(lang: str) -> str:
    """Converts language codes like 'yo-NG' to the 'yo' format Spitch expects."""
    return lang.split('-')[0]

# --- API Endpoints ---
@app.post("/speech-to-text/", summary="Transcribe Audio to Text")
async def speech_to_text(file: UploadFile = File(...), language: str = "yo-NG"):
    if not file.content_type.startswith("audio/"):
        raise HTTPException(status_code=400, detail="Invalid file type. Please upload an audio file.")
    try:
        content = await file.read()
        spitch_lang = get_spitch_language_code(language)
        response = spitch_client.speech.transcribe(content=content, language=spitch_lang)
        return {"text": response.text}
    except Exception as e:
        print(f"Spitch STT Error: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/chat/", summary="Chat with the AI Assistant")
async def chat(request: ChatRequest):
    try:
        model = genai.GenerativeModel('gemini-1.5-flash')
        
        # We use the full 'yo-NG' style code for the Gemini prompt
        prompt = (
            "You are a helpful and friendly Naija AI assistant. "
            f"Please respond in the {request.language} language to the following message: '{request.text}'"
        )

        gemini_response = await model.generate_content_async(prompt)
        ai_text = gemini_response.text

        # Map voices based on the full language code
        voice_map = {
            "ha-NG": "hasan",
            "ig-NG": "ngozi",
        }
        selected_voice = voice_map.get(request.language, "femi")

        # Convert to the simple 'yo' style code for the Spitch API call
        spitch_lang = get_spitch_language_code(request.language)

        tts_response = spitch_client.speech.generate(
            text=ai_text,
            language=spitch_lang,
            voice=selected_voice,
        )

        audio_content = tts_response.read()
        return StreamingResponse(io.BytesIO(audio_content), media_type="audio/wav")
    except Exception as e:
        print(f"Chat Endpoint Error: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/", summary="Health Check", include_in_schema=False)
def read_root():
    return {"status": "ok", "message": "Naija AI Assistant is running!"}

