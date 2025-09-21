import os
from fastapi import FastAPI, UploadFile, File, HTTPException
from fastapi.responses import StreamingResponse
from pydantic import BaseModel
import google.generativeai as genai
from spitch import Spitch
import io

# Initialize FastAPI app
app = FastAPI()

# Configure API keys from environment variables
spitch_api_key = os.environ.get("SPITCH_API_KEY")
gemini_api_key = os.environ.get("GEMINI_API_KEY")

# Initialize Spitch and Gemini clients
if not spitch_api_key:
    raise ValueError("SPITCH_API_KEY environment variable not set.")
spitch_client = Spitch(api_key=spitch_api_key)

if not gemini_api_key:
    raise ValueError("GEMINI_API_KEY environment variable not set.")
genai.configure(api_key=gemini_api_key)

# Data models for API requests and responses
class TextToSpeechRequest(BaseModel):
    text: str
    language: str
    voice: str

class ChatRequest(BaseModel):
    text: str
    language: str

class ChatResponse(BaseModel):
    text: str
    audio_url: str

# API Endpoints
@app.post("/speech-to-text/", summary="Transcribe audio to text")
async def speech_to_text(file: UploadFile = File(...), language: str = "pcm"):
    """
    Transcribes audio to text using the Spitch API.
    """
    try:
        content = await file.read()
        response = spitch_client.speech.transcribe(
            content=content,
            language=language,
            model="mansa_v1"
        )
        return {"text": response.text}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/text-to-speech/", summary="Convert text to speech")
async def text_to_speech(request: TextToSpeechRequest):
    """
    Converts text to speech using the Spitch API.
    """
    try:
        response = spitch_client.speech.generate(
            text=request.text,
            language=request.language,
            voice=request.voice
        )
        return StreamingResponse(io.BytesIO(response.read()), media_type="audio/wav")
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/chat/", summary="Chat with the AI assistant")
async def chat(request: ChatRequest):
    """
    Handles the chat logic: gets a response from Gemini and converts it to speech.
    """
    try:
        # Get response from Gemini
        model = genai.GenerativeModel('gemini-1.5-flash')
        prompt = f"You are a helpful assistant. Respond in {request.language}. User: {request.text}"
        gemini_response = await model.generate_content_async(prompt)

        ai_text = gemini_response.text

        # Convert Gemini's response to speech using Spitch
        tts_response = spitch_client.speech.generate(
            text=ai_text,
            language=request.language,
            voice="femi"  # Defaulting to 'femi', you can change this
        )

        audio_content = tts_response.read()

        # For simplicity, we'll return the audio as a direct download link
        # In a real-world application, you would store this and provide a URL
        return StreamingResponse(io.BytesIO(audio_content), media_type="audio/wav")

    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
