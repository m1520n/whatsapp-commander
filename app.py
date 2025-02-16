from fastapi import FastAPI, Request
import requests
import json
import logging
import ollama
from fastapi.middleware.cors import CORSMiddleware
from openai import OpenAI
from dotenv import load_dotenv
import os

# Load environment variables
load_dotenv()

client = OpenAI(
    api_key=os.getenv("OPENAI_API_KEY")
)

def transcribe_audio_with_openai(audio_path):
    audio_file= open(audio_path, "rb")
    transcription = client.audio.transcriptions.create(
      model="whisper-1", 
      file=audio_file
    )
    return transcription.text


app = FastAPI()

# Add CORS middleware to allow requests from ngrok
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Allows all origins
    allow_credentials=True,
    allow_methods=["*"],  # Allows all methods
    allow_headers=["*"],  # Allows all headers
)

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# API Configuration from environment variables
WHATSAPP_PHONE_NUMBER_ID = os.getenv("WHATSAPP_PHONE_NUMBER")
WHATSAPP_API_URL = f"https://graph.facebook.com/v17.0/{WHATSAPP_PHONE_NUMBER_ID}/messages"
WHATSAPP_API_TOKEN = os.getenv("WHATSAPP_API_TOKEN")
WEBHOOK_VERIFY_TOKEN = os.getenv("WEBHOOK_VERIFY_TOKEN")

@app.get("/")
async def root():
    return {"message": "Hello World"}

# app.get("/messaging-webhook", (req, res) => {
  
# // Parse the query params
#   let mode = req.query["hub.mode"];
#   let token = req.query["hub.verify_token"];
#   let challenge = req.query["hub.challenge"];

#   // Check if a token and mode is in the query string of the request
#   if (mode && token) {
#     // Check the mode and token sent is correct
#     if (mode === "subscribe" && token === config.verifyToken) {
#       // Respond with the challenge token from the request
#       console.log("WEBHOOK_VERIFIED");
#       res.status(200).send(challenge);
#     } else {
#       // Respond with '403 Forbidden' if verify tokens do not match
#       res.sendStatus(403);
#     }
#   }
# });

@app.get("/callback")
async def callback(request: Request):
    mode = request.query_params.get("hub.mode")
    token = request.query_params.get("hub.verify_token")
    challenge = request.query_params.get("hub.challenge")

    if mode and token:
        if mode == "subscribe" and token == WEBHOOK_VERIFY_TOKEN:
            return int(challenge)
        else:
            return "Verification failed", 403
    else:
        return "Missing parameters", 400

def download_whatsapp_media(media_id):
    try:
        # First, get the media URL
        media_url = f"https://graph.facebook.com/v17.0/{media_id}"
        headers = {
            "Authorization": f"Bearer {WHATSAPP_API_TOKEN}",
        }
        
        logger.info(f"Requesting media URL for ID: {media_id}")
        response = requests.get(media_url, headers=headers)
        response.raise_for_status()
        
        media_data = response.json()
        if 'url' not in media_data:
            raise ValueError("Media URL not found in response")
            
        # Download the actual media file
        media_download_url = media_data['url']
        logger.info(f"Downloading media from URL: {media_download_url}")
        
        media_response = requests.get(media_download_url, headers=headers)
        media_response.raise_for_status()
        
        return media_response.content
    except Exception as e:
        logger.error(f"Error downloading media: {str(e)}")
        raise

@app.post("/callback")
async def whatsapp_webhook(request: Request):
    try:
        data = await request.json()
        logger.info(f"Received webhook data: {json.dumps(data, indent=2)}")

        message = data["entry"][0]["changes"][0]["value"]["messages"][0]
        sender_id = message["from"]
        message_type = message["type"]
        user_text = None

        logger.info(f"Processing {message_type} message from {sender_id}")

        if message_type == "text":
            user_text = message["text"]["body"]
            logger.info(f"Text message: {user_text}")
          
        elif message_type == "audio":
            try:
                # Get the media ID from the audio message
                media_id = message["audio"]["id"]
                logger.info(f"Processing audio with ID: {media_id}")
                
                # Download the audio file
                audio_content = download_whatsapp_media(media_id)
                
                # Save the audio temporarily
                audio_path = "temp_audio.ogg"
                with open(audio_path, "wb") as f:
                    f.write(audio_content)
                
                logger.info("Transcribing audio with Whisper...")
                # Use Ollama's Whisper model for transcription
                user_text = transcribe_audio_with_openai(audio_path)

                logger.info(f"Transcription: {user_text}")
                
                # Clean up the temporary file
                import os
                os.remove(audio_path)
                
            except Exception as e:
                logger.error(f"Error processing audio: {str(e)}")
                user_text = "Sorry, I couldn't process the audio message."

        elif message_type == "image":
            user_text = "I received your image, but I'm currently configured to handle only text and audio messages."
            logger.info("Image received but not processed")

        elif message_type == "video":
            user_text = "I received your video, but I'm currently configured to handle only text and audio messages."
            logger.info("Video received but not processed")
        
        if not user_text:
            user_text = "I'm sorry, I couldn't process that type of message."

        # Send the query to DeepSeek
        prompt = f"You are a helpful assistant. Answer the following question or respond to this message: {user_text}"
        bot_response = ollama.chat(model="deepseek-r1:7b", messages=[{"role": "user", "content": prompt}])
        
        response_text = bot_response['message']['content']
        logger.info(f"Bot response: {response_text}")

        # Send response back to WhatsApp
        send_whatsapp_message(sender_id, response_text)
        
        return {"status": "success"}

    except KeyError as e:
        logger.error(f"Invalid webhook data structure: {str(e)}")
        logger.error(f"Received data: {json.dumps(data, indent=2)}")
        return {"status": "error", "message": "Invalid webhook data structure"}
    except Exception as e:
        logger.error(f"Error processing webhook: {str(e)}")
        logger.error(f"Full error: {str(e)}")
        return {"status": "error", "message": "Internal server error"}

def send_whatsapp_message(to, text):
    try:
        headers = {
            "Authorization": f"Bearer {WHATSAPP_API_TOKEN}",
            "Content-Type": "application/json",
        }

        payload = {
            "messaging_product": "whatsapp",
            "recipient_type": "individual",
            "to": to,
            "type": "text",
            "text": {
                "preview_url": False,
                "body": text
            }
        }

        # Log the request for debugging
        logger.info(f"Sending message to WhatsApp API:")
        logger.info(f"URL: {WHATSAPP_API_URL}")
        logger.info(f"Payload: {json.dumps(payload, indent=2)}")

        # Send the payload to WhatsApp
        response = requests.post(WHATSAPP_API_URL, json=payload, headers=headers)
        
        # Log the response for debugging
        logger.info(f"WhatsApp API Response Status: {response.status_code}")
        logger.info(f"WhatsApp API Response: {response.text}")
        
        response.raise_for_status()
        logger.info(f"Message sent successfully to {to}")
    except requests.exceptions.RequestException as e:
        logger.error(f"WhatsApp API error: {str(e)}")
        logger.error(f"Response content: {e.response.text if hasattr(e, 'response') else 'No response content'}")
        raise

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)