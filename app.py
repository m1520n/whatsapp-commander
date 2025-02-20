from fastapi import FastAPI, Request, Response, status
import requests
import json
import logging
import ollama
from fastapi.middleware.cors import CORSMiddleware
from openai import OpenAI
from dotenv import load_dotenv
import os
import re
from duckduckgo_search import DDGS

# Load environment variables and don't use cache
load_dotenv(verbose=True, override=True)


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
WHATSAPP_PHONE_NUMBER_ID = os.getenv("WHATSAPP_PHONE_NUMBER_ID")
WHATSAPP_API_URL = f"https://graph.facebook.com/v17.0/{WHATSAPP_PHONE_NUMBER_ID}/messages"
WHATSAPP_API_TOKEN = os.getenv("WHATSAPP_API_TOKEN")
WEBHOOK_VERIFY_TOKEN = os.getenv("WEBHOOK_VERIFY_TOKEN")

@app.get("/")
async def root():
    return {"message": "Hello World"}

def search_duckduckgo(query):
    """
    Perform a web search using DuckDuckGo and return the top results.
    """
    results = DDGS().text(query, max_results=5)
    logger.info(f"DuckDuckGo Results: {results}")
    return results
    
def decide_if_search_needed(user_query):
    """
    Ask DeepSeek if the query requires a web search.
    """
    prompt = f"""
    You are an AI assistant. If you can confidently answer the question, do so. 
    Always include JSON in your response.
    The JSON should have a key "answer" and a key "search_required".
    If you need to search the web, set search_required to True.
    If you can answer the question confidently, set search_required to false.
    If you need to search the web, include "search_query" in the JSON that contains the user prompt converted to a search query.
    If you don't need to search the web, set "search_query" to null.

    example 1:
    Question: What is the capital of France?
    Answer: 
    ```json
    {{
        "answer": "The capital of France is Paris",
        "search_required": false,
        "search_query": null
    }}
    ```

    example 2:
    Question: What is the price of Bitcoin?
    Answer:
    ```json
    {{
        "answer": null,
        "search_required": true,
        "search_query": "price of bitcoin"
    }}
    ```

    example 3:
    Question: What is true,
    Answer:
    ```json
    {{
        "answer": null,
        "search_required": true,
        "search_query": "true"
    }}
    ```

    example 3:
    Question: Can you check the weather in Tokyo?
    Answer:
    ```json
    {{
        "answer": null,
        "search_required": true,
        "search_query": "weather in tokyo"
    }}
    ```

    example 4:
    Question: Can you summarize the news about the stock market?
    Answer:
    ```json
    {{
        "answer": null,
        "search_required": true,
        "search_query": "stock market news"
    }}
    ```
    
    Question: {user_query}
    """

    response = ollama.chat(model="deepseek-r1:7b", messages=[{"role": "user", "content": prompt}])
    answer = response["message"]["content"]

    logger.info(f"DeepSeek answer: {answer}")
    
    json_data = {"answer": None, "search_required": True, "search_query": None}
    json_match = re.search(r'```json(.*?)```', response['message']['content'], re.DOTALL)

    logger.info(f"JSON match: {json_match}")
    if json_match:
        try:
            extracted_data = json.loads(json_match.group(1))
            json_data.update(extracted_data)
            logger.info(f"Successfully extracted JSON: {json_data}")

            return extracted_data
        except json.JSONDecodeError:
            logger.error(f"Error decoding JSON: {answer}")
            return {
                "answer": None,
                "search_required": True,
                "search_query": {user_query}
        }

def clean_response(text):
    """Remove <think> sections and unwanted formatting from model responses."""
    return re.sub(r'<think>.*?</think>', '', text, flags=re.DOTALL).strip()

def summarize_search_results(query, results):
    """
    Use DeepSeek to summarize search results into a brief response.
    """
    if not results:
        return "I couldn't find any relevant search results."

    prompt = f"Summarize the following search results for: {query}\n\n{results}"

    response = ollama.chat(model="deepseek-r1:7b", messages=[{"role": "user", "content": prompt}])
    return clean_response(response['message']['content'])

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

        # Early return with 200 if no messages
        if "messages" not in data["entry"][0]["changes"][0]["value"]:
            logger.info("No messages found in the webhook data")
            return Response(
                content='{"status": "success", "message": "No messages found"}',
                media_type="application/json",
                status_code=status.HTTP_200_OK
            )

        message = data["entry"][0]["changes"][0]["value"]["messages"][0]
        sender_id = message["from"]
        message_type = message["type"]
        user_text = None

        logger.info(f"Processing {message_type} message from {sender_id}")

        if message_type == "text":
            user_text = message["text"]["body"].strip()
            logger.info(f"Text message received: {user_text}")

        elif message_type == "audio":
            try:
                # Get media ID and download audio
                media_id = message["audio"]["id"]

                logger.info(f"Downloading audio with ID: {media_id}")
                audio_content = download_whatsapp_media(media_id)
                audio_path = "temp_audio.ogg"

                with open(audio_path, "wb") as f:
                    f.write(audio_content)

                logger.info("Transcribing audio with Whisper...")
                user_text = transcribe_audio_with_openai(audio_path)

                logger.info(f"Transcription: {user_text}")
                os.remove(audio_path)  # Cleanup

            except Exception as e:
                logger.error(f"Error processing audio: {str(e)}")
                user_text = "Sorry, I couldn't process the audio message."
                os.remove(audio_path)  # Cleanup

        else:
            user_text = "I'm sorry, I can only process text and audio messages at the moment."
        
        # Step 1: Ask DeepSeek if it knows the answer
        logger.info(f"Deciding if search is needed for: {user_text}")
        deepseek_answer = decide_if_search_needed(user_text)
        logger.info(f"DeepSeek answer: {deepseek_answer}")

        if deepseek_answer["search_required"] == True:
          # Step 2: If DeepSeek requires a search, perform it
          search_query = deepseek_answer["search_query"] or user_text
          logger.info(f"Performing web search for: {search_query}")
          results = search_duckduckgo(search_query)
          logger.info(f"Search results: {results}")

          if results:
            summary = summarize_search_results(user_text, results)
            logger.info(f"Summarized search results: {summary}")
          else:
            summary = "I couldn't find any relevant search results."

          # Step 3: Send the summarized response back to WhatsApp
          send_whatsapp_message(sender_id, summary)
          return Response(
                content='{"status": "success", "message": "Answered using web search"}',
                media_type="application/json",
                status_code=status.HTTP_200_OK
            )

        else:
          # Step 4: If DeepSeek doesn't require a search, send the answer
          send_whatsapp_message(sender_id, deepseek_answer["answer"])
          return Response(
                content='{"status": "success", "message": "Answered using AI"}',
                media_type="application/json",
                status_code=status.HTTP_200_OK
            )

    except Exception as e:
        logger.error(f"Error processing webhook: {str(e)}")
        # Still return 200 even on error to prevent retries
        return Response(
            content='{"status": "error", "message": "Internal server error"}',
            media_type="application/json",
            status_code=status.HTTP_200_OK
        )
    
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