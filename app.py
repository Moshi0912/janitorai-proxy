"""
JanitorAI Proxy Server
Forwards requests to Google Gemini API in OpenAI-compatible format.
Deploy on Render free tier.
"""

from flask import Flask, request, jsonify
import requests
import os

app = Flask(__name__)

GEMINI_API_KEY = os.environ.get("GEMINI_API_KEY", "")
GEMINI_MODEL = os.environ.get("GEMINI_MODEL", "gemini-3-flash-preview")
VALID_API_KEY = os.environ.get("PROXY_API_KEY", "changeme")

# Gemini to OpenAI message mapping
def convert_messages_to_gemini(messages):
    """Convert OpenAI-style messages to Gemini contents format."""
    contents = []
    system_instruction = None
    
    for msg in messages:
        role = msg.get("role", "user")
        content = msg.get("content", "")
        
        if role == "system":
            system_instruction = {"parts": [{"text": content}]}
            continue
        
        # Gemini uses "user" and "model" roles
        gemini_role = "user" if role in ("user", "system") else "model"
        
        if isinstance(content, str):
            contents.append({"role": gemini_role, "parts": [{"text": content}]})
        elif isinstance(content, list):
            parts = []
            for part in content:
                if part.get("type") == "text":
                    parts.append({"text": part["text"]})
                elif part.get("type") == "image_url":
                    url = part["image_url"]["url"]
                    if url.startswith("data:"):
                        # Base64 image
                        import base64
                        header, data = url.split(",", 1)
                        mime = header.split(":")[1].split(";")[0]
                        parts.append({"inline_data": {"mime_type": mime, "data": data}})
                    else:
                        parts.append({"text": f"[Image: {url}]"})
            contents.append({"role": gemini_role, "parts": parts})
    
    return contents, system_instruction


def gemini_response_to_openai(gemini_response, model_name):
    """Convert Gemini API response to OpenAI format."""
    try:
        candidates = gemini_response.get("candidates", [])
        if not candidates:
            return {
                "id": "chatcmpl-" + gemini_response.get("responseId", "unknown"),
                "object": "chat.completion",
                "model": model_name,
                "choices": [{
                    "index": 0,
                    "message": {"role": "assistant", "content": ""},
                    "finish_reason": "stop"
                }],
                "usage": {"prompt_tokens": 0, "completion_tokens": 0, "total_tokens": 0}
            }
        
        candidate = candidates[0]
        content_parts = candidate.get("content", {}).get("parts", [])
        text = "".join([p.get("text", "") for p in content_parts])
        
        usage = gemini_response.get("usageMetadata", {})
        
        return {
            "id": "chatcmpl-" + gemini_response.get("responseId", "unknown"),
            "object": "chat.completion",
            "model": model_name,
            "choices": [{
                "index": 0,
                "message": {"role": "assistant", "content": text},
                "finish_reason": "stop"
            }],
            "usage": {
                "prompt_tokens": usage.get("promptTokenCount", 0),
                "completion_tokens": usage.get("candidatesTokenCount", 0),
                "total_tokens": usage.get("totalTokenCount", 0)
            }
        }
    except Exception as e:
        return {
            "id": "chatcmpl-error",
            "object": "chat.completion",
            "model": model_name,
            "choices": [{
                "index": 0,
                "message": {"role": "assistant", "content": f"Error parsing response: {str(e)}"},
                "finish_reason": "error"
            }],
            "usage": {"prompt_tokens": 0, "completion_tokens": 0, "total_tokens": 0}
        }


@app.route("/v1/chat/completions", methods=["POST"])
def chat_completions():
    # Validate API key
    auth = request.headers.get("Authorization", "")
    if auth.replace("Bearer ", "") != VALID_API_KEY:
        return jsonify({"error": {"message": "Invalid API key", "type": "auth_error"}}), 401
    
    data = request.json
    if not data:
        return jsonify({"error": {"message": "No JSON body", "type": "invalid_request"}}), 400
    
    messages = data.get("messages", [])
    temperature = data.get("temperature", 0.7)
    max_tokens = data.get("max_tokens", 4096)
    stream = data.get("stream", False)
    
    # Use model from request or default
    requested_model = data.get("model", GEMINI_MODEL)
    
    # Convert messages to Gemini format
    contents, system_instruction = convert_messages_to_gemini(messages)
    
    # Build Gemini request
    gemini_url = f"https://generativelanguage.googleapis.com/v1beta/models/{GEMINI_MODEL}:generateContent?key={GEMINI_API_KEY}"
    
    gemini_payload = {
        "contents": contents,
        "generationConfig": {
            "temperature": temperature,
            "maxOutputTokens": max_tokens,
        }
    }
    
    if system_instruction:
        gemini_payload["systemInstruction"] = system_instruction
    
    try:
        resp = requests.post(gemini_url, json=gemini_payload, timeout=120)
        gemini_data = resp.json()
        
        if "error" in gemini_data:
            return jsonify({
                "error": {
                    "message": gemini_data["error"].get("message", "Gemini API error"),
                    "type": "api_error"
                }
            }), resp.status_code
        
        openai_response = gemini_response_to_openai(gemini_data, requested_model)
        return jsonify(openai_response)
    
    except requests.exceptions.Timeout:
        return jsonify({"error": {"message": "Request timed out", "type": "timeout"}}), 504
    except Exception as e:
        return jsonify({"error": {"message": str(e), "type": "server_error"}}), 500


@app.route("/v1/models", methods=["GET"])
def list_models():
    auth = request.headers.get("Authorization", "")
    if auth.replace("Bearer ", "") != VALID_API_KEY:
        return jsonify({"error": {"message": "Invalid API key", "type": "auth_error"}}), 401
    
    return jsonify({
        "object": "list",
        "data": [
            {"id": GEMINI_MODEL, "object": "model"},
            {"id": "gemini-2.0-flash", "object": "model"},
            {"id": "gemini-2.5-flash", "object": "model"},
            {"id": "gemini-2.5-pro", "object": "model"},
        ]
    })


@app.route("/", methods=["GET"])
def health():
    return jsonify({"status": "ok", "service": "janitorai-proxy", "model": GEMINI_MODEL})


if __name__ == "__main__":
    app.run(host="0.0.0.0", port=int(os.environ.get("PORT", 10000)))
