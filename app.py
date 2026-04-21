"""
JanitorAI Proxy Server
Forwards requests to Google Gemini API in OpenAI-compatible format.
Deploy on Render free tier.
"""

from flask import Flask, request, jsonify, Response
import requests
import os
import json
import uuid

app = Flask(__name__)

@app.after_request
def add_cors(response):
    response.headers['Access-Control-Allow-Origin'] = '*'
    response.headers['Access-Control-Allow-Methods'] = 'GET, POST, OPTIONS'
    response.headers['Access-Control-Allow-Headers'] = 'Content-Type, Authorization'
    return response

GEMINI_API_KEY = os.environ.get("GEMINI_API_KEY", "")
GEMINI_MODEL = os.environ.get("GEMINI_MODEL", "gemini-3-flash-preview")
VALID_API_KEY = os.environ.get("PROXY_API_KEY", "changeme")


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
                        import base64
                        header, data = url.split(",", 1)
                        mime = header.split(":")[1].split(";")[0]
                        parts.append({"inline_data": {"mime_type": mime, "data": data}})
                    else:
                        parts.append({"text": f"[Image: {url}]"})
            contents.append({"role": gemini_role, "parts": parts})
    
    return contents, system_instruction


def make_openai_response(gemini_data, model_name):
    """Convert Gemini response to OpenAI format."""
    cid = "chatcmpl-" + str(uuid.uuid4())[:12]
    candidates = gemini_data.get("candidates", [])
    
    if not candidates:
        return {
            "id": cid, "object": "chat.completion", "model": model_name,
            "choices": [{"index": 0, "message": {"role": "assistant", "content": ""}, "finish_reason": "stop"}],
            "usage": {"prompt_tokens": 0, "completion_tokens": 0, "total_tokens": 0}
        }
    
    candidate = candidates[0]
    content_parts = candidate.get("content", {}).get("parts", [])
    text = "".join([p.get("text", "") for p in content_parts])
    usage = gemini_data.get("usageMetadata", {})
    
    return {
        "id": cid, "object": "chat.completion", "model": model_name,
        "choices": [{"index": 0, "message": {"role": "assistant", "content": text}, "finish_reason": "stop"}],
        "usage": {
            "prompt_tokens": usage.get("promptTokenCount", 0),
            "completion_tokens": usage.get("candidatesTokenCount", 0),
            "total_tokens": usage.get("totalTokenCount", 0)
        }
    }


def make_stream_chunks(text, model_name):
    """Generate SSE stream chunks for OpenAI compatibility."""
    cid = "chatcmpl-" + str(uuid.uuid4())[:12]
    words = text.split()
    
    for i, word in enumerate(words):
        delta = {"content": word + (" " if i < len(words) - 1 else "")}
        if i == 0:
            delta["role"] = "assistant"
        chunk = {
            "id": cid, "object": "chat.completion.chunk", "model": model_name,
            "choices": [{"index": 0, "delta": delta, "finish_reason": None}]
        }
        yield f"data: {json.dumps(chunk)}\n\n"
    
    # Final chunk with finish_reason
    final = {
        "id": cid, "object": "chat.completion.chunk", "model": model_name,
        "choices": [{"index": 0, "delta": {}, "finish_reason": "stop"}]
    }
    yield f"data: {json.dumps(final)}\n\n"
    yield "data: [DONE]\n\n"


@app.route("/v1/chat/completions", methods=["POST", "OPTIONS"])
def chat_completions():
    if request.method == "OPTIONS":
        return jsonify({"status": "ok"}), 200
    
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
    requested_model = data.get("model", GEMINI_MODEL)
    
    contents, system_instruction = convert_messages_to_gemini(messages)
    
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
        
        if stream:
            openai_resp = make_openai_response(gemini_data, requested_model)
            text = openai_resp["choices"][0]["message"]["content"]
            return Response(make_stream_chunks(text, requested_model), mimetype="text/event-stream")
        else:
            return jsonify(make_openai_response(gemini_data, requested_model))
    
    except requests.exceptions.Timeout:
        return jsonify({"error": {"message": "Request timed out", "type": "timeout"}}), 504
    except Exception as e:
        return jsonify({"error": {"message": str(e), "type": "server_error"}}), 500


@app.route("/v1/models", methods=["GET", "OPTIONS"])
def list_models():
    if request.method == "OPTIONS":
        return jsonify({"status": "ok"}), 200
    
    auth = request.headers.get("Authorization", "")
    if auth.replace("Bearer ", "") != VALID_API_KEY:
        return jsonify({"error": {"message": "Invalid API key", "type": "auth_error"}}), 401
    
    return jsonify({
        "object": "list",
        "data": [
            {"id": GEMINI_MODEL, "object": "model", "owned_by": "google"},
            {"id": "gemini-2.0-flash", "object": "model", "owned_by": "google"},
            {"id": "gemini-2.5-flash", "object": "model", "owned_by": "google"},
            {"id": "gemini-2.5-pro", "object": "model", "owned_by": "google"},
        ]
    })


@app.route("/", methods=["GET"])
def health():
    return jsonify({"status": "ok", "service": "janitorai-proxy", "model": GEMINI_MODEL})


if __name__ == "__main__":
    app.run(host="0.0.0.0", port=int(os.environ.get("PORT", 10000)))
