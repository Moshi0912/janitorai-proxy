# JanitorAI Proxy → Gemini

Personal proxy for JanitorAI that routes to Google Gemini API.

## Setup

### 1. Get your keys
- **Gemini API Key:** https://aistudio.google.com/apikey
- **Proxy API Key:** Make up your own (e.g., `my-secret-key-123`)

### 2. Deploy on Render (free)
1. Push this folder to a GitHub repo
2. Go to https://render.com → New Web Service
3. Connect your GitHub repo
4. Set environment variables:
   - `GEMINI_API_KEY` = your Google AI Studio key
   - `PROXY_API_KEY` = your chosen proxy key
   - `GEMINI_MODEL` = `gemini-2.0-flash` (or your preferred model)
5. Deploy — takes ~2 minutes

### 3. Plug into JanitorAI
- **Proxy URL:** `https://your-app.onrender.com/v1/chat/completions`
- **API Key:** whatever you set as `PROXY_API_KEY`
- **Model:** `gemini-2.0-flash` (or whatever you set)

## Local testing
```bash
pip install -r requirements.txt
export GEMINI_API_KEY="your-gemini-key"
export PROXY_API_KEY="your-proxy-key"
python app.py
```

Then test:
```bash
curl -X POST http://localhost:10000/v1/chat/completions \
  -H "Authorization: Bearer your-proxy-key" \
  -H "Content-Type: application/json" \
  -d '{"messages": [{"role": "user", "content": "Hello!"}]}'
```
