# Fake News Collector

A Flask application for reviewing claims and video content with source search, AI-assisted evidence summaries, speech transcription, frame OCR, and downloadable PDF reports.

## Deploy to Render

1. Push this repository to GitHub.
2. In Render, select **New** > **Blueprint** and connect the repository. Render reads `render.yaml` from the repository root.
3. Enter a Gemini API key when prompted for `GEMINI_API_KEY`. Render generates `FLASK_SECRET_KEY`.
4. Deploy the Blueprint and open the service URL when the build finishes.

The `Procfile` and `runtime.txt` are also included for deployment flows that use those files directly. `.python-version` pins Python to 3.11.9.

## Local setup

```powershell
py -3.11 -m venv .venv
.venv\Scripts\Activate.ps1
python -m pip install -r requirements.txt
Copy-Item .env.example .env
python app.py
```

Add `GEMINI_API_KEY` to `.env` for AI claim extraction and review. Never commit `.env` or service-account credentials.

## Deployment notes

- Video processing uses Whisper, EasyOCR, OpenCV, and FFmpeg. These dependencies use significant build time and memory; select a Render instance with enough RAM for media analysis. Text-only reviews need substantially fewer resources.
- Uploaded files and in-memory PDF report links are temporary. Render's local filesystem is not persistent across deploys or restarts.
- Reverse image matching requires Google Cloud Vision credentials configured securely in Render. Without them, the report notes that reverse matching was unavailable.
- Results depend on third-party APIs and available captions. Treat verdicts and confidence scores as provisional, and follow the linked evidence.