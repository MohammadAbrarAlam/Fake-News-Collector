from flask import Flask, abort, render_template, request, send_file, url_for
from dotenv import load_dotenv
import os
import uuid
from typing import Any, TypedDict, cast
from urllib.parse import parse_qs, urlparse

from tools import analyze_submission, create_pdf_report

load_dotenv()

app = Flask(__name__)
app.secret_key = os.getenv("FLASK_SECRET_KEY") or os.urandom(32)
app.config["MAX_CONTENT_LENGTH"] = 150 * 1024 * 1024

UPLOAD_FOLDER = os.path.join(app.root_path, "uploads")
os.makedirs(UPLOAD_FOLDER, exist_ok=True)
REPORTS = {}
ALLOWED_EXTENSIONS = {"mp4", "mov", "mkv", "webm", "avi"}


class _DownloadOptions(TypedDict):
    outtmpl: str
    format: str
    noplaylist: bool
    quiet: bool
    no_warnings: bool
    max_filesize: int


@app.route("/")
def home():
    return render_template("index.html")

@app.route("/analyze", methods=["POST"])
def analyze():
    uploaded = request.files.get("video")
    video_url = request.form.get("video_url", "").strip()
    claim_text = request.form.get("claim_text", "").strip()
    language = request.form.get("language", "en")
    if language not in {"en", "hi"}:
        language = "en"

    video_path = None
    youtube_transcript = ""
    try:
        if uploaded and uploaded.filename:
            extension = uploaded.filename.rsplit(".", 1)[-1].lower() if "." in uploaded.filename else ""
            if extension not in ALLOWED_EXTENSIONS:
                raise ValueError("Choose a video file in MP4, MOV, MKV, WEBM, or AVI format.")
            video_path = os.path.join(UPLOAD_FOLDER, f"{uuid.uuid4().hex}.{extension}")
            uploaded.save(video_path)
        elif video_url:
            parsed_url = urlparse(video_url)
            if parsed_url.scheme not in {"http", "https"} or not parsed_url.netloc:
                raise ValueError("Enter a valid HTTP or HTTPS video URL.")
            host = (parsed_url.hostname or "").lower().rstrip(".")
            allowed_hosts = {
                "youtube.com", "youtu.be", "vimeo.com", "dailymotion.com",
                "facebook.com", "instagram.com", "tiktok.com", "x.com", "twitter.com",
            }
            if not any(host == domain or host.endswith(f".{domain}") for domain in allowed_hosts):
                raise ValueError("Video URL downloads are supported from YouTube, Vimeo, Dailymotion, Facebook, Instagram, TikTok, and X.")
            is_youtube = host in {"youtube.com", "youtu.be"} or host.endswith(".youtube.com")
            youtube_transcript = _youtube_transcript(video_url, language) if is_youtube else ""
            if youtube_transcript:
                claim_text = "\n".join(part for part in [claim_text, youtube_transcript] if part)
            try:
                video_path = _download_video(video_url)
            except ValueError as download_error:
                if not is_youtube:
                    raise
                if not youtube_transcript:
                    raise ValueError(
                        f"{download_error} YouTube captions were also unavailable; paste the claim or transcript to continue."
                    ) from download_error

        if not claim_text and not video_path:
            raise ValueError("Enter a claim, upload a video, or provide a video URL.")

        report = analyze_submission(claim_text, video_path=video_path, language=language)
        if youtube_transcript:
            report["transcript"] = report.get("transcript") or youtube_transcript
        report_id = uuid.uuid4().hex
        REPORTS[report_id] = report
        return render_template(
            "index.html", report=report, report_url=url_for("download_report", report_id=report_id),
            language=language, claim_text=claim_text, video_url=video_url,
        )
    except ValueError as error:
        return render_template(
            "index.html", error=str(error), language=language,
            claim_text=claim_text, video_url=video_url,
        ), 400
    except Exception as error:
        app.logger.exception("Analysis failed")
        return render_template(
            "index.html", error="Analysis could not be completed. Check service credentials and installed dependencies, then retry.", language=language,
            claim_text=claim_text, video_url=video_url,
        ), 500
    finally:
        if video_path and os.path.exists(video_path):
            os.remove(video_path)


@app.route("/reports/<report_id>.pdf")
def download_report(report_id):
    report = REPORTS.get(report_id)
    if not report:
        abort(404)
    return send_file(
        create_pdf_report(report),
        mimetype="application/pdf",
        as_attachment=True,
        download_name="fact-check-report.pdf",
    )


def _download_video(video_url):
    try:
        import yt_dlp
    except ImportError as error:
        raise ValueError("URL downloads require yt-dlp. Install the project requirements first.") from error

    target = os.path.join(UPLOAD_FOLDER, f"{uuid.uuid4().hex}.%(ext)s")
    options: _DownloadOptions = {
        "outtmpl": target,
        "format": "bestvideo*+bestaudio/best",
        "noplaylist": True,
        "quiet": True,
        "no_warnings": True,
        "max_filesize": 150 * 1024 * 1024,
    }
    try:
        with yt_dlp.YoutubeDL(cast(Any, options)) as downloader:
            info = downloader.extract_info(video_url, download=True)
            path = downloader.prepare_filename(info)
    except Exception as error:
        raise ValueError(f"The video URL could not be downloaded: {error}") from error
    if not os.path.exists(path):
        raise ValueError("The video downloader did not produce a supported video file.")
    return path


def _youtube_transcript(video_url, language):
    parsed_url = urlparse(video_url)
    host = (parsed_url.hostname or "").lower()
    if host == "youtu.be":
        video_id = parsed_url.path.strip("/").split("/", 1)[0]
    else:
        video_id = parse_qs(parsed_url.query).get("v", [""])[0]
        if not video_id:
            path_parts = [part for part in parsed_url.path.split("/") if part]
            if len(path_parts) > 1 and path_parts[0] in {"shorts", "embed", "live"}:
                video_id = path_parts[1]
    if not video_id:
        return ""

    try:
        from youtube_transcript_api import YouTubeTranscriptApi

        language_codes = ["hi", "en"] if language == "hi" else ["en", "hi"]
        snippets = YouTubeTranscriptApi().fetch(video_id, languages=language_codes)
        return " ".join(snippet.text.strip() for snippet in snippets if snippet.text.strip())
    except Exception:
        return ""

if __name__ == "__main__":
    app.run(debug=True)