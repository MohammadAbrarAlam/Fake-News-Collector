import importlib
import json
import os
import re
from functools import lru_cache
from urllib.parse import urlparse


TRUSTED_SOURCES = {
    "World Health Organization": "who.int",
    "Press Information Bureau": "pib.gov.in",
    "Reuters Fact Check": "reuters.com/fact-check",
    "AFP Fact Check": "factcheck.afp.com",
}


def transcribe_video(video_path, language):
    result = _load_whisper().transcribe(video_path, language=language or None)
    text = result.get("text", "")
    return text.strip() if isinstance(text, str) else ""


@lru_cache(maxsize=1)
def _load_whisper():
    import whisper

    return whisper.load_model("base")


def inspect_video_frames(video_path, language):
    warnings = []
    recognized_text = []
    frame_notes = []
    reverse_matches = []

    try:
        cv2 = importlib.import_module("cv2")
        easyocr = importlib.import_module("easyocr")
    except ImportError:
        return recognized_text, frame_notes, [
            "Frame OCR is unavailable. Install opencv-python-headless and easyocr."
        ], reverse_matches

    capture = cv2.VideoCapture(video_path)
    if not capture.isOpened():
        return recognized_text, frame_notes, ["The video could not be opened for frame analysis."], reverse_matches

    frame_rate = capture.get(cv2.CAP_PROP_FPS) or 1
    frame_count = capture.get(cv2.CAP_PROP_FRAME_COUNT) or 0
    duration = frame_count / frame_rate if frame_count else 0
    sample_times = sorted(set([0, int(duration / 3), int(2 * duration / 3)]))

    try:
        _load_vision_client()
        vision_available = True
    except Exception:
        vision_available = False
        warnings.append(
            "Reverse image matching requires Google Cloud Vision credentials; no matching pages were retrieved."
        )

    try:
        reader = easyocr.Reader(["en", "hi"] if language == "hi" else ["en"], gpu=False)
        for seconds in sample_times:
            capture.set(cv2.CAP_PROP_POS_MSEC, seconds * 1000)
            success, frame = capture.read()
            if not success:
                continue
            detections = reader.readtext(frame, detail=0, paragraph=True)
            text = " ".join(item.strip() for item in detections if item.strip())
            if text:
                recognized_text.append(text)
            note = {"time_seconds": seconds, "text": text, "reverse_matches": []}
            if vision_available:
                try:
                    encoded, image_buffer = cv2.imencode(".jpg", frame)
                    if encoded:
                        note["reverse_matches"] = reverse_image_search(image_buffer.tobytes())
                        reverse_matches.extend(note["reverse_matches"])
                except Exception as error:
                    warnings.append(f"Reverse image search failed for a sampled frame: {error}")
            frame_notes.append(note)
    except Exception as error:
        warnings.append(f"Frame OCR could not complete: {error}")
    finally:
        capture.release()

    if vision_available and not reverse_matches:
        warnings.append("No visually matching web pages were found for the sampled frames.")
    warnings.append("Automated frame review is not a validated deepfake detector; visual authenticity is inconclusive.")
    return recognized_text, frame_notes, list(dict.fromkeys(warnings)), reverse_matches


def reverse_image_search(image_bytes):
    vision, client = _load_vision_client()
    response = client.web_detection(image=vision.Image(content=image_bytes))
    if response.error.message:
        raise RuntimeError(response.error.message)

    matches = []
    for page in response.web_detection.pages_with_matching_images[:5]:
        if page.url:
            matches.append({
                "title": page.page_title or page.url,
                "url": page.url,
                "source": _source_for_url(page.url),
            })
    return matches


@lru_cache(maxsize=1)
def _load_vision_client():
    vision = importlib.import_module("google.cloud.vision")

    return vision, vision.ImageAnnotatorClient()


def search_claim(claim, max_results=8):
    from ddgs import DDGS

    results = []
    seen_urls = set()
    queries: list[tuple[str, str | None]] = [(claim, None)]
    queries.extend(
        (f"{claim} site:{domain}", source)
        for source, domain in TRUSTED_SOURCES.items()
    )

    with DDGS() as ddgs:
        for query, requested_source in queries:
            try:
                matches = ddgs.text(query, max_results=max_results if not requested_source else 3)
                for match in matches:
                    url = match.get("href") or match.get("url")
                    if not url or url in seen_urls:
                        continue
                    seen_urls.add(url)
                    source = _source_for_url(url)
                    if requested_source and source != requested_source:
                        continue
                    results.append({
                        "title": match.get("title", "Source"),
                        "url": url,
                        "source": source,
                        "snippet": match.get("body", ""),
                        "trusted": source in TRUSTED_SOURCES,
                    })
            except Exception:
                continue
    return results


def analyze_submission(text, video_path=None, language="en"):
    warnings = []
    transcript = ""
    frame_text = []
    frame_notes = []
    reverse_matches = []

    if video_path:
        try:
            transcript = transcribe_video(video_path, language)
        except Exception as error:
            warnings.append(f"Speech transcription was unavailable: {error}")
        ocr_text, frame_notes, ocr_warnings, reverse_matches = inspect_video_frames(video_path, language)
        frame_text.extend(ocr_text)
        warnings.extend(ocr_warnings)

    material = "\n".join(part for part in [text.strip(), transcript, *frame_text] if part)
    if not material:
        raise ValueError(
            "No speech or readable frame text was found. Paste a claim or transcript, "
            "or use a video with clear speech or on-screen text."
        )

    claims = _extract_claims(material, language)
    claim_results = []
    for claim in claims:
        evidence = search_claim(claim)
        claim_results.append({"claim": claim, "evidence": evidence})

    review = _review_with_ai(claim_results, language)
    if review:
        claims_with_verdicts = _merge_review(claim_results, review)
        summary = review.get("summary", "Evidence gathered; review the linked sources.")
        ai_used = True
    else:
        claims_with_verdicts = [
            {
                **item,
                "verdict": "Insufficient evidence",
                "confidence": min(35, 10 + sum(source["trusted"] for source in item["evidence"]) * 5),
                "explanation": "AI review is unavailable. Search results are provided for manual verification.",
            }
            for item in claim_results
        ]
        summary = "Automated assessment is unavailable; use the linked evidence for manual verification."
        ai_used = False

    trusted_count = len({
        evidence["url"]
        for item in claims_with_verdicts
        for evidence in item["evidence"]
        if evidence["trusted"]
    })
    return {
        "summary": summary,
        "claims": claims_with_verdicts,
        "transcript": transcript,
        "frame_text": frame_text,
        "frame_notes": frame_notes,
        "reverse_matches": reverse_matches,
        "warnings": warnings,
        "trusted_source_count": trusted_count,
        "ai_used": ai_used,
        "language": language,
    }


def create_pdf_report(report):
    from io import BytesIO
    from xml.sax.saxutils import escape

    colors = importlib.import_module("reportlab.lib.colors")
    enums = importlib.import_module("reportlab.lib.enums")
    pagesizes = importlib.import_module("reportlab.lib.pagesizes")
    styles_module = importlib.import_module("reportlab.lib.styles")
    units = importlib.import_module("reportlab.lib.units")
    pdfmetrics = importlib.import_module("reportlab.pdfbase.pdfmetrics")
    ttfonts = importlib.import_module("reportlab.pdfbase.ttfonts")
    platypus = importlib.import_module("reportlab.platypus")
    Paragraph = platypus.Paragraph
    SimpleDocTemplate = platypus.SimpleDocTemplate
    ParagraphStyle = styles_module.ParagraphStyle
    getSampleStyleSheet = styles_module.getSampleStyleSheet
    TTFont = ttfonts.TTFont
    A4 = pagesizes.A4
    TA_LEFT = enums.TA_LEFT
    mm = units.mm

    font_name = "Helvetica"
    font_bold = "Helvetica-Bold"
    font_candidates = [
        ("C:/Windows/Fonts/Nirmala.ttf", "C:/Windows/Fonts/NirmalaB.ttf"),
        ("C:/Windows/Fonts/arial.ttf", "C:/Windows/Fonts/arialbd.ttf"),
    ]
    for regular_path, bold_path in font_candidates:
        if os.path.exists(regular_path):
            pdfmetrics.registerFont(TTFont("ReportFont", regular_path))
            font_name = "ReportFont"
            if os.path.exists(bold_path):
                pdfmetrics.registerFont(TTFont("ReportFont-Bold", bold_path))
                font_bold = "ReportFont-Bold"
            break

    buffer = BytesIO()
    document = SimpleDocTemplate(
        buffer,
        pagesize=A4,
        rightMargin=18 * mm,
        leftMargin=18 * mm,
        topMargin=18 * mm,
        bottomMargin=18 * mm,
        title="Fact-check report",
    )
    styles = getSampleStyleSheet()
    styles.add(ParagraphStyle(
        name="ReportTitle", parent=styles["Title"], fontName=font_bold,
        textColor=colors.HexColor("#205c46"), alignment=TA_LEFT, spaceAfter=8 * mm,
    ))
    styles.add(ParagraphStyle(
        name="ReportBody", parent=styles["BodyText"], fontName=font_name,
        leading=15, spaceAfter=3 * mm,
    ))
    styles.add(ParagraphStyle(
        name="ReportHeading", parent=styles["Heading2"], fontName=font_bold,
        textColor=colors.HexColor("#205c46"), spaceBefore=5 * mm, spaceAfter=2 * mm,
    ))
    story = [
        Paragraph("Fact-check report", styles["ReportTitle"]),
        Paragraph(escape(report["summary"]), styles["ReportBody"]),
    ]
    for item in report["claims"]:
        story.extend([
            Paragraph(escape(item["claim"]), styles["ReportHeading"]),
            Paragraph(
                f"<b>{escape(item['verdict'])}</b> | Confidence: {item['confidence']}%",
                styles["ReportBody"],
            ),
            Paragraph(escape(item["explanation"]), styles["ReportBody"]),
        ])
        for source in item["evidence"]:
            story.append(Paragraph(
                f"{escape(source['source'])}: <link href=\"{escape(source['url'])}\" color=\"#205c46\">"
                f"{escape(source['title'])}</link>",
                styles["ReportBody"],
            ))
    if report.get("reverse_matches"):
        story.append(Paragraph("Reverse image matches", styles["ReportHeading"]))
        for match in report["reverse_matches"]:
            story.append(Paragraph(
                f"<link href=\"{escape(match['url'])}\" color=\"#205c46\">"
                f"{escape(match['title'])}</link>",
                styles["ReportBody"],
            ))
    if report["warnings"]:
        story.append(Paragraph("Limitations", styles["ReportHeading"]))
        for warning in report["warnings"]:
            story.append(Paragraph(escape(warning), styles["ReportBody"]))
    document.build(story)
    buffer.seek(0)
    return buffer


def _extract_claims(material, language):
    api_key = os.getenv("GEMINI_API_KEY", "")
    if api_key and api_key != "your_gemini_api_key_here":
        try:
            from google import genai

            response = genai.Client(api_key=api_key).models.generate_content(
                model="gemini-2.5-flash",
                contents=(
                    "Extract up to five independently checkable factual claims from this material. "
                    "Return only a JSON array of strings, no markdown. Keep claim wording in the source language.\n"
                    f"Material:\n{material[:12000]}"
                ),
            )
            claims = _parse_json(response.text)
            if isinstance(claims, list):
                filtered = [str(claim).strip() for claim in claims if str(claim).strip()]
                if filtered:
                    return filtered[:5]
        except Exception:
            pass
    sentence = re.split(r"(?<=[.!?।])\s+", material.strip())
    return [item.strip() for item in sentence[:5] if item.strip()]


def _review_with_ai(claim_results, language):
    api_key = os.getenv("GEMINI_API_KEY", "")
    if not api_key or api_key == "your_gemini_api_key_here":
        return None
    try:
        from crews import run_fact_check

        language_name = "Hindi" if language == "hi" else "English"
        case_data = json.dumps({"language": language_name, "claims": claim_results}, ensure_ascii=False)
        raw = run_fact_check(case_data, api_key)
        parsed = _parse_json(raw)
        return parsed if isinstance(parsed, dict) else None
    except Exception:
        return None


def _parse_json(value):
    if not value:
        return None
    cleaned = re.sub(r"^```(?:json)?\s*|\s*```$", "", value.strip(), flags=re.IGNORECASE)
    try:
        return json.loads(cleaned)
    except json.JSONDecodeError:
        match = re.search(r"\{.*\}|\[.*\]", cleaned, flags=re.DOTALL)
        if not match:
            return None
        try:
            return json.loads(match.group(0))
        except json.JSONDecodeError:
            return None


def _merge_review(claim_results, review):
    reviewed = review.get("claims", [])
    output = []
    for index, item in enumerate(claim_results):
        assessment = reviewed[index] if index < len(reviewed) and isinstance(reviewed[index], dict) else {}
        valid_urls = {source["url"] for source in item["evidence"]}
        cited = assessment.get("evidence")
        if not isinstance(cited, list):
            cited = item["evidence"]
        cited_urls = {source.get("url") for source in cited if isinstance(source, dict)}
        evidence = [source for source in item["evidence"] if source["url"] in (valid_urls & cited_urls)]
        if not evidence:
            evidence = item["evidence"]
        try:
            confidence = max(0, min(100, int(assessment.get("confidence", 0))))
        except (TypeError, ValueError):
            confidence = 0
        output.append({
            **item,
            "verdict": str(assessment.get("verdict", "Insufficient evidence")),
            "confidence": confidence,
            "explanation": str(assessment.get("explanation", "")),
            "evidence": evidence,
        })
    return output


def _source_for_url(url):
    hostname = (urlparse(url).hostname or "").lower()
    path = urlparse(url).path.lower()
    for source, domain in TRUSTED_SOURCES.items():
        trusted_host, _, trusted_path = domain.partition("/")
        if hostname == trusted_host or hostname.endswith(f".{trusted_host}"):
            if not trusted_path or path.startswith(f"/{trusted_path}"):
                return source
    return "Web source"
  
