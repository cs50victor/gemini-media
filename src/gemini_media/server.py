"""
gemini-media: local stdio MCP server for Gemini native multimodal models.

The MCP process runs on the same machine as the client. Local file inputs are
read from that machine, uploaded to GCS, then passed to Gemini as gs:// URIs.
"""

import hashlib
import os
import sys
from pathlib import Path

from fastmcp import FastMCP

DEFAULT_GCS_BUCKET = None
GEMINI_3_MODELS = [
    "gemini-3.5-flash",
    "gemini-3.1-pro-preview",
]
IMAGE_GENERATION_MODEL = "gemini-3.1-pro-image-preview"
IMAGE_SIZES = ["1K", "2K", "4K"]
ASPECT_RATIOS = ["1:1", "2:3", "3:2", "3:4", "4:3", "4:5", "5:4", "9:16", "16:9", "21:9"]

MIME_TYPES = {
    ".mp3": "audio/mpeg",
    ".wav": "audio/wav",
    ".m4a": "audio/mp4",
    ".aac": "audio/aac",
    ".ogg": "audio/ogg",
    ".flac": "audio/flac",
    ".mp4": "video/mp4",
    ".mov": "video/quicktime",
    ".avi": "video/x-msvideo",
    ".mkv": "video/x-matroska",
    ".webm": "video/webm",
    ".jpg": "image/jpeg",
    ".jpeg": "image/jpeg",
    ".png": "image/png",
    ".gif": "image/gif",
    ".webp": "image/webp",
    ".pdf": "application/pdf",
}

DEFAULT_PROMPT = """Analyze this media comprehensively. Provide:
1. Summary (2-3 sentences)
2. Key points (bullet list)
3. Full transcript if audio/speech is present
4. Visual descriptions for key moments (with timestamps if video)
"""

mcp = FastMCP("gemini-media")


def get_mime_type(file_path: Path) -> str:
    return MIME_TYPES.get(file_path.suffix.lower(), "application/octet-stream")


def get_source_type(file_path: Path) -> str:
    ext = file_path.suffix.lower()
    if ext in [".mp3", ".wav", ".m4a", ".aac", ".ogg", ".flac"]:
        return "audio"
    if ext in [".mp4", ".mov", ".avi", ".mkv", ".webm"]:
        return "video"
    if ext in [".jpg", ".jpeg", ".png", ".gif", ".webp"]:
        return "image"
    if ext == ".pdf":
        return "document"
    return "file"


def is_youtube_url(source: str) -> bool:
    youtube_patterns = [
        "https://youtube.com/",
        "https://www.youtube.com/",
        "https://youtu.be/",
        "http://youtube.com/",
        "http://www.youtube.com/",
    ]
    return any(source.startswith(pattern) for pattern in youtube_patterns)


def upload_to_gcs(file_path: Path, bucket_name: str, verbose: bool = True) -> str:
    from google.cloud import storage

    storage_client = storage.Client()
    bucket = storage_client.bucket(bucket_name)

    with open(file_path, "rb") as f:
        file_hash = hashlib.file_digest(f, "sha256").hexdigest()[:16]
    blob_name = f"gemini-media/{file_hash}{file_path.suffix.lower()}"

    if verbose:
        print(f"Uploading to gs://{bucket_name}/{blob_name}...", file=sys.stderr)
    blob = bucket.blob(blob_name)
    blob.upload_from_filename(str(file_path), content_type=get_mime_type(file_path))
    if verbose:
        print("Upload complete.", file=sys.stderr)

    return f"gs://{bucket_name}/{blob_name}"


def generate_temp_path(index: int) -> Path:
    from datetime import datetime

    timestamp = datetime.now().strftime("%H%M%S")
    return Path(f"/tmp/gemini-media-{timestamp}-{index}.png")


def save_generated_image(data: bytes, output_path: Path | None, index: int, count: int) -> Path:
    if output_path:
        if count > 1:
            stem = output_path.stem
            suffix = output_path.suffix or ".png"
            path = output_path.parent / f"{stem}-{index}{suffix}"
        else:
            path = output_path
    else:
        path = generate_temp_path(index)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(data)
    return path


def do_generate_image(
    prompt: str,
    source: str | None = None,
    size: str = "2K",
    aspect_ratio: str = "1:1",
    count: int = 1,
    output_path: Path | None = None,
) -> dict:
    from google import genai
    from google.genai import types

    if size not in IMAGE_SIZES:
        return {"error": f"Invalid size. Choose from: {IMAGE_SIZES}"}
    if aspect_ratio not in ASPECT_RATIOS:
        return {"error": f"Invalid aspect ratio. Choose from: {ASPECT_RATIOS}"}
    if not 1 <= count <= 4:
        return {"error": "Count must be between 1 and 4"}

    api_key = os.environ.get("GEMINI_API_KEY") or os.environ.get("GOOGLE_API_KEY")
    if not api_key:
        return {"error": "GEMINI_API_KEY or GOOGLE_API_KEY not set"}

    client = genai.Client(vertexai=True, api_key=api_key)
    contents = []

    if source:
        source_path = Path(source)
        if not source_path.is_absolute():
            return {"error": f"Source path must be absolute. Got: {source}"}
        if not source_path.exists():
            return {"error": f"Source file not found: {source}"}
        mime_type = get_mime_type(source_path)
        image_data = source_path.read_bytes()
        contents.append(types.Part.from_bytes(data=image_data, mime_type=mime_type))

    contents.append(types.Part(text=prompt))

    config = types.GenerateContentConfig(
        response_modalities=["TEXT", "IMAGE"],
        image_config=types.ImageConfig(
            image_size=size,
            aspect_ratio=aspect_ratio,
        ),
    )

    images = []
    text_parts = []

    try:
        for i in range(count):
            response = client.models.generate_content(
                model=IMAGE_GENERATION_MODEL,
                contents=contents,
                config=config,
            )
            for part in response.candidates[0].content.parts:
                if part.inline_data and part.inline_data.mime_type.startswith("image/"):
                    saved_path = save_generated_image(
                        part.inline_data.data, output_path, i + 1, count
                    )
                    images.append(str(saved_path))
                elif part.text:
                    text_parts.append(part.text)
    except Exception as e:
        return {"error": str(e)}

    return {
        "images": images,
        "text": "\n".join(text_parts) if text_parts else None,
        "model": IMAGE_GENERATION_MODEL,
        "prompt": prompt,
        "source": source,
        "size": size,
        "aspect_ratio": aspect_ratio,
    }


def process_youtube(client, url: str, prompt: str, model: str) -> str:
    from google.genai import types

    video_part = types.Part(file_data=types.FileData(file_uri=url, mime_type="video/mp4"))
    text_part = types.Part(text=prompt)
    response = client.models.generate_content(
        model=model,
        contents=[video_part, text_part],
    )
    return response.text


def process_local_file(
    client,
    file_path: Path,
    prompt: str,
    model: str,
    bucket_name: str,
    verbose: bool = True,
) -> tuple[str, str]:
    from google.genai import types

    gcs_uri = upload_to_gcs(file_path, bucket_name, verbose=verbose)
    mime_type = get_mime_type(file_path)

    response = client.models.generate_content(
        model=model,
        contents=[
            types.Part.from_uri(file_uri=gcs_uri, mime_type=mime_type),
            prompt,
        ],
    )
    return response.text, gcs_uri


@mcp.tool
def analyze_media(
    source: str,
    prompt: str = DEFAULT_PROMPT,
    model: str = "gemini-3.5-flash",
) -> dict:
    """
    Analyze media using Gemini native multimodal capabilities.

    Args:
        source: YouTube URL or absolute path to a file on the local MCP host
        prompt: Analysis prompt
        model: gemini-3.5-flash or gemini-3.1-pro-preview
    """
    from google import genai

    if model not in GEMINI_3_MODELS:
        return {"error": f"Invalid model. Choose from: {GEMINI_3_MODELS}"}

    api_key = os.environ.get("GEMINI_API_KEY") or os.environ.get("GOOGLE_API_KEY")
    if not api_key:
        return {"error": "GEMINI_API_KEY or GOOGLE_API_KEY not set"}

    bucket_name = os.environ.get("GEMINI_MEDIA_GCS_BUCKET", DEFAULT_GCS_BUCKET)
    is_youtube = is_youtube_url(source)

    if not is_youtube and not bucket_name:
        return {"error": "GEMINI_MEDIA_GCS_BUCKET environment variable required for local file analysis"}

    if not is_youtube:
        file_path = Path(source)
        if not file_path.is_absolute():
            return {"error": f"File path must be absolute. Got: {source}"}
        if not file_path.exists():
            return {"error": f"File not found: {source}"}

    client = genai.Client(vertexai=True, api_key=api_key)

    gcs_uri = None
    try:
        if is_youtube:
            response_text = process_youtube(client, source, prompt, model)
            source_type = "youtube"
        else:
            response_text, gcs_uri = process_local_file(
                client, file_path, prompt, model, bucket_name, verbose=False
            )
            source_type = get_source_type(file_path)
    except Exception as e:
        return {"error": str(e)}

    result = {
        "source": source,
        "source_type": source_type,
        "model": model,
        "prompt": prompt,
        "response": response_text,
    }
    if not is_youtube:
        result["gcs_uri"] = gcs_uri
    return result


@mcp.tool
def list_models() -> list[str]:
    """List available Gemini media analysis models."""
    return GEMINI_3_MODELS


@mcp.tool
def generate_image(
    prompt: str,
    source: str | None = None,
    size: str = "2K",
    aspect_ratio: str = "1:1",
    count: int = 1,
) -> dict:
    """
    Generate or edit images using Gemini native multimodal capabilities.

    Args:
        prompt: Text description or edit instruction
        source: Optional absolute path to source image on the local MCP host
        size: Output resolution
        aspect_ratio: Aspect ratio
        count: Number of image variations
    """
    return do_generate_image(
        prompt=prompt,
        source=source,
        size=size,
        aspect_ratio=aspect_ratio,
        count=count,
    )


def main() -> None:
    mcp.run(transport="stdio", show_banner=False)


if __name__ == "__main__":
    main()
