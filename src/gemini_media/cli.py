"""
gemini-media: Multimodal understanding via Gemini 3

Modes:
- CLI: gemini-media <source> -m <model> [-p "prompt"]
- MCP: gemini-media mcp

Supports:
- YouTube URLs (passed directly to Gemini)
- Local audio files (uploaded to GCS, then processed)
- Local video files (uploaded to GCS, then processed)

Requires:
- GEMINI_API_KEY or GOOGLE_API_KEY environment variable
- GEMINI_MEDIA_GCS_BUCKET environment variable (required for local file analysis)
- GCP Application Default Credentials for GCS access
"""

import hashlib
import json
import os
import sys
from pathlib import Path

import click
from fastmcp import FastMCP

CACHE_DIR = Path.home() / ".cache" / "gemini-media"
DEFAULT_GCS_BUCKET = None  # Must set GEMINI_MEDIA_GCS_BUCKET env var
GEMINI_3_MODELS = [
    "gemini-3-pro-preview",
    "gemini-3-flash-preview",
]
IMAGE_GENERATION_MODEL = "gemini-3-pro-image-preview"
IMAGE_SIZES = ["1K", "2K", "4K"]
ASPECT_RATIOS = ["1:1", "2:3", "3:2", "3:4", "4:3", "4:5", "5:4", "9:16", "16:9", "21:9"]

MIME_TYPES = {
    # Audio
    ".mp3": "audio/mpeg",
    ".wav": "audio/wav",
    ".m4a": "audio/mp4",
    ".aac": "audio/aac",
    ".ogg": "audio/ogg",
    ".flac": "audio/flac",
    # Video
    ".mp4": "video/mp4",
    ".mov": "video/quicktime",
    ".avi": "video/x-msvideo",
    ".mkv": "video/x-matroska",
    ".webm": "video/webm",
    # Images
    ".jpg": "image/jpeg",
    ".jpeg": "image/jpeg",
    ".png": "image/png",
    ".gif": "image/gif",
    ".webp": "image/webp",
    # Documents
    ".pdf": "application/pdf",
}

DEFAULT_PROMPT = """Analyze this media comprehensively. Provide:
1. Summary (2-3 sentences)
2. Key points (bullet list)
3. Full transcript if audio/speech is present
4. Visual descriptions for key moments (with timestamps if video)
"""

mcp = FastMCP("gemini-media")


def get_cache_key(source: str, prompt: str, model: str) -> str:
    """Generate cache key from source, prompt, and model."""
    content = f"{source}|{prompt}|{model}"
    return hashlib.sha256(content.encode()).hexdigest()[:16]


def get_cached(cache_key: str) -> dict | None:
    """Retrieve cached result if exists."""
    cache_file = CACHE_DIR / f"{cache_key}.json"
    if cache_file.exists():
        return json.loads(cache_file.read_text())
    return None


def save_cache(cache_key: str, data: dict) -> None:
    """Save result to cache."""
    CACHE_DIR.mkdir(parents=True, exist_ok=True)
    cache_file = CACHE_DIR / f"{cache_key}.json"
    cache_file.write_text(json.dumps(data, indent=2))


def get_mime_type(file_path: Path) -> str:
    """Get MIME type from file extension."""
    ext = file_path.suffix.lower()
    return MIME_TYPES.get(ext, "application/octet-stream")


def get_source_type(file_path: Path) -> str:
    """Determine source type from file extension."""
    ext = file_path.suffix.lower()
    if ext in [".mp3", ".wav", ".m4a", ".aac", ".ogg", ".flac"]:
        return "audio"
    elif ext in [".mp4", ".mov", ".avi", ".mkv", ".webm"]:
        return "video"
    elif ext in [".jpg", ".jpeg", ".png", ".gif", ".webp"]:
        return "image"
    elif ext == ".pdf":
        return "document"
    return "file"


def is_youtube_url(source: str) -> bool:
    """Check if source is a YouTube URL."""
    youtube_patterns = [
        "https://youtube.com/",
        "https://www.youtube.com/",
        "https://youtu.be/",
        "http://youtube.com/",
        "http://www.youtube.com/",
    ]
    return any(source.startswith(p) for p in youtube_patterns)


def upload_to_gcs(file_path: Path, bucket_name: str, verbose: bool = True) -> str:
    """Upload file to GCS, return gs:// URI. Skips if already exists."""
    from google.cloud import storage

    storage_client = storage.Client()
    bucket = storage_client.bucket(bucket_name)

    file_hash = hashlib.sha256(file_path.read_bytes()).hexdigest()[:16]
    blob_name = f"gemini-media/{file_hash}{file_path.suffix.lower()}"

    blob = bucket.blob(blob_name)
    if not blob.exists():
        if verbose:
            click.echo(f"Uploading to gs://{bucket_name}/{blob_name}...", err=True)
        blob.upload_from_filename(str(file_path), content_type=get_mime_type(file_path))
        if verbose:
            click.echo("Upload complete.", err=True)
    elif verbose:
        click.echo(f"Using cached GCS file: gs://{bucket_name}/{blob_name}", err=True)

    return f"gs://{bucket_name}/{blob_name}"


def generate_temp_path(index: int) -> Path:
    """Generate temp file path with timestamp and index."""
    from datetime import datetime
    timestamp = datetime.now().strftime("%H%M%S")
    return Path(f"/tmp/gemini-media-{timestamp}-{index}.png")


def save_generated_image(data: bytes, output_path: Path | None, index: int, count: int) -> Path:
    """Save image bytes to file, return path."""
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
    """Core image generation logic shared by MCP and CLI."""
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
    """Process YouTube URL directly via Gemini."""
    from google.genai import types

    video_part = types.Part(file_data=types.FileData(file_uri=url, mime_type="video/mp4"))
    text_part = types.Part(text=prompt)
    response = client.models.generate_content(
        model=model,
        contents=[video_part, text_part],
    )
    return response.text


def process_local_file(client, file_path: Path, prompt: str, model: str, bucket_name: str, verbose: bool = True) -> str:
    """Upload to GCS and process via Gemini using gs:// URI."""
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
    return response.text


@mcp.tool
def analyze_media(
    source: str,
    prompt: str = DEFAULT_PROMPT,
    model: str = "gemini-3-flash-preview",
    no_cache: bool = False,
) -> dict:
    """
    Analyze media using Gemini 3 multimodal capabilities.

    Args:
        source: YouTube URL or absolute path to local audio/video/image/PDF file
        prompt: Analysis prompt (default: comprehensive analysis)
        model: gemini-3-pro-preview or gemini-3-flash-preview
        no_cache: Bypass cache and force fresh API call
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

    cache_key = get_cache_key(source, prompt, model)
    if not no_cache:
        cached = get_cached(cache_key)
        if cached:
            cached["cached"] = True
            return cached

    client = genai.Client(vertexai=True, api_key=api_key)

    try:
        if is_youtube:
            response_text = process_youtube(client, source, prompt, model)
            source_type = "youtube"
        else:
            response_text = process_local_file(client, file_path, prompt, model, bucket_name, verbose=False)
            source_type = get_source_type(file_path)
    except Exception as e:
        return {"error": str(e)}

    result = {
        "source": source,
        "source_type": source_type,
        "model": model,
        "prompt": prompt,
        "response": response_text,
        "cached": False,
    }
    save_cache(cache_key, result)
    return result


@mcp.tool
def clear_cache() -> dict:
    """Clear all cached gemini-media results."""
    if not CACHE_DIR.exists():
        return {"cleared": 0}
    count = sum(1 for f in CACHE_DIR.glob("*.json") if f.unlink() or True)
    return {"cleared": count}


@mcp.tool
def list_models() -> list[str]:
    """List available Gemini 3 models."""
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
    Generate or edit images using Gemini 3 Pro (nano-banana-pro).

    Args:
        prompt: Text description or edit instruction
        source: Optional absolute path to source image for editing
        size: Output resolution (1K, 2K, 4K)
        aspect_ratio: Aspect ratio (1:1, 16:9, etc.)
        count: Number of image variations to generate (1-4)

    Returns:
        Dict with images (list of paths), text (optional model commentary),
        model, prompt, source, size, and aspect_ratio.
    """
    return do_generate_image(
        prompt=prompt,
        source=source,
        size=size,
        aspect_ratio=aspect_ratio,
        count=count,
    )


@click.group(invoke_without_command=True)
@click.pass_context
def cli(ctx):
    """
    Gemini Media: Multimodal understanding and generation via Gemini 3.

    \b
    Commands:
        analyze   - Analyze media (YouTube URLs, audio, video, images, PDFs)
        generate  - Generate or edit images
        mcp       - Run as MCP server

    \b
    Legacy mode (backward compatible):
        gemini-media <source> -m <model> [-p "prompt"]
    """
    pass


@cli.command()
@click.argument("source")
@click.option(
    "--model", "-m",
    required=True,
    type=click.Choice(GEMINI_3_MODELS, case_sensitive=False),
    help="Gemini 3 model to use",
)
@click.option(
    "--prompt", "-p",
    default=DEFAULT_PROMPT,
    help="Prompt for analysis (default: comprehensive analysis)",
)
@click.option(
    "--no-cache",
    is_flag=True,
    help="Bypass cache and force fresh API call",
)
@click.option(
    "--raw",
    is_flag=True,
    help="Output raw text only (no JSON wrapper)",
)
def analyze(source: str, model: str, prompt: str, no_cache: bool, raw: bool):
    """
    Analyze media using Gemini 3 multimodal capabilities.

    \b
    Usage: gemini-media <source> -m <model> [-p "prompt"]
    Models: gemini-3-pro-preview, gemini-3-flash-preview
    Source: YouTube URL or absolute path to local audio/video file.
    Caches to ~/.cache/gemini-media/

    \b
    Note: Text output is inherently lossy - tone, pacing, visuals, and emphasis
    do not survive transcription. Factor this into prompt design.

    \b
    Frame extraction for visual context: Claude can read images natively via
    the Read tool. When analyzing video, ask Gemini to include specific
    timestamps for key moments. Then extract frames at those timestamps:
        ffmpeg -ss MM:SS -i video.mp4 -frames:v 1 frame.png
    This gives you actual visual context rather than relying solely on
    Gemini's text description - useful for UI analysis, diagrams, code on
    screen, or verifying details.

    \b
    Examples:
        gemini-media "https://youtube.com/watch?v=..." -m gemini-3-flash-preview
        gemini-media /path/to/recording.mp4 -m gemini-3-pro-preview -p "Summarize this"
        gemini-media /path/to/meeting.mp3 -m gemini-3-flash-preview -p "List action items"
    """
    # Validate API key
    api_key = os.environ.get("GEMINI_API_KEY") or os.environ.get("GOOGLE_API_KEY")
    if not api_key:
        click.echo("Error: GEMINI_API_KEY or GOOGLE_API_KEY environment variable not set", err=True)
        sys.exit(1)

    # Get GCS bucket (for local files)
    bucket_name = os.environ.get("GEMINI_MEDIA_GCS_BUCKET", DEFAULT_GCS_BUCKET)

    # Determine source type and validate
    is_youtube = is_youtube_url(source)

    if not is_youtube and not bucket_name:
        click.echo("Error: GEMINI_MEDIA_GCS_BUCKET environment variable required for local file analysis", err=True)
        sys.exit(1)

    if not is_youtube:
        file_path = Path(source)
        # Require absolute path
        if not file_path.is_absolute():
            click.echo(f"Error: File path must be absolute. Got: {source}", err=True)
            click.echo(f"Try: {file_path.absolute()}", err=True)
            sys.exit(1)
        if not file_path.exists():
            click.echo(f"Error: File not found: {source}", err=True)
            sys.exit(1)

    # Check cache
    cache_key = get_cache_key(source, prompt, model)
    if not no_cache:
        cached = get_cached(cache_key)
        if cached:
            if raw:
                click.echo(cached["response"])
            else:
                cached["cached"] = True
                click.echo(json.dumps(cached, indent=2))
            return

    # Initialize client - Vertex AI Express Mode with API key
    from google import genai
    client = genai.Client(vertexai=True, api_key=api_key)

    # Process based on source type
    try:
        if is_youtube:
            response_text = process_youtube(client, source, prompt, model)
            source_type = "youtube"
        else:
            response_text = process_local_file(client, file_path, prompt, model, bucket_name)
            source_type = get_source_type(file_path)

    except Exception as e:
        click.echo(f"Error: {e}", err=True)
        sys.exit(1)

    # Build result
    result = {
        "source": source,
        "source_type": source_type,
        "model": model,
        "prompt": prompt,
        "response": response_text,
        "cached": False,
    }

    # Save to cache
    save_cache(cache_key, result)

    # Output
    if raw:
        click.echo(response_text)
    else:
        click.echo(json.dumps(result, indent=2))


@cli.command(name="mcp")
def mcp_command():
    """Run as MCP server (for Claude Code integration)."""
    import logging
    logging.getLogger("fastmcp").setLevel(logging.WARNING)
    mcp.run(show_banner=False)


@cli.command()
@click.argument("source_or_prompt")
@click.argument("prompt_if_editing", required=False)
@click.option(
    "--size", "-s",
    default="2K",
    type=click.Choice(IMAGE_SIZES, case_sensitive=False),
    help="Output resolution (default: 2K)",
)
@click.option(
    "--aspect", "-a",
    default="1:1",
    type=click.Choice(ASPECT_RATIOS),
    help="Aspect ratio (default: 1:1)",
)
@click.option(
    "--count", "-n",
    default=1,
    type=click.IntRange(1, 4),
    help="Number of variations (1-4, default: 1)",
)
@click.option(
    "--output", "-o",
    type=click.Path(),
    help="Output file path (default: temp directory)",
)
@click.option(
    "--raw", "-r",
    is_flag=True,
    help="Output path(s) only, no JSON",
)
def generate(source_or_prompt: str, prompt_if_editing: str | None, size: str, aspect: str, count: int, output: str | None, raw: bool):
    """
    Generate or edit images using Gemini 3 Pro.

    \b
    Pure generation:
        gemini-media generate "A futuristic cityscape at sunset"

    \b
    Image editing (source path first, then prompt):
        gemini-media generate /path/to/photo.jpg "Make the sky more dramatic"

    \b
    With options:
        gemini-media generate "A cat" -s 4K -a 16:9 -n 3 -o output.png
    """
    if prompt_if_editing:
        source = source_or_prompt
        prompt = prompt_if_editing
    else:
        source = None
        prompt = source_or_prompt

    output_path = Path(output) if output else None

    result = do_generate_image(
        prompt=prompt,
        source=source,
        size=size,
        aspect_ratio=aspect,
        count=count,
        output_path=output_path,
    )

    if "error" in result:
        click.echo(f"Error: {result['error']}", err=True)
        sys.exit(1)

    if result.get("text"):
        click.echo(result["text"], err=True)

    if raw:
        for path in result["images"]:
            click.echo(path)
    else:
        click.echo(json.dumps(result, indent=2))


def is_subcommand(arg: str) -> bool:
    """Check if argument is a known subcommand."""
    return arg in ("analyze", "generate", "mcp", "--help", "-h")


def main():
    """Entry point for package installation."""
    if len(sys.argv) > 1 and not is_subcommand(sys.argv[1]):
        sys.argv.insert(1, "analyze")
    cli()


if __name__ == "__main__":
    main()
