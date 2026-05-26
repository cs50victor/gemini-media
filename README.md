# gemini-media

Multimodal media analysis and image generation via Google Gemini 3.1.

## Installation

```bash
# Via uvx (ephemeral)
uvx --from git+https://github.com/cs50victor/gemini-media gemini-media --help

# Via uv (permanent)
uv tool install git+https://github.com/cs50victor/gemini-media
```

## Usage

### CLI

```bash
# Analyze YouTube video
gemini-media analyze "https://youtube.com/watch?v=..." -m gemini-3.1-pro-preview

# Analyze local file (requires GCS bucket for upload)
export GEMINI_MEDIA_GCS_BUCKET="your-bucket"
gemini-media analyze /path/to/video.mp4 -m gemini-3.1-pro-preview

# Generate image
gemini-media generate "A futuristic cityscape at sunset" -s 4K -a 16:9
```

### MCP Server

Run as a local stdio MCP server for MCP clients:

```bash
gemini-media mcp
```

This server is intentionally local-only. It reads absolute file paths from the
same machine that starts the MCP process, uploads those files directly to GCS,
then sends the resulting `gs://` URI to Gemini. It does not expose HTTP, SSE, or
remote MCP transports.

Claude Code config (`~/.claude/settings.json`):

```json
{
  "mcpServers": {
    "gemini-media": {
      "command": "uvx",
      "args": ["--from", "git+https://github.com/cs50victor/gemini-media", "gemini-media", "mcp"],
      "env": {
        "GEMINI_API_KEY": "your-api-key",
        "GEMINI_MEDIA_GCS_BUCKET": "your-bucket"
      }
    }
  }
}
```

## Environment Variables

| Variable | Required | Description |
|----------|----------|-------------|
| `GEMINI_API_KEY` or `GOOGLE_API_KEY` | Yes | Google AI API key |
| `GEMINI_MEDIA_GCS_BUCKET` | For local files | GCS bucket for uploading local media |

## Supported Media

- **YouTube URLs** - passed directly to Gemini
- **Audio** - mp3, wav, m4a, aac, ogg, flac
- **Video** - mp4, mov, avi, mkv, webm
- **Images** - jpg, jpeg, png, gif, webp
- **Documents** - pdf

## License

MIT
