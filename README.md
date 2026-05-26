# gemini-media

Local stdio MCP server for Gemini media analysis and image generation.

## Installation

```bash
# Via uvx (ephemeral)
uvx --from git+https://github.com/cs50victor/gemini-media gemini-media

# Via uv (permanent)
uv tool install git+https://github.com/cs50victor/gemini-media
```

## Usage

Run as a local stdio MCP server for MCP clients:

```bash
gemini-media
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
      "args": ["--from", "git+https://github.com/cs50victor/gemini-media", "gemini-media"],
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

## MCP Tools

- `analyze_media` - analyze a YouTube URL or absolute local media path
- `generate_image` - generate or edit images
- `list_models` - list supported Gemini models

## Supported Media

- **YouTube URLs** - passed directly to Gemini
- **Audio** - mp3, wav, m4a, aac, ogg, flac
- **Video** - mp4, mov, avi, mkv, webm
- **Images** - jpg, jpeg, png, gif, webp
- **Documents** - pdf

## License

MIT
