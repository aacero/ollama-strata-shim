# ollama-strata-shim

A lightweight, high-performance API proxy that translates native **Ollama REST API** endpoints (`/api/tags`, `/api/version`, `/api/show`, `/api/chat`, `/api/generate`) into standard **OpenAI Chat Completions** (`/v1/chat/completions`, `/v1/models`).

Originally built to connect mobile apps, PWAs, and local tools expecting an Ollama daemon (such as [SciREPL](https://github.com/s243a/SciREPL), Open WebUI, and IDE extensions) to [Strata](https://github.com/Niko1221/Strata), `llama-server`, or `vLLM` without running multiple inference engines simultaneously.

---

## Key Features

- **Full Streaming & Non-Streaming Translation:** Seamlessly bridges Server-Sent Events (SSE) from OpenAI backends into Ollama NDJSON streams (`/api/chat` and `/api/generate`).
- **Chromium Private Network Access (PNA) Compliant:** Implements `Access-Control-Allow-Private-Network: true` and preflight handling. Prevents `TypeError: Failed to fetch` errors when secure WebViews, PWAs (`https://localhost`), or browser clients make requests to local network/loopback daemons.
- **Thinking / Reasoning Passthrough:** Extracts reasoning tokens (`reasoning_content`) from models like Qwen 2.5/3 or DeepSeek-R1 and maps them into Ollama's `message.thinking` delta streams.
- **Tool Calling Support:** Preserves `tools` and `tool_calls` payloads across multi-turn assistant workflows.
- **Flexible Path Routing:** Automatically normalizes paths whether the client posts to root (`/`), `/chat/completions`, or standard `/v1/*` routes.
- **Dynamic Model Discovery:** Queries the underlying backend's `/v1/models` and exposes active models plus standard aliases (`default:latest`, `qwen2.5-coder:32b`, etc.) in `/api/tags` and `/api/ps`.

---

## Architecture

```
[Mobile App / WebView / PWA / IDE]
               │ (Ollama REST: /api/chat, /api/tags, etc.)
               ▼
     [ollama-strata-shim]  (:11434)
               │ (OpenAI REST: /v1/chat/completions)
               ▼
 [Strata / llama-server / vLLM] (:8081)
```

---

## Quick Start

### Run with `uv` (Instant, No Installation)
```bash
uv run --with fastapi --with "uvicorn[standard]" --with httpx shim.py
```

### Installation with `pip`
```bash
pip install -e .
ollama-strata-shim --host 0.0.0.0 --port 11434
```

---

## Configuration

All configuration options support environment variables with sensible defaults:

| Option | Environment Variable | Default | Description |
| :--- | :--- | :--- | :--- |
| `--host` | `SHIM_HOST` | `0.0.0.0` | IP/Interface to bind |
| `--port` | `SHIM_PORT` | `11434` | Port to listen on (standard Ollama port) |
| Backend URL | `STRATA_BASE_URL` | `http://127.0.0.1:8081` | URL of the OpenAI-compatible backend |

---

## Running as a systemd User Service

To run automatically in the background across reboots:

1. Copy or symlink `systemd/ollama-strata-shim.service` to `~/.config/systemd/user/`:
   ```bash
   mkdir -p ~/.config/systemd/user
   cp systemd/ollama-strata-shim.service ~/.config/systemd/user/
   ```
2. Enable and start:
   ```bash
   systemctl --user daemon-reload
   systemctl --user enable --now ollama-strata-shim.service
   ```
3. Check status:
   ```bash
   systemctl --user status ollama-strata-shim.service
   ```

---

## License

MIT License. See [LICENSE](LICENSE) for details.
