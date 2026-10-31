#!/usr/bin/env python3
"""
Ollama-to-Strata API Shim
Translates Ollama REST endpoints (/api/tags, /api/version, /api/show, /api/chat, /api/generate)
into Strata's OpenAI-compatible API (/v1/chat/completions, /v1/models).
Supports CORS Private Network Access (PNA) for Android WebView / PWA / browser clients.
"""

import argparse
import asyncio
import datetime
import json
import logging
import os
import sys
import time
from typing import Any, AsyncGenerator, Dict, List, Optional

import httpx
import uvicorn
from fastapi import FastAPI, Request, Response
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse, PlainTextResponse, StreamingResponse

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
logger = logging.getLogger("ollama-shim")

STRATA_BASE_URL = os.environ.get("STRATA_BASE_URL", "http://127.0.0.1:8081").rstrip("/")
DEFAULT_MODEL = "qwen3.8-flash-next-unsloth-ud-iq4_xs"

app = FastAPI(title="Ollama-to-Strata Shim")

# Enable full CORS including Private Network Access (PNA)
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
    allow_private_network=True,
)


@app.middleware("http")
async def add_pna_and_cors_headers(request: Request, call_next):
    # Handle user putting path like /v1/chat/completions as base URL:
    # If path starts with /v1/chat/completions/api/... rewrite to /api/...
    path = request.url.path
    if path.startswith("/v1/chat/completions/api/") or path.startswith("/v1/chat/completions/v1/") or path.startswith("/v1/chat/completions/models"):
        sub = path[len("/v1/chat/completions"):]
        logger.info(f"Rewriting path from {path} to {sub}")
        request.scope["path"] = sub
    elif path == "/v1/chat/completions/":
        request.scope["path"] = "/v1/chat/completions"

    if request.method == "OPTIONS":
        headers = {
            "Access-Control-Allow-Origin": request.headers.get("origin", "*"),
            "Access-Control-Allow-Methods": "GET, POST, PUT, DELETE, OPTIONS, HEAD",
            "Access-Control-Allow-Headers": request.headers.get("access-control-request-headers", "*"),
            "Access-Control-Allow-Credentials": "true",
            "Access-Control-Allow-Private-Network": "true",
            "Access-Control-Max-Age": "86400",
        }
        return Response(status_code=204, headers=headers)

    response = await call_next(request)
    response.headers["Access-Control-Allow-Origin"] = request.headers.get("origin", "*")
    response.headers["Access-Control-Allow-Credentials"] = "true"
    response.headers["Access-Control-Allow-Private-Network"] = "true"
    return response


http_client = httpx.AsyncClient(timeout=600.0)


@app.on_event("shutdown")
async def shutdown():
    await http_client.aclose()


def now_iso() -> str:
    return datetime.datetime.now(datetime.timezone.utc).strftime("%Y-%m-%dT%H:%M:%S.%fZ")


async def get_strata_models() -> List[str]:
    try:
        r = await http_client.get(f"{STRATA_BASE_URL}/v1/models")
        if r.status_code == 200:
            data = r.json()
            return [m["id"] for m in data.get("data", [])]
    except Exception as e:
        logger.warning(f"Failed to fetch models from Strata: {e}")
    return [DEFAULT_MODEL]


def build_model_entry(name: str) -> Dict[str, Any]:
    return {
        "name": name,
        "model": name,
        "modified_at": now_iso(),
        "size": 55430000000,
        "digest": "qwen38flashnext",
        "details": {
          "parent_model": "",
          "format": "gguf",
          "family": "qwen2",
          "families": ["qwen2"],
          "parameter_size": "125B",
          "quantization_level": "UD-IQ4_XS",
        },
    }


@app.get("/")
@app.head("/")
async def root():
    return PlainTextResponse("Ollama is running\n")


@app.post("/")
async def root_post(req: Request):
    return await passthrough_to_strata("POST", "v1/chat/completions", req)


@app.get("/api/version")
async def api_version():
    return {"version": "0.4.0"}


@app.get("/api/tags")
async def api_tags():
    active_models = await get_strata_models()
    model_names = set()
    for m in active_models:
        model_names.add(m)
        model_names.add(f"{m}:latest")
    # Common aliases
    model_names.add("qwen2.5-coder:32b")
    model_names.add("qwen2.5-coder:latest")
    model_names.add("qwen:latest")
    model_names.add("default:latest")

    return {"models": [build_model_entry(m) for m in sorted(model_names)]}


@app.get("/api/ps")
async def api_ps():
    active_models = await get_strata_models()
    res = []
    for m in active_models:
        entry = build_model_entry(f"{m}:latest")
        entry["expires_at"] = "2099-01-01T00:00:00Z"
        entry["size_vram"] = 55430000000
        res.append(entry)
    return {"models": res}


@app.post("/api/show")
async def api_show(req: Request):
    body = await req.json() if (await req.body()) else {}
    name = body.get("name") or body.get("model") or DEFAULT_MODEL
    return {
        "license": "Apache-2.0",
        "modelfile": f"# Modelfile for {name}\nFROM {DEFAULT_MODEL}\n",
        "parameters": 'stop "<|im_start|>"\nstop "<|im_end|>"',
        "template": "{{ if .System }}<|im_start|>system\n{{ .System }}<|im_end|>\n{{ end }}{{ if .Prompt }}<|im_start|>user\n{{ .Prompt }}<|im_end|>\n{{ end }}<|im_start|>assistant\n{{ .Response }}<|im_end|>",
        "details": {
            "parent_model": "",
            "format": "gguf",
            "family": "qwen2",
            "families": ["qwen2"],
            "parameter_size": "125B",
            "quantization_level": "UD-IQ4_XS",
        },
    }


def translate_options_to_openai(options: Optional[Dict[str, Any]]) -> Dict[str, Any]:
    kw: Dict[str, Any] = {}
    if not options or not isinstance(options, dict):
        return kw
    if "temperature" in options:
        kw["temperature"] = options["temperature"]
    if "top_p" in options:
        kw["top_p"] = options["top_p"]
    if "num_predict" in options:
        kw["max_tokens"] = options["num_predict"]
    elif "max_tokens" in options:
        kw["max_tokens"] = options["max_tokens"]
    if "stop" in options:
        kw["stop"] = options["stop"]
    return kw


@app.post("/api/chat")
async def api_chat(req: Request):
    data = await req.json()
    model = data.get("model") or DEFAULT_MODEL
    messages = data.get("messages", [])
    stream = data.get("stream", True)
    options = data.get("options", {})

    openai_payload = {
        "model": DEFAULT_MODEL,
        "messages": messages,
        "stream": stream,
        **translate_options_to_openai(options),
    }

    t0 = time.time()
    logger.info(f"POST /api/chat model={model} stream={stream} msgs={len(messages)}")

    if not stream:
        resp = await http_client.post(
            f"{STRATA_BASE_URL}/v1/chat/completions",
            json=openai_payload,
        )
        if resp.status_code != 200:
            return Response(content=resp.content, status_code=resp.status_code, media_type="application/json")
        res_json = resp.json()
        choice = res_json.get("choices", [{}])[0]
        msg = choice.get("message", {})
        usage = res_json.get("usage", {})
        duration_ns = int((time.time() - t0) * 1e9)

        ollama_resp = {
            "model": model,
            "created_at": now_iso(),
            "message": {
                "role": "assistant",
                "content": msg.get("content") or "",
            },
            "done": True,
            "total_duration": duration_ns,
            "load_duration": 0,
            "prompt_eval_count": usage.get("prompt_tokens", 0),
            "eval_count": usage.get("completion_tokens", 0),
        }
        if msg.get("reasoning_content"):
            ollama_resp["message"]["thinking"] = msg["reasoning_content"]
        return JSONResponse(ollama_resp)

    # Streaming mode
    async def chat_stream_generator() -> AsyncGenerator[bytes, None]:
        total_prompt_tokens = 0
        total_completion_tokens = 0
        async with http_client.stream(
            "POST",
            f"{STRATA_BASE_URL}/v1/chat/completions",
            json=openai_payload,
        ) as r:
            async for line in r.aiter_lines():
                if not line or not line.startswith("data: "):
                    continue
                data_str = line[6:].strip()
                if data_str == "[DONE]":
                    break
                try:
                    chunk = json.loads(data_str)
                    choice = chunk.get("choices", [{}])[0]
                    delta = choice.get("delta", {})
                    usage = chunk.get("usage")
                    if usage:
                        total_prompt_tokens = usage.get("prompt_tokens", total_prompt_tokens)
                        total_completion_tokens = usage.get("completion_tokens", total_completion_tokens)

                    content_delta = delta.get("content")
                    reasoning_delta = delta.get("reasoning_content")

                    if content_delta or reasoning_delta:
                        item = {
                            "model": model,
                            "created_at": now_iso(),
                            "message": {
                                "role": "assistant",
                                "content": content_delta or "",
                            },
                            "done": False,
                        }
                        if reasoning_delta:
                            item["message"]["thinking"] = reasoning_delta
                        yield json.dumps(item).encode("utf-8") + b"\n"
                except Exception as e:
                    logger.debug(f"Chunk decode error: {e}")

        duration_ns = int((time.time() - t0) * 1e9)
        final_item = {
            "model": model,
            "created_at": now_iso(),
            "message": {
                "role": "assistant",
                "content": "",
            },
            "done": True,
            "total_duration": duration_ns,
            "load_duration": 0,
            "prompt_eval_count": total_prompt_tokens,
            "eval_count": total_completion_tokens,
        }
        yield json.dumps(final_item).encode("utf-8") + b"\n"

    return StreamingResponse(chat_stream_generator(), media_type="application/x-ndjson")


@app.post("/api/generate")
async def api_generate(req: Request):
    data = await req.json()
    model = data.get("model") or DEFAULT_MODEL
    prompt = data.get("prompt", "")
    system = data.get("system", "")
    stream = data.get("stream", True)
    options = data.get("options", {})

    messages = []
    if system:
        messages.append({"role": "system", "content": system})
    messages.append({"role": "user", "content": prompt})

    openai_payload = {
        "model": DEFAULT_MODEL,
        "messages": messages,
        "stream": stream,
        **translate_options_to_openai(options),
    }

    t0 = time.time()
    logger.info(f"POST /api/generate model={model} stream={stream} prompt_len={len(prompt)}")

    if not stream:
        resp = await http_client.post(
            f"{STRATA_BASE_URL}/v1/chat/completions",
            json=openai_payload,
        )
        if resp.status_code != 200:
            return Response(content=resp.content, status_code=resp.status_code, media_type="application/json")
        res_json = resp.json()
        choice = res_json.get("choices", [{}])[0]
        msg = choice.get("message", {})
        usage = res_json.get("usage", {})
        duration_ns = int((time.time() - t0) * 1e9)

        ollama_resp = {
            "model": model,
            "created_at": now_iso(),
            "response": msg.get("content") or "",
            "done": True,
            "total_duration": duration_ns,
            "load_duration": 0,
            "prompt_eval_count": usage.get("prompt_tokens", 0),
            "eval_count": usage.get("completion_tokens", 0),
        }
        if msg.get("reasoning_content"):
            ollama_resp["thinking"] = msg["reasoning_content"]
        return JSONResponse(ollama_resp)

    # Streaming mode
    async def gen_stream_generator() -> AsyncGenerator[bytes, None]:
        total_prompt_tokens = 0
        total_completion_tokens = 0
        async with http_client.stream(
            "POST",
            f"{STRATA_BASE_URL}/v1/chat/completions",
            json=openai_payload,
        ) as r:
            async for line in r.aiter_lines():
                if not line or not line.startswith("data: "):
                    continue
                data_str = line[6:].strip()
                if data_str == "[DONE]":
                    break
                try:
                    chunk = json.loads(data_str)
                    choice = chunk.get("choices", [{}])[0]
                    delta = choice.get("delta", {})
                    usage = chunk.get("usage")
                    if usage:
                        total_prompt_tokens = usage.get("prompt_tokens", total_prompt_tokens)
                        total_completion_tokens = usage.get("completion_tokens", total_completion_tokens)

                    content_delta = delta.get("content")
                    reasoning_delta = delta.get("reasoning_content")

                    if content_delta or reasoning_delta:
                        item = {
                            "model": model,
                            "created_at": now_iso(),
                            "response": content_delta or "",
                            "done": False,
                        }
                        if reasoning_delta:
                            item["thinking"] = reasoning_delta
                        yield json.dumps(item).encode("utf-8") + b"\n"
                except Exception as e:
                    logger.debug(f"Chunk decode error: {e}")

        duration_ns = int((time.time() - t0) * 1e9)
        final_item = {
            "model": model,
            "created_at": now_iso(),
            "response": "",
            "done": True,
            "total_duration": duration_ns,
            "load_duration": 0,
            "prompt_eval_count": total_prompt_tokens,
            "eval_count": total_completion_tokens,
        }
        yield json.dumps(final_item).encode("utf-8") + b"\n"

    return StreamingResponse(gen_stream_generator(), media_type="application/x-ndjson")


# Common endpoint aliases for clients querying root-level routes
@app.get("/models")
async def get_models_alias(req: Request):
    return await passthrough_to_strata("GET", "models", req)


@app.api_route("/chat/completions", methods=["POST", "OPTIONS"])
async def chat_completions_alias(req: Request):
    return await passthrough_to_strata(req.method, "v1/chat/completions", req)


# Pass-through for /v1/*
@app.api_route("/v1/{path:path}", methods=["GET", "POST", "PUT", "DELETE", "OPTIONS", "HEAD"])
async def passthrough_v1(req: Request, path: str):
    return await passthrough_to_strata(req.method, f"v1/{path}", req)


async def passthrough_to_strata(method: str, path: str, req: Optional[Request] = None):
    url = f"{STRATA_BASE_URL}/{path}"
    headers = dict(req.headers) if req else {}
    headers.pop("host", None)
    content = (await req.body()) if req else b""
    params = req.query_params if req else {}

    # Strata streaming support for direct /v1/chat/completions requests
    if method == "POST" and "stream" in content.decode("utf-8", "ignore"):
        try:
            body_json = json.loads(content)
            if body_json.get("stream"):
                async def sse_generator():
                    async with http_client.stream(method, url, headers=headers, json=body_json, params=params) as r:
                        async for chunk in r.aiter_raw():
                            yield chunk
                return StreamingResponse(sse_generator(), media_type="text/event-stream")
        except Exception:
            pass

    resp = await http_client.request(
        method,
        url,
        headers=headers,
        content=content,
        params=params,
    )
    return Response(content=resp.content, status_code=resp.status_code, media_type=resp.headers.get("content-type"))


def main():
    parser = argparse.ArgumentParser(description="Ollama-to-Strata API Shim")
    parser.add_argument("--host", default=os.environ.get("SHIM_HOST", "0.0.0.0"), help="Host to bind (default: 0.0.0.0)")
    parser.add_argument("--port", type=int, default=int(os.environ.get("SHIM_PORT", 11434)), help="Port to bind (default: 11434)")
    args = parser.parse_args()

    logger.info(f"Starting Ollama Shim on {args.host}:{args.port} -> Strata at {STRATA_BASE_URL}")
    uvicorn.run(app, host=args.host, port=args.port, log_level="info")


if __name__ == "__main__":
    main()
