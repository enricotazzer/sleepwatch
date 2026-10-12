"""Chat backends: a local Ollama server, and a scripted stand-in for tests.

A hosted backend (for a public demo) plugs in through the same ``ChatBackend`` protocol in
Phase 5.
"""

from __future__ import annotations

import time
from collections.abc import Callable, Sequence
from dataclasses import dataclass, field
from typing import Protocol

import httpx
from pydantic_settings import BaseSettings, SettingsConfigDict

from sleepwatch.config import PROJECT_ROOT

Messages = list[dict[str, str]]  # {"role": "system" | "user" | "assistant", "content": ...}


class AgentSettings(BaseSettings):
    """``SLEEPWATCH_OLLAMA_URL`` from the environment or ``.env``. Kept apart from
    ``sleepwatch.config`` so the code earlier phases ran stays unchanged."""

    model_config = SettingsConfigDict(
        env_prefix="SLEEPWATCH_", env_file=PROJECT_ROOT / ".env", extra="ignore"
    )

    ollama_url: str = "http://localhost:11434"


@dataclass
class Reply:
    text: str
    seconds: float
    prompt_tokens: int | None = None
    output_tokens: int | None = None


class ChatBackend(Protocol):
    model: str

    def chat(
        self,
        messages: Messages,
        *,
        schema: dict | None = None,
        temperature: float,
        seed: int,
        max_tokens: int = 512,
    ) -> Reply: ...


class OllamaBackend:
    """``/api/chat`` on an Ollama server, with thinking off, at most ``max_tokens`` of output
    and, given a schema, output constrained to it."""

    def __init__(
        self,
        model: str,
        url: str | None = None,
        num_ctx: int = 8192,
        timeout: float = 600.0,
        keep_alive: str = "30m",
    ) -> None:
        self.model = model
        self.url = (url or AgentSettings().ollama_url).rstrip("/")
        self.num_ctx = num_ctx
        self.keep_alive = keep_alive
        self.client = httpx.Client(timeout=timeout)

    def chat(
        self,
        messages: Messages,
        *,
        schema: dict | None = None,
        temperature: float,
        seed: int,
        max_tokens: int = 512,
    ) -> Reply:
        body = {
            "model": self.model,
            "messages": messages,
            "stream": False,
            "think": False,
            "keep_alive": self.keep_alive,
            "options": {
                "temperature": temperature,
                "seed": seed,
                "num_ctx": self.num_ctx,
                "num_predict": max_tokens,
            },
        }
        if schema is not None:
            body["format"] = schema
        start = time.perf_counter()
        response = self.client.post(f"{self.url}/api/chat", json=body)
        response.raise_for_status()
        data = response.json()
        if data.get("prompt_eval_count", 0) + max_tokens > self.num_ctx:
            # Ollama silently drops the start of a prompt that does not fit
            raise RuntimeError(
                f"{self.model}: prompt of {data['prompt_eval_count']} tokens plus {max_tokens} "
                f"for the answer exceeds num_ctx={self.num_ctx}"
            )
        return Reply(
            text=data["message"]["content"],
            seconds=time.perf_counter() - start,
            prompt_tokens=data.get("prompt_eval_count"),
            output_tokens=data.get("eval_count"),
        )


def ollama_info(models: Sequence[str], url: str | None = None) -> dict:
    """Server version and each model's digest, recorded with every run."""
    url = (url or AgentSettings().ollama_url).rstrip("/")
    version = httpx.get(f"{url}/api/version", timeout=10).json()["version"]
    tags = httpx.get(f"{url}/api/tags", timeout=10).json()["models"]
    digests = {m["name"]: m["digest"] for m in tags}
    missing = [m for m in models if m not in digests]
    if missing:
        raise RuntimeError(f"not pulled on {url}: {', '.join(missing)}")
    return {"url": url, "version": version, "digests": {m: digests[m] for m in models}}


@dataclass
class ScriptedBackend:
    """Replies from a list (in order) or a function of the messages, for tests."""

    replies: Sequence[str] | Callable[[Messages, dict | None], str]
    model: str = "scripted"
    calls: list[tuple[Messages, dict | None]] = field(default_factory=list)
    max_tokens: list[int] = field(default_factory=list)

    def chat(
        self,
        messages: Messages,
        *,
        schema: dict | None = None,
        temperature: float,
        seed: int,
        max_tokens: int = 512,
    ) -> Reply:
        self.calls.append(([dict(m) for m in messages], schema))
        self.max_tokens.append(max_tokens)
        if callable(self.replies):
            text = self.replies(messages, schema)
        else:
            text = self.replies[len(self.calls) - 1]
        return Reply(text=text, seconds=0.0)
