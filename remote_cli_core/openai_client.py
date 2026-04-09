from __future__ import annotations

import ipaddress
import json
import os
import urllib.error
import urllib.parse
import urllib.request
from dataclasses import dataclass
from typing import Any


@dataclass
class OpenAITextResponse:
    response_id: str
    text: str
    raw_payload: dict[str, Any]


class OpenAIAPIError(RuntimeError):
    def __init__(self, message: str, *, code: int | None = None, body: str = "") -> None:
        super().__init__(message)
        self.code = code
        self.body = body


class OpenAIResponsesClient:
    def __init__(
        self,
        *,
        api_key: str | None = None,
        base_url: str | None = None,
        timeout_seconds: int = 180,
    ) -> None:
        self.api_key = api_key or os.environ.get("OPENAI_API_KEY")
        self.base_url = (base_url or os.environ.get("OPENAI_BASE_URL") or "https://api.openai.com/v1").rstrip("/")
        self.timeout_seconds = timeout_seconds
        self.local_base_url = _is_local_base_url(self.base_url)
        self.force_chat_completions = os.environ.get("REMOTE_CLI_ASSISTANT_FORCE_CHAT_COMPLETIONS", "").strip().lower() in {
            "1",
            "true",
            "yes",
            "on",
        }
        if not self.api_key and not self.local_base_url:
            raise RuntimeError("OPENAI_API_KEY is required to use the installation assistant.")

    def create_json_response(
        self,
        *,
        model: str,
        prompt: str,
        schema: dict[str, Any],
        previous_response_id: str | None = None,
        tools: list[dict[str, Any]] | None = None,
        reasoning_effort: str = "medium",
    ) -> OpenAITextResponse:
        responses_payload: dict[str, Any] = {
            "model": model,
            "input": prompt,
            "reasoning": {"effort": reasoning_effort},
            "text": {"format": schema},
        }
        if previous_response_id:
            responses_payload["previous_response_id"] = previous_response_id
        if tools:
            responses_payload["tools"] = tools

        attempts = [responses_payload]
        if tools and any(tool.get("type") == "web_search" for tool in tools):
            fallback_tools = [
                {"type": "web_search_preview"} if tool.get("type") == "web_search" else tool
                for tool in tools
            ]
            attempts.append({**responses_payload, "tools": fallback_tools})

        last_error: OpenAIAPIError | None = None
        if not self.force_chat_completions:
            for payload in attempts:
                try:
                    return self._post_responses(payload)
                except OpenAIAPIError as exc:
                    last_error = exc
                    if not self._should_fallback_to_chat_completions(exc):
                        continue

        if self.local_base_url:
            return self._post_chat_completions(model=model, prompt=prompt, schema=schema)

        if last_error is not None:
            raise last_error
        raise RuntimeError("The OpenAI-compatible client did not return a response.")

    def _post_responses(self, payload: dict[str, Any]) -> OpenAITextResponse:
        raw = self._post_json("/responses", payload)
        return OpenAITextResponse(
            response_id=raw.get("id", ""),
            text=_extract_text(raw),
            raw_payload=raw,
        )

    def _post_chat_completions(
        self,
        *,
        model: str,
        prompt: str,
        schema: dict[str, Any],
    ) -> OpenAITextResponse:
        payload = {
            "model": model,
            "messages": [{"role": "user", "content": prompt}],
            "response_format": _to_chat_completions_response_format(schema),
        }
        raw = self._post_json("/chat/completions", payload)
        return OpenAITextResponse(
            response_id=raw.get("id", ""),
            text=_extract_chat_completions_text(raw),
            raw_payload=raw,
        )

    def _post_json(self, endpoint: str, payload: dict[str, Any]) -> dict[str, Any]:
        request = urllib.request.Request(
            url=f"{self.base_url}{endpoint}",
            data=json.dumps(payload).encode("utf-8"),
            headers=self._headers(),
            method="POST",
        )

        try:
            with urllib.request.urlopen(request, timeout=self.timeout_seconds) as response:
                return json.loads(response.read().decode("utf-8"))
        except urllib.error.HTTPError as exc:
            body = exc.read().decode("utf-8", errors="replace")
            raise OpenAIAPIError(
                f"OpenAI-compatible API error {exc.code}: {body}",
                code=exc.code,
                body=body,
            ) from exc
        except urllib.error.URLError as exc:
            raise OpenAIAPIError(f"OpenAI-compatible API connection failed: {exc}") from exc

    def _headers(self) -> dict[str, str]:
        headers = {"Content-Type": "application/json"}
        if self.api_key:
            headers["Authorization"] = f"Bearer {self.api_key}"
        return headers

    def _should_fallback_to_chat_completions(self, exc: OpenAIAPIError) -> bool:
        if not self.local_base_url:
            return False
        if exc.code in {400, 404, 405, 422, 500, 501}:
            return True
        body = exc.body.lower()
        return any(
            token in body
            for token in (
                "not found",
                "unsupported",
                "unknown field",
                "previous_response_id",
                "response_format",
                "text.format",
                "/responses",
            )
        )


def _extract_text(payload: dict[str, Any]) -> str:
    output = payload.get("output", [])
    chunks: list[str] = []
    for item in output:
        for content in item.get("content", []):
            if content.get("type") == "output_text":
                chunks.append(content.get("text", ""))
    return "".join(chunks).strip()


def _extract_chat_completions_text(payload: dict[str, Any]) -> str:
    choices = payload.get("choices", [])
    if not choices:
        return ""
    message = choices[0].get("message", {})
    content = message.get("content", "")
    if isinstance(content, str):
        return content.strip()
    if isinstance(content, list):
        chunks: list[str] = []
        for item in content:
            if isinstance(item, dict) and item.get("type") in {"text", "output_text"}:
                chunks.append(str(item.get("text", "")))
        return "".join(chunks).strip()
    return str(content).strip()


def _to_chat_completions_response_format(schema: dict[str, Any]) -> dict[str, Any]:
    if schema.get("type") != "json_schema":
        return schema
    return {
        "type": "json_schema",
        "json_schema": {
            "name": schema.get("name", "structured_response"),
            "strict": bool(schema.get("strict", False)),
            "schema": schema.get("schema", {}),
        },
    }


def _is_local_base_url(value: str) -> bool:
    parsed = urllib.parse.urlparse(value)
    hostname = parsed.hostname or ""
    if hostname in {"localhost", "127.0.0.1", "::1"} or hostname.endswith(".local"):
        return True
    try:
        ip = ipaddress.ip_address(hostname)
    except ValueError:
        return False
    return ip.is_loopback or ip.is_private
