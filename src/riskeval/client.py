from __future__ import annotations

import http.client
import json
import socket
import time
from urllib import error, parse, request


class LLMClient:
    def __init__(
        self,
        api_key: str,
        base_url: str,
        api_version: str,
        model: str,
        temperature: float,
        max_tokens: int,
        request_timeout_sec: int,
        max_retries: int,
    ):
        self.api_key = api_key
        self.base_url = base_url.rstrip("/")
        self.api_version = api_version
        self.model = model
        self.temperature = temperature
        self.max_tokens = max_tokens
        self.request_timeout_sec = request_timeout_sec
        self.max_retries = max_retries

    def complete(
        self,
        prompt: str,
        system: str | None = None,
        model: str | None = None,
        image_url: str | None = None,
    ) -> str:
        use_model = model or self.model
        messages = []
        if system:
            messages.append({"role": "system", "content": system})
        if image_url:
            messages.append(
                {
                    "role": "user",
                    "content": [
                        {"type": "text", "text": prompt},
                        {"type": "image_url", "image_url": {"url": image_url}},
                    ],
                }
            )
        else:
            messages.append({"role": "user", "content": prompt})

        payload = {
            "messages": messages,
            "max_tokens": self.max_tokens,
            "top_p": 1,
            "stream": False,
        }
        if self.temperature != 1.0:
            payload["temperature"] = self.temperature

        errors: list[str] = []
        for path_prefix in ("", "/openai"):
            url = self._build_url(path_prefix, use_model)
            try:
                data = self._post_json(url, payload)
                return self._extract_chat_text(data)
            except RuntimeError as exc:
                errors.append(f"{url}: {exc}")

        raise RuntimeError(
            "No supported HKBU chat completion endpoint succeeded.\n" + "\n".join(errors)
        )

    def _build_url(self, path_prefix: str, model: str) -> str:
        quoted_model = parse.quote(model, safe="")
        query = parse.urlencode({"api-version": self.api_version})
        return (
            f"{self.base_url}{path_prefix}/deployments/{quoted_model}/chat/completions?{query}"
        )

    def _post_json(self, url: str, payload: dict) -> dict:
        body = json.dumps(payload).encode("utf-8")
        req = request.Request(url=url, data=body, method="POST")
        req.add_header("Content-Type", "application/json")
        req.add_header("Accept", "application/json")
        # Some gateways expect Bearer auth, others Azure-style api-key.
        req.add_header("Authorization", f"Bearer {self.api_key}")
        req.add_header("api-key", self.api_key)

        last_error: Exception | None = None
        for attempt in range(1, self.max_retries + 1):
            try:
                with request.urlopen(req, timeout=self.request_timeout_sec) as resp:
                    raw = resp.read().decode("utf-8")
                break
            except error.HTTPError as exc:
                detail = exc.read().decode("utf-8", errors="replace")
                if exc.code in {429, 500, 502, 503, 504}:
                    last_error = exc
                    if attempt == self.max_retries:
                        raise RuntimeError(
                            f"HTTP {exc.code} after {self.max_retries} attempts: {detail}"
                        ) from exc
                    print(
                        f"[retry {attempt}/{self.max_retries}] transient HTTP {exc.code}, retrying in {attempt}s",
                        flush=True,
                    )
                    time.sleep(attempt)
                    continue
                raise RuntimeError(f"HTTP {exc.code}: {detail}") from exc
            except (
                TimeoutError,
                socket.timeout,
                error.URLError,
                http.client.RemoteDisconnected,
                http.client.IncompleteRead,
            ) as exc:
                last_error = exc
                if attempt == self.max_retries:
                    reason = getattr(exc, "reason", str(exc))
                    raise RuntimeError(
                        f"Network timeout/error after {self.max_retries} attempts: {reason}"
                    ) from exc
                print(
                    f"[retry {attempt}/{self.max_retries}] request failed, retrying in {attempt}s: {exc}",
                    flush=True,
                )
                time.sleep(attempt)
        else:
            raise RuntimeError(f"Request failed: {last_error}")

        try:
            return json.loads(raw)
        except json.JSONDecodeError as exc:
            raise RuntimeError(f"Invalid JSON response: {raw[:400]}") from exc

    @staticmethod
    def _extract_chat_text(data: dict) -> str:
        choices = data.get("choices")
        if not isinstance(choices, list) or not choices:
            raise RuntimeError(f"Missing choices in response: {json.dumps(data)[:400]}")

        message = choices[0].get("message", {})
        content = message.get("content")
        if isinstance(content, str):
            return content.strip()

        if isinstance(content, list):
            parts: list[str] = []
            for item in content:
                if isinstance(item, dict) and item.get("type") == "text":
                    text = item.get("text")
                    if isinstance(text, str):
                        parts.append(text)
            if parts:
                return "\n".join(parts).strip()

        raise RuntimeError(f"Missing assistant content in response: {json.dumps(data)[:400]}")
