"""Small secret-safe client for OpenAI-compatible model endpoints."""

from __future__ import annotations

import json
import urllib.error
import urllib.request
from dataclasses import dataclass
from typing import Any


class InferenceError(RuntimeError):
    """Base class for endpoint failures safe to display in logs."""


class AuthenticationError(InferenceError):
    """Raised when the endpoint rejects its API credential."""


@dataclass(frozen=True)
class InferenceClient:
    """Call one model-specific endpoint without inheriting proxy settings."""

    endpoint: str
    api_key: str
    request_path: str = "/"
    model: str | None = None
    timeout: int = 120
    retries: int = 2

    def _url(self) -> str:
        endpoint = self.endpoint.rstrip("/")
        path = self.request_path if self.request_path.startswith("/") else f"/{self.request_path}"
        return endpoint + path

    def _payload(self, system_prompt: str, user_prompt: str) -> dict[str, Any]:
        payload: dict[str, Any] = {
            "messages": [
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt},
            ],
            "temperature": 0,
            "response_format": {"type": "json_object"},
            "stream": False,
        }
        if self.model:
            payload["model"] = self.model
        return payload

    def _request(self, payload: dict[str, Any]) -> dict[str, Any]:
        if not self.api_key:
            raise AuthenticationError("Inference API authentication is not configured")
        request = urllib.request.Request(
            self._url(),
            data=json.dumps(payload, ensure_ascii=False).encode("utf-8"),
            headers={
                "Authorization": f"Bearer {self.api_key}",
                "Content-Type": "application/json",
            },
            method="POST",
        )
        opener = urllib.request.build_opener(urllib.request.ProxyHandler({}))
        if self.retries < 0:
            raise ValueError("retries cannot be negative")
        for attempt in range(self.retries + 1):
            try:
                with opener.open(request, timeout=self.timeout) as response:
                    body = json.loads(response.read().decode("utf-8"))
                break
            except urllib.error.HTTPError as error:
                if error.code in {401, 403}:
                    raise AuthenticationError(
                        f"Inference API authentication failed with HTTP {error.code}"
                    ) from None
                if error.code not in {408, 429} and error.code < 500:
                    raise InferenceError(
                        f"Inference API returned HTTP {error.code}"
                    ) from None
                if attempt == self.retries:
                    raise InferenceError(
                        f"Inference API returned HTTP {error.code} after "
                        f"{attempt + 1} attempts"
                    ) from None
            except urllib.error.URLError as error:
                if attempt == self.retries:
                    reason = getattr(error, "reason", None)
                    raise InferenceError(
                        "Inference API transport failed after "
                        f"{attempt + 1} attempts: {type(reason).__name__}"
                    ) from None
            except (UnicodeDecodeError, json.JSONDecodeError):
                raise InferenceError("Inference API response was not valid JSON") from None
        if not isinstance(body, dict):
            raise InferenceError("Inference API response must be a JSON object")
        return body

    @staticmethod
    def _content(body: dict[str, Any]) -> str:
        choices = body.get("choices")
        if not isinstance(choices, list) or not choices:
            raise InferenceError("Inference API response has no choices")
        first = choices[0]
        if not isinstance(first, dict):
            raise InferenceError("Inference API choice must be an object")
        message = first.get("message")
        content = message.get("content") if isinstance(message, dict) else None
        if not isinstance(content, str) or not content.strip():
            raise InferenceError("Inference API response has empty message content")
        return content

    def generate_json(self, *, system_prompt: str, user_prompt: str) -> dict[str, Any]:
        """Generate and parse one strict JSON model response."""

        body = self._request(self._payload(system_prompt, user_prompt))
        try:
            result = json.loads(self._content(body))
        except json.JSONDecodeError:
            raise ValueError("Model content was not valid JSON") from None
        if not isinstance(result, dict):
            raise ValueError("Model content must be a JSON object")
        return result

    def preflight(self) -> None:
        """Verify authentication and request compatibility before a batch run."""

        self.generate_json(
            system_prompt="Return strict JSON only.",
            user_prompt='Return exactly {"ok": true}.',
        )
