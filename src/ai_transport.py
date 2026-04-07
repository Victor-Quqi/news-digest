from __future__ import annotations

import json
import logging
import re
from typing import Any, Dict, List, Sequence, Tuple

import httpx
from openai import AsyncOpenAI

from .ai_debug import AIDebugSink
from .ai_processor_types import AIProcessingError
from .config import AIConfig, AIRetryTarget, EnvConfig


class AITransport:
    def __init__(
        self,
        ai_config: AIConfig,
        env_config: EnvConfig,
        logger: logging.Logger,
        debug_sink: AIDebugSink,
    ) -> None:
        self._cfg = ai_config
        self._logger = logger
        self._debug = debug_sink

        async def _rewrite_ua(request: httpx.Request) -> None:
            request.headers["user-agent"] = "python-httpx"

        self._http_client = httpx.AsyncClient(
            timeout=ai_config.timeout,
            trust_env=env_config.openai_use_env_proxy,
            event_hooks={"request": [_rewrite_ua]},
        )
        self._clients: Dict[Tuple[str, str], AsyncOpenAI] = {}

    async def aclose(self) -> None:
        try:
            await self._http_client.aclose()
        except Exception:
            pass

    async def request_json(
        self,
        system_prompt: str,
        user_prompt: str,
        schema: Dict[str, Any],
        phase: str,
        *,
        target: AIRetryTarget,
    ) -> Any:
        messages = [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt},
        ]
        required_keys = self._schema_required_keys(schema)
        format_names, phase_policy, fallback_to_text = self._structured_policy(phase)
        policy = {
            "formats": format_names,
            "policy": phase_policy,
            "fallback_to_text": fallback_to_text,
        }

        structured_errors: List[str] = []
        for response_format in self._structured_formats(schema, format_names):
            text = ""
            try:
                self._debug.dump(
                    "request_json.start",
                    {
                        "phase": phase,
                        "target": self._target_payload(target),
                        "structured_policy": policy,
                        "response_format": response_format,
                        "required_keys": list(required_keys),
                        "messages": messages,
                    },
                )
                text = await self.chat(messages, response_format=response_format, target=target)
                parsed = self.parse_json_text(text, required_keys=required_keys)
                self._debug.dump(
                    "request_json.success",
                    {
                        "phase": phase,
                        "target": self._target_payload(target),
                        "structured_policy": policy,
                        "response_format": response_format,
                        "response_text": text,
                        "parsed": parsed,
                    },
                )
                return parsed
            except Exception as exc:
                structured_errors.append(str(exc))
                self._debug.dump(
                    "request_json.error",
                    {
                        "phase": phase,
                        "target": self._target_payload(target),
                        "structured_policy": policy,
                        "response_format": response_format,
                        "error": str(exc),
                        "response_text": text,
                    },
                    force=True,
                )
                self._logger.warning(
                    "%s structured output failed, trying next format: format=%s, error=%s",
                    phase,
                    response_format.get("type"),
                    exc,
                )

        if phase_policy == "strict":
            detail = structured_errors[-1] if structured_errors else "no structured output formats configured"
            raise AIProcessingError(f"{phase} structured output policy is strict, all attempts failed: {detail}")

        if not fallback_to_text:
            detail = structured_errors[-1] if structured_errors else "text fallback disabled"
            raise AIProcessingError(f"{phase} text fallback not enabled: {detail}")

        if structured_errors:
            self._logger.warning("%s all structured output failed, falling back to plain text JSON", phase)

        text = ""
        try:
            self._debug.dump(
                "request_json.start",
                {
                    "phase": phase,
                    "target": self._target_payload(target),
                    "structured_policy": policy,
                    "response_format": None,
                    "required_keys": list(required_keys),
                    "messages": messages,
                },
            )
            text = await self.chat(messages, response_format=None, target=target)
            parsed = self.parse_json_text(text, required_keys=required_keys)
            self._debug.dump(
                "request_json.success",
                {
                    "phase": phase,
                    "target": self._target_payload(target),
                    "structured_policy": policy,
                    "response_format": None,
                    "response_text": text,
                    "parsed": parsed,
                },
            )
            return parsed
        except Exception as exc:
            self._debug.dump(
                "request_json.error",
                {
                    "phase": phase,
                    "target": self._target_payload(target),
                    "structured_policy": policy,
                    "response_format": None,
                    "error": str(exc),
                    "response_text": text,
                },
                force=True,
            )
            raise

    async def chat(
        self,
        messages: List[Dict[str, str]],
        *,
        response_format: Dict[str, Any] | None,
        target: AIRetryTarget,
    ) -> str:
        kwargs: Dict[str, Any] = {
            "model": target.model,
            "messages": messages,
            "temperature": self._cfg.temperature,
            "max_tokens": self._cfg.max_tokens,
        }
        if response_format is not None:
            kwargs["response_format"] = response_format

        self._debug.dump(
            "chat.request",
            {
                "target": self._target_payload(target),
                "model": target.model,
                "temperature": self._cfg.temperature,
                "max_tokens": self._cfg.max_tokens,
                "response_format": response_format,
                "messages": messages,
            },
        )
        response: Any = None
        try:
            client = self._client_for_target(target)
            response = await client.chat.completions.create(**kwargs)
            content = self.extract_chat_content(response)
            if not content:
                self._debug.dump(
                    "chat.empty_content",
                    {
                        "target": self._target_payload(target),
                        "response_format": response_format,
                        "response_summary": self._response_summary(response),
                        "raw_response": self._response_debug_payload(response),
                    },
                    force=True,
                )
                raise ValueError("Model returned empty content")
            self._debug.dump(
                "chat.response",
                {
                    "target": self._target_payload(target),
                    "response_format": response_format,
                    "content": content,
                    "response_summary": self._response_summary(response),
                },
            )
            return content
        except Exception as exc:
            self._debug.dump(
                "chat.error",
                {
                    "target": self._target_payload(target),
                    "response_format": response_format,
                    "error": str(exc),
                    "response_summary": self._response_summary(response),
                    "raw_response": self._response_debug_payload(response),
                },
                force=True,
            )
            raise

    def _client_for_target(self, target: AIRetryTarget) -> AsyncOpenAI:
        key = (target.base_url, target.api_key)
        client = self._clients.get(key)
        if client is None:
            client = AsyncOpenAI(
                api_key=target.api_key,
                base_url=target.base_url,
                timeout=self._cfg.timeout,
                http_client=self._http_client,
            )
            self._clients[key] = client
        return client

    def _target_payload(self, target: AIRetryTarget) -> Dict[str, Any]:
        return {
            "name": target.name,
            "base_url": target.base_url,
            "model": target.model,
        }

    def extract_chat_content(self, response: Any) -> str:
        if response is None:
            return ""

        if isinstance(response, str):
            raw = response.strip()
            sse_text = self._extract_text_from_sse(raw)
            if "data:" in raw:
                return sse_text
            return sse_text or raw

        if isinstance(response, dict):
            direct = self.extract_chat_content_from_dict(response)
            if direct:
                return direct
            return ""

        choices = getattr(response, "choices", None)
        if isinstance(choices, list) and choices:
            first = choices[0]
            if isinstance(first, dict):
                direct = self.extract_chat_content_from_dict({"choices": [first]})
                if direct:
                    return direct
                return ""
            else:
                message = getattr(first, "message", None)
                if message is not None:
                    content = getattr(message, "content", None)
                    joined = self._join_content_parts(content)
                    if joined:
                        return joined
                delta = getattr(first, "delta", None)
                if delta is not None:
                    joined = self._join_content_parts(getattr(delta, "content", None))
                    if joined:
                        return joined
                return ""

        model_dump = getattr(response, "model_dump", None)
        if callable(model_dump):
            try:
                dumped = model_dump()
                direct = self.extract_chat_content_from_dict(dumped)
                if direct:
                    return direct
                return ""
            except Exception:
                pass

        raw_text = str(response).strip()
        sse_text = self._extract_text_from_sse(raw_text)
        if "data:" in raw_text:
            return sse_text
        return sse_text or raw_text

    def extract_chat_content_from_dict(self, data: Dict[str, Any]) -> str:
        choices = data.get("choices")
        if not isinstance(choices, list) or not choices:
            return ""
        first = choices[0]
        if not isinstance(first, dict):
            return ""

        message = first.get("message")
        if isinstance(message, dict):
            joined = self._join_content_parts(message.get("content"))
            if joined:
                return joined

        delta = first.get("delta")
        if isinstance(delta, dict):
            joined = self._join_content_parts(delta.get("content"))
            if joined:
                return joined
        return ""

    def try_parse_json_payload(self, text: str) -> Any | None:
        payload = (text or "").strip()
        if not payload:
            return None
        try:
            return json.loads(payload)
        except Exception:
            pass

        block = self.extract_first_json_block(payload)
        if block:
            try:
                return json.loads(block)
            except Exception:
                return None
        return None

    def parse_json_text(self, text: str, required_keys: Sequence[str] = ()) -> Any:
        payload = self.extract_first_json_block(text) or text
        parsed = json.loads(payload)
        if isinstance(parsed, dict):
            missing = [key for key in required_keys if key not in parsed]
            if missing:
                raise ValueError(f"missing required keys: {', '.join(missing)}")
        return parsed

    def extract_first_json_block(self, text: str) -> str:
        payload = (text or "").strip()
        if not payload:
            return ""

        fence_match = _extract_json_fence(payload)
        if fence_match:
            return fence_match

        start_positions = [idx for idx in (payload.find("{"), payload.find("[")) if idx >= 0]
        if not start_positions:
            return ""
        start = min(start_positions)
        opening = payload[start]
        closing = "}" if opening == "{" else "]"
        depth = 0
        in_string = False
        escaped = False
        for idx in range(start, len(payload)):
            ch = payload[idx]
            if in_string:
                if escaped:
                    escaped = False
                elif ch == "\\":
                    escaped = True
                elif ch == '"':
                    in_string = False
                continue
            if ch == '"':
                in_string = True
            elif ch == opening:
                depth += 1
            elif ch == closing:
                depth -= 1
                if depth == 0:
                    return payload[start : idx + 1]
        return ""

    def _schema_required_keys(self, schema: Dict[str, Any]) -> Tuple[str, ...]:
        required = schema.get("required")
        if not isinstance(required, list):
            return ()
        return tuple(str(item).strip() for item in required if str(item).strip())

    def _structured_policy(self, phase: str) -> Tuple[List[str], str, bool]:
        if phase == "phase1":
            raw_formats = self._cfg.structured_output_phase1_formats
            raw_policy = str(self._cfg.structured_output_phase1_policy or "").strip()
        elif phase == "phase2":
            raw_formats = self._cfg.structured_output_phase2_formats
            raw_policy = str(self._cfg.structured_output_phase2_policy or "").strip()
        else:
            raw_formats = ["json_schema", "json_object"]
            raw_policy = "prefer"

        formats = self._normalize_structured_format_names(raw_formats, phase=phase)
        policy = self._normalize_structured_policy(raw_policy, phase=phase)
        fallback_to_text = policy == "prefer"
        return formats, policy, fallback_to_text

    def _normalize_structured_policy(self, raw_policy: str, *, phase: str) -> str:
        normalized = str(raw_policy or "").strip().lower()
        if normalized in {"strict", "prefer"}:
            return normalized
        if normalized:
            self._logger.warning(
                "%s ignoring unknown structured policy: %s, falling back to strict",
                phase,
                raw_policy,
            )
        return "strict"

    def _normalize_structured_format_names(
        self,
        raw_formats: Sequence[str],
        *,
        phase: str,
    ) -> List[str]:
        alias = {
            "json_schema": "json_schema",
            "schema": "json_schema",
            "json-schema": "json_schema",
            "json_object": "json_object",
            "object": "json_object",
            "json-object": "json_object",
            "json": "json_object",
        }
        normalized: List[str] = []
        seen: set[str] = set()
        unknown: List[str] = []

        for item in raw_formats:
            key = str(item or "").strip().lower()
            if not key:
                continue
            mapped = alias.get(key)
            if not mapped:
                unknown.append(str(item))
                continue
            if mapped in seen:
                continue
            seen.add(mapped)
            normalized.append(mapped)

        if unknown:
            self._logger.warning(
                "%s ignoring unknown structured output formats: %s",
                phase,
                ", ".join(unknown),
            )
        return normalized

    def _structured_formats(
        self,
        schema: Dict[str, Any],
        format_names: Sequence[str],
    ) -> List[Dict[str, Any]]:
        formats: List[Dict[str, Any]] = []
        for name in format_names:
            if name == "json_schema":
                formats.append(
                    {
                        "type": "json_schema",
                        "json_schema": {
                            "name": "news_digest_response",
                            "strict": True,
                            "schema": schema,
                        },
                    }
                )
            elif name == "json_object":
                formats.append({"type": "json_object"})
        return formats

    def _join_content_parts(self, content: Any) -> str:
        if content is None:
            return ""
        if isinstance(content, str):
            return content
        if isinstance(content, list):
            parts: List[str] = []
            for item in content:
                if isinstance(item, str):
                    parts.append(item)
                elif isinstance(item, dict):
                    text = item.get("text")
                    if isinstance(text, str):
                        parts.append(text)
            return "".join(parts).strip()
        return str(content).strip()

    def _extract_text_from_sse(self, payload: str) -> str:
        text = (payload or "").strip()
        if "data:" not in text:
            return ""

        chunks: List[str] = []
        for raw_line in text.splitlines():
            line = raw_line.strip()
            if not line.startswith("data:"):
                continue

            data_part = line[5:].strip()
            if not data_part or data_part == "[DONE]":
                continue

            try:
                obj = json.loads(data_part)
            except Exception:
                continue

            content = self.extract_chat_content_from_dict(obj)
            if content:
                chunks.append(content)

        return "".join(chunks).strip()

    def _response_debug_payload(self, response: Any) -> Any:
        if response is None:
            return None
        model_dump = getattr(response, "model_dump", None)
        if callable(model_dump):
            try:
                return model_dump()
            except Exception:
                pass
        if isinstance(response, dict):
            return response
        return str(response)

    def _response_summary(self, response: Any) -> Dict[str, Any]:
        if response is None:
            return {}

        payload = self._response_debug_payload(response)
        if not isinstance(payload, dict):
            return {
                "python_type": type(response).__name__,
                "preview": str(payload)[:2000],
            }

        choices = payload.get("choices")
        choice_summaries: List[Dict[str, Any]] = []
        if isinstance(choices, list):
            for item in choices[:3]:
                if not isinstance(item, dict):
                    continue
                message = item.get("message")
                delta = item.get("delta")
                choice_summaries.append(
                    {
                        "index": item.get("index"),
                        "finish_reason": item.get("finish_reason"),
                        "message": self._message_summary(message),
                        "delta": self._message_summary(delta),
                    }
                )

        summary: Dict[str, Any] = {
            "python_type": type(response).__name__,
            "object": payload.get("object"),
            "id": payload.get("id"),
            "model": payload.get("model"),
            "system_fingerprint": payload.get("system_fingerprint"),
            "service_tier": payload.get("service_tier"),
            "usage": payload.get("usage"),
            "choices": choice_summaries,
        }

        error = payload.get("error")
        if error is not None:
            summary["error"] = error

        extra_keys = [
            key
            for key in ("output", "response", "data", "result")
            if key in payload
        ]
        if extra_keys:
            summary["extra_top_level_keys"] = extra_keys
        return summary

    def _message_summary(self, message: Any) -> Dict[str, Any] | None:
        if message is None:
            return None
        if not isinstance(message, dict):
            model_dump = getattr(message, "model_dump", None)
            if callable(model_dump):
                try:
                    message = model_dump()
                except Exception:
                    return {"python_type": type(message).__name__, "preview": str(message)[:1000]}
            else:
                return {"python_type": type(message).__name__, "preview": str(message)[:1000]}
        if not isinstance(message, dict):
            return {"preview": str(message)[:1000]}

        content = message.get("content")
        content_text = self._join_content_parts(content)
        return {
            "role": message.get("role"),
            "content_type": type(content).__name__ if content is not None else None,
            "content_preview": content_text[:1000] if content_text else "",
            "refusal": message.get("refusal"),
            "tool_calls": message.get("tool_calls"),
            "function_call": message.get("function_call"),
            "annotations": message.get("annotations"),
        }


def _extract_json_fence(payload: str) -> str:
    match = re.search(r"```(?:json)?\s*(.*?)```", payload, flags=re.DOTALL | re.IGNORECASE)
    if not match:
        return ""
    return match.group(1).strip()
