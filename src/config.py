from __future__ import annotations

import os
import re
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List

import yaml
from dotenv import load_dotenv


ENV_PATTERN = re.compile(r"\$\{([A-Z0-9_]+)\}")


@dataclass
class RSSSource:
    url: str
    name: str


@dataclass
class EmailConfig:
    recipients: List[str] = field(default_factory=list)
    subject: str = "News Digest"


@dataclass
class ScheduleConfig:
    cron: str = "0 8 * * *"
    timezone: str = "Asia/Shanghai"


@dataclass
class AIConfig:
    temperature: float = 0.3
    max_tokens: int = 4000
    timeout: int = 60
    context_window: int = 128000
    shard_threshold_ratio: float = 0.6
    shard_max_articles: int = 35
    shard_max_chars: int = 120000
    structured_output_phase1_formats: List[str] = field(
        default_factory=lambda: ["json_schema", "json_object"]
    )
    structured_output_phase1_policy: str = "strict"  # strict / prefer
    structured_output_phase2_formats: List[str] = field(
        default_factory=lambda: ["json_schema", "json_object"]
    )
    structured_output_phase2_policy: str = "strict"  # strict / prefer
    transient_retry_max: int = 3
    schema_retry_max: int = 3
    backoff_seconds: List[int] = field(default_factory=lambda: [1, 2, 4])
    jitter_ms_max: int = 300
    taxonomy: List[str] = field(
        default_factory=lambda: ["Tech", "Finance", "Policy", "Market", "International", "Other"]
    )
    phase2_text_fallback: bool = False
    phase2_local_fallback: bool = False
    fallback_send_raw_email: bool = False
    fallback_warning_text: str = "⚠ AI summarization/categorization failed. This email is a degraded raw-article version."
    one_line_hard_units: float = 42.0
    one_line_soft_units: float = 50.0
    one_line_trim_target_units: float = 48.0
    summary_line_target_len: int = 96
    summary_line_hard_len: int = 112
    summary_line_soft_len: int = 132
    debug_dump_on_error: bool = True
    debug_dump_all: bool = False
    debug_dump_dir: str = "logs/ai-debug"
    debug_dump_max_bytes: int = 20 * 1024 * 1024
    debug_dump_retention_days: int = 14
    debug_dump_max_files: int = 300


@dataclass
class FilterConfig:
    hours_back: int = 24
    max_content_length: int = 6000


@dataclass
class LoggingConfig:
    level: str = "INFO"
    file: str = "logs/news-digest.log"
    max_bytes: int = 10 * 1024 * 1024
    backup_count: int = 5


@dataclass
class EnvConfig:
    openai_api_key: str
    openai_base_url: str
    openai_model: str
    openai_use_env_proxy: bool
    smtp_host: str
    smtp_port: int
    smtp_user: str
    smtp_password: str


@dataclass
class AppConfig:
    rss_sources: List[RSSSource]
    email: EmailConfig
    schedule: ScheduleConfig
    ai: AIConfig
    filter: FilterConfig
    logging: LoggingConfig
    env: EnvConfig
    locale: str


def _replace_env_in_string(value: str) -> str:
    def repl(match: re.Match[str]) -> str:
        key = match.group(1)
        env_value = os.getenv(key)
        if env_value is None:
            raise ValueError(f"Config references unset env var: {key}")
        return env_value

    return ENV_PATTERN.sub(repl, value)


def _replace_env(data: Any) -> Any:
    if isinstance(data, dict):
        return {k: _replace_env(v) for k, v in data.items()}
    if isinstance(data, list):
        return [_replace_env(v) for v in data]
    if isinstance(data, str):
        return _replace_env_in_string(data)
    return data


def _required_env(key: str) -> str:
    value = os.getenv(key, "").strip()
    if not value:
        raise ValueError(f"Missing env var: {key}")
    return value


def _env_bool(key: str, default: bool = False) -> bool:
    value = os.getenv(key)
    if value is None:
        return default
    normalized = value.strip().lower()
    if not normalized:
        return default
    return normalized in {"1", "true", "yes", "on"}


def _to_string_list(value: Any, fallback: List[str]) -> List[str]:
    if isinstance(value, str):
        parts = [x.strip() for x in value.split(",") if x.strip()]
        return parts or list(fallback)
    if isinstance(value, list):
        parts = [str(x).strip() for x in value if str(x).strip()]
        return parts or list(fallback)
    return list(fallback)


def _to_int_list(value: Any, fallback: List[int]) -> List[int]:
    if isinstance(value, list):
        items = [int(x) for x in value]
        return items or list(fallback)
    if value in (None, ""):
        return list(fallback)
    return [int(value)]


def _to_bool(value: Any, fallback: bool) -> bool:
    if value is None:
        return fallback
    if isinstance(value, bool):
        return value
    if isinstance(value, str):
        normalized = value.strip().lower()
        if not normalized:
            return fallback
        if normalized in {"1", "true", "yes", "on"}:
            return True
        if normalized in {"0", "false", "no", "off"}:
            return False
    return bool(value)


def _load_env() -> EnvConfig:
    return EnvConfig(
        openai_api_key=_required_env("OPENAI_API_KEY"),
        openai_base_url=os.getenv("OPENAI_BASE_URL", "https://api.openai.com/v1").strip(),
        openai_model=os.getenv("OPENAI_MODEL", "gpt-4o-mini").strip(),
        openai_use_env_proxy=_env_bool("OPENAI_USE_ENV_PROXY", False),
        smtp_host=_required_env("SMTP_HOST"),
        smtp_port=int(os.getenv("SMTP_PORT", "587")),
        smtp_user=_required_env("SMTP_USER"),
        smtp_password=_required_env("SMTP_PASSWORD"),
    )


def load_config(
    config_path: str = "config.yaml",
    sources_path: str = "sources.yaml",
) -> AppConfig:
    path = Path(config_path)
    if not path.exists():
        raise FileNotFoundError(f"Config file not found: {config_path}")
    path = path.resolve()
    base_dir = path.parent

    load_dotenv(dotenv_path=base_dir / ".env")

    with path.open("r", encoding="utf-8") as f:
        raw: Dict[str, Any] = yaml.safe_load(f) or {}

    resolved = _replace_env(raw)
    locale = str(raw.get("locale", resolved.get("locale", "zh")) or "zh").strip() or "zh"
    env = _load_env()

    # Load RSS sources from separate file (sources.yaml)
    src_path = Path(sources_path)
    if not src_path.is_absolute():
        src_path = base_dir / src_path
    if not src_path.exists():
        raise FileNotFoundError(f"RSS sources file not found: {src_path}")
    with src_path.open("r", encoding="utf-8") as f:
        raw_sources: Any = yaml.safe_load(f) or []
    resolved_sources = _replace_env(raw_sources)

    sources = [
        RSSSource(url=item["url"], name=item["name"])
        for item in resolved_sources
        if isinstance(item, dict) and item.get("url") and item.get("name")
    ]
    if not sources:
        raise ValueError("sources.yaml must not be empty")

    email_cfg = resolved.get("email", {})
    schedule_cfg = resolved.get("schedule", {})
    ai_cfg = resolved.get("ai", {})
    filter_cfg = resolved.get("filter", {})
    logging_cfg = resolved.get("logging", {})
    email_defaults = EmailConfig()
    schedule_defaults = ScheduleConfig()
    ai_defaults = AIConfig()
    filter_defaults = FilterConfig()
    logging_defaults = LoggingConfig()

    return AppConfig(
        rss_sources=sources,
        email=EmailConfig(
            recipients=_to_string_list(email_cfg.get("to", email_defaults.recipients), email_defaults.recipients),
            subject=str(email_cfg.get("subject", email_defaults.subject)),
        ),
        schedule=ScheduleConfig(
            cron=str(schedule_cfg.get("cron", schedule_defaults.cron)),
            timezone=str(schedule_cfg.get("timezone", schedule_defaults.timezone)),
        ),
        ai=AIConfig(
            temperature=float(ai_cfg.get("temperature", ai_defaults.temperature)),
            max_tokens=int(ai_cfg.get("max_tokens", ai_defaults.max_tokens)),
            timeout=int(ai_cfg.get("timeout", ai_defaults.timeout)),
            context_window=int(ai_cfg.get("context_window", ai_defaults.context_window)),
            shard_threshold_ratio=float(ai_cfg.get("shard_threshold_ratio", ai_defaults.shard_threshold_ratio)),
            shard_max_articles=int(ai_cfg.get("shard_max_articles", ai_defaults.shard_max_articles)),
            shard_max_chars=int(ai_cfg.get("shard_max_chars", ai_defaults.shard_max_chars)),
            structured_output_phase1_formats=_to_string_list(
                ai_cfg.get("structured_output_phase1_formats", ai_defaults.structured_output_phase1_formats),
                ai_defaults.structured_output_phase1_formats,
            ),
            structured_output_phase1_policy=str(
                ai_cfg.get("structured_output_phase1_policy", ai_defaults.structured_output_phase1_policy)
            ).strip(),
            structured_output_phase2_formats=_to_string_list(
                ai_cfg.get("structured_output_phase2_formats", ai_defaults.structured_output_phase2_formats),
                ai_defaults.structured_output_phase2_formats,
            ),
            structured_output_phase2_policy=str(
                ai_cfg.get("structured_output_phase2_policy", ai_defaults.structured_output_phase2_policy)
            ).strip(),
            transient_retry_max=int(ai_cfg.get("transient_retry_max", ai_defaults.transient_retry_max)),
            schema_retry_max=int(ai_cfg.get("schema_retry_max", ai_defaults.schema_retry_max)),
            backoff_seconds=_to_int_list(ai_cfg.get("backoff_seconds", ai_defaults.backoff_seconds), ai_defaults.backoff_seconds),
            jitter_ms_max=int(ai_cfg.get("jitter_ms_max", ai_defaults.jitter_ms_max)),
            taxonomy=_to_string_list(ai_cfg.get("taxonomy", ai_defaults.taxonomy), ai_defaults.taxonomy),
            phase2_text_fallback=_to_bool(ai_cfg.get("phase2_text_fallback"), ai_defaults.phase2_text_fallback),
            phase2_local_fallback=_to_bool(ai_cfg.get("phase2_local_fallback"), ai_defaults.phase2_local_fallback),
            fallback_send_raw_email=_to_bool(ai_cfg.get("fallback_send_raw_email"), ai_defaults.fallback_send_raw_email),
            fallback_warning_text=str(
                ai_cfg.get("fallback_warning_text", ai_defaults.fallback_warning_text)
            ),
            one_line_hard_units=float(ai_cfg.get("one_line_hard_units", ai_defaults.one_line_hard_units)),
            one_line_soft_units=float(ai_cfg.get("one_line_soft_units", ai_defaults.one_line_soft_units)),
            one_line_trim_target_units=float(ai_cfg.get("one_line_trim_target_units", ai_defaults.one_line_trim_target_units)),
            summary_line_target_len=int(ai_cfg.get("summary_line_target_len", ai_defaults.summary_line_target_len)),
            summary_line_hard_len=int(ai_cfg.get("summary_line_hard_len", ai_defaults.summary_line_hard_len)),
            summary_line_soft_len=int(ai_cfg.get("summary_line_soft_len", ai_defaults.summary_line_soft_len)),
            debug_dump_on_error=_to_bool(ai_cfg.get("debug_dump_on_error"), ai_defaults.debug_dump_on_error),
            debug_dump_all=_to_bool(ai_cfg.get("debug_dump_all"), ai_defaults.debug_dump_all),
            debug_dump_dir=str(ai_cfg.get("debug_dump_dir", ai_defaults.debug_dump_dir)),
            debug_dump_max_bytes=int(ai_cfg.get("debug_dump_max_bytes", ai_defaults.debug_dump_max_bytes)),
            debug_dump_retention_days=int(ai_cfg.get("debug_dump_retention_days", ai_defaults.debug_dump_retention_days)),
            debug_dump_max_files=int(ai_cfg.get("debug_dump_max_files", ai_defaults.debug_dump_max_files)),
        ),
        filter=FilterConfig(
            hours_back=int(filter_cfg.get("hours_back", filter_defaults.hours_back)),
            max_content_length=int(filter_cfg.get("max_content_length", filter_defaults.max_content_length)),
        ),
        logging=LoggingConfig(
            level=str(logging_cfg.get("level", logging_defaults.level)).upper(),
            file=str(logging_cfg.get("file", logging_defaults.file)),
            max_bytes=int(logging_cfg.get("max_bytes", logging_defaults.max_bytes)),
            backup_count=int(logging_cfg.get("backup_count", logging_defaults.backup_count)),
        ),
        env=env,
        locale=locale,
    )
