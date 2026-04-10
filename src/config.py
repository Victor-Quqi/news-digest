from __future__ import annotations

import os
import re
import logging
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
class AIRetryTarget:
    name: str = ""
    base_url: str = ""
    api_key: str = ""
    model: str = ""


@dataclass
class AIConfig:
    temperature: float = 0.3
    max_tokens: int = 4000
    timeout: int = 60
    context_window: int = 128000
    shard_threshold_ratio: float = 0.6
    shard_max_articles: int = 35
    shard_max_chars: int = 120000
    structured_output_summarization_formats: List[str] = field(
        default_factory=lambda: ["json_schema", "json_object"]
    )
    structured_output_summarization_policy: str = "strict"  # strict / prefer
    structured_output_overview_formats: List[str] = field(
        default_factory=lambda: ["json_schema", "json_object"]
    )
    structured_output_overview_policy: str = "strict"  # strict / prefer
    transient_retry_max: int = 8
    schema_retry_max: int = 8
    retry_error_keywords: List[str] = field(default_factory=list)
    retry_target_failure_threshold: int = 3
    summarization_retry_targets: List[AIRetryTarget] = field(default_factory=list)
    overview_retry_targets: List[AIRetryTarget] = field(default_factory=list)
    backoff_seconds: List[int] = field(default_factory=lambda: [1, 2, 4])
    jitter_ms_max: int = 300
    preferred_categories: List[str] = field(default_factory=list)
    categorization_strict: bool = True
    overview_text_fallback: bool = False
    overview_local_fallback: bool = False
    fallback_send_raw_email: bool = False
    fallback_warning_text: str = "⚠ AI summarization/categorization failed. This email is a degraded raw-article version."
    one_line_hard_units: float = 42.0
    one_line_soft_units: float = 50.0
    one_line_trim_target_units: float = 48.0
    summary_line_target_len: int = 120
    summary_line_hard_len: int = 140
    summary_line_soft_len: int = 168
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
    rss_missing_pub_date_strict: bool = True


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


_logger = logging.getLogger(__name__)


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


def _to_optional_string_list(value: Any, fallback: List[str]) -> List[str]:
    if value is None:
        return list(fallback)
    if isinstance(value, str):
        return [x.strip() for x in value.split(",") if x.strip()]
    if isinstance(value, list):
        return [str(x).strip() for x in value if str(x).strip()]
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


def _validate_retry_target(target: AIRetryTarget, *, label: str, index: int) -> AIRetryTarget:
    name = str(target.name or "").strip() or f"{label}-{index + 1}"
    base_url = str(target.base_url or "").strip()
    api_key = str(target.api_key or "").strip()
    model = str(target.model or "").strip()

    if not base_url:
        raise ValueError(f"{label}[{index}] missing base_url")
    if not api_key:
        raise ValueError(f"{label}[{index}] missing api_key")
    if not model:
        raise ValueError(f"{label}[{index}] missing model")

    return AIRetryTarget(
        name=name,
        base_url=base_url,
        api_key=api_key,
        model=model,
    )


def _parse_retry_targets(
    value: Any,
    *,
    label: str,
    fallback: AIRetryTarget,
) -> List[AIRetryTarget]:
    if value in (None, ""):
        items: List[Any] = [fallback]
    elif isinstance(value, list):
        items = value
    else:
        raise ValueError(f"{label} must be a list")

    targets: List[AIRetryTarget] = []
    for idx, item in enumerate(items):
        if isinstance(item, AIRetryTarget):
            target = item
        elif isinstance(item, dict):
            target = AIRetryTarget(
                name=str(item.get("name", "") or "").strip(),
                base_url=str(item.get("base_url", "") or "").strip(),
                api_key=str(item.get("api_key", "") or "").strip(),
                model=str(item.get("model", "") or "").strip(),
            )
        else:
            raise ValueError(f"{label}[{idx}] must be an object")
        targets.append(_validate_retry_target(target, label=label, index=idx))

    if not targets:
        raise ValueError(f"{label} must not be empty")
    return targets


def _load_env() -> EnvConfig:
    return EnvConfig(
        openai_api_key=os.getenv("OPENAI_API_KEY", "").strip(),
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
    locale = str(resolved.get("locale", "zh") or "zh").strip() or "zh"
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

    sources: List[RSSSource] = []
    for index, item in enumerate(resolved_sources):
        if not isinstance(item, dict):
            _logger.warning("Skipping invalid RSS source at index %d: item must be an object", index)
            continue
        url = str(item.get("url", "") or "").strip()
        name = str(item.get("name", "") or "").strip()
        if not url or not name:
            _logger.warning(
                "Skipping invalid RSS source at index %d: missing url or name",
                index,
            )
            continue
        sources.append(RSSSource(url=url, name=name))
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
    default_retry_target = AIRetryTarget(
        name="primary",
        base_url=env.openai_base_url,
        api_key=env.openai_api_key,
        model=env.openai_model,
    )
    summarization_retry_targets = _parse_retry_targets(
        ai_cfg.get("summarization_retry_targets"),
        label="ai.summarization_retry_targets",
        fallback=default_retry_target,
    )
    overview_retry_targets = _parse_retry_targets(
        ai_cfg.get("overview_retry_targets"),
        label="ai.overview_retry_targets",
        fallback=default_retry_target,
    )

    return AppConfig(
        rss_sources=sources,
        email=EmailConfig(
            recipients=_to_string_list(email_cfg.get("to", email_defaults.recipients), email_defaults.recipients),
            subject=str(email_cfg.get("subject", "") or "").strip(),
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
            structured_output_summarization_formats=_to_optional_string_list(
                ai_cfg.get("structured_output_summarization_formats", ai_defaults.structured_output_summarization_formats),
                ai_defaults.structured_output_summarization_formats,
            ),
            structured_output_summarization_policy=str(
                ai_cfg.get("structured_output_summarization_policy", ai_defaults.structured_output_summarization_policy)
            ).strip(),
            structured_output_overview_formats=_to_optional_string_list(
                ai_cfg.get("structured_output_overview_formats", ai_defaults.structured_output_overview_formats),
                ai_defaults.structured_output_overview_formats,
            ),
            structured_output_overview_policy=str(
                ai_cfg.get("structured_output_overview_policy", ai_defaults.structured_output_overview_policy)
            ).strip(),
            transient_retry_max=int(ai_cfg.get("transient_retry_max", ai_defaults.transient_retry_max)),
            schema_retry_max=int(ai_cfg.get("schema_retry_max", ai_defaults.schema_retry_max)),
            retry_error_keywords=_to_string_list(
                ai_cfg.get("retry_error_keywords", ai_defaults.retry_error_keywords),
                ai_defaults.retry_error_keywords,
            ),
            retry_target_failure_threshold=max(
                int(ai_cfg.get("retry_target_failure_threshold", ai_defaults.retry_target_failure_threshold)),
                1,
            ),
            summarization_retry_targets=summarization_retry_targets,
            overview_retry_targets=overview_retry_targets,
            backoff_seconds=_to_int_list(ai_cfg.get("backoff_seconds", ai_defaults.backoff_seconds), ai_defaults.backoff_seconds),
            jitter_ms_max=int(ai_cfg.get("jitter_ms_max", ai_defaults.jitter_ms_max)),
            preferred_categories=_to_optional_string_list(
                ai_cfg.get("preferred_categories", ai_defaults.preferred_categories),
                ai_defaults.preferred_categories,
            ),
            categorization_strict=_to_bool(
                ai_cfg.get("categorization_strict"),
                ai_defaults.categorization_strict,
            ),
            overview_text_fallback=_to_bool(ai_cfg.get("overview_text_fallback"), ai_defaults.overview_text_fallback),
            overview_local_fallback=_to_bool(ai_cfg.get("overview_local_fallback"), ai_defaults.overview_local_fallback),
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
            rss_missing_pub_date_strict=_to_bool(
                filter_cfg.get("rss_missing_pub_date_strict"),
                filter_defaults.rss_missing_pub_date_strict,
            ),
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
