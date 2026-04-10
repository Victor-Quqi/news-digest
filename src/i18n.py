from __future__ import annotations

import ast
import gettext
from pathlib import Path
from typing import Any

try:
    import yaml
except ImportError:  # pragma: no cover
    yaml = None  # type: ignore[assignment]


_LOCALES_DIR = Path(__file__).resolve().parent.parent / "locales"
_MISSING = object()


class _DictTranslations(gettext.NullTranslations):
    def __init__(self, catalog: dict[str, str]) -> None:
        super().__init__()
        self._catalog = catalog

    def gettext(self, message: str) -> str:
        return self._catalog.get(message, message)

    def ngettext(self, singular: str, plural: str, n: int) -> str:
        message = singular if n == 1 else plural
        return self._catalog.get(message, message)


class Locale:
    def __init__(self, lang: str, locales_dir: Path | None = None) -> None:
        self.lang = (lang or "en").strip() or "en"
        self.locales_dir = Path(locales_dir) if locales_dir is not None else _LOCALES_DIR
        self._translations = self._load_translations()
        self._data = self._load_locale_data()

    def _load_translations(self) -> gettext.NullTranslations:
        translations = gettext.translation(
            "messages",
            localedir=str(self.locales_dir),
            languages=[self.lang],
            fallback=True,
        )
        if translations.__class__ is not gettext.NullTranslations:
            return translations

        po_path = self.locales_dir / self.lang / "LC_MESSAGES" / "messages.po"
        if not po_path.exists():
            return translations

        catalog = self._load_po_catalog(po_path)
        if not catalog:
            return translations
        return _DictTranslations(catalog)

    def _load_locale_data(self) -> dict[str, Any]:
        if yaml is None:
            raise RuntimeError("PyYAML is required to load locale.yaml files")

        locale_path = self.locales_dir / self.lang / "locale.yaml"
        if not locale_path.exists():
            return {}

        with locale_path.open("r", encoding="utf-8") as file:
            data = yaml.safe_load(file) or {}

        if not isinstance(data, dict):
            raise ValueError(f"Locale data must be a mapping: {locale_path}")
        return data

    def _load_po_catalog(self, po_path: Path) -> dict[str, str]:
        catalog: dict[str, str] = {}
        msgid_parts: list[str] = []
        msgstr_parts: list[str] = []
        state = ""

        def flush() -> None:
            if not msgid_parts and not msgstr_parts:
                return

            msgid = "".join(msgid_parts)
            msgstr = "".join(msgstr_parts)
            if msgid:
                catalog[msgid] = msgstr or msgid

            msgid_parts.clear()
            msgstr_parts.clear()

        with po_path.open("r", encoding="utf-8") as file:
            for raw_line in file:
                line = raw_line.strip()
                if not line:
                    flush()
                    state = ""
                    continue
                if line.startswith("#"):
                    continue
                if line.startswith("msgid "):
                    flush()
                    msgid_parts.append(self._parse_po_string(line[6:].strip()))
                    state = "msgid"
                    continue
                if line.startswith("msgstr "):
                    msgstr_parts.append(self._parse_po_string(line[7:].strip()))
                    state = "msgstr"
                    continue
                if line.startswith('"'):
                    if state == "msgid":
                        msgid_parts.append(self._parse_po_string(line))
                    elif state == "msgstr":
                        msgstr_parts.append(self._parse_po_string(line))
            flush()

        return catalog

    def _parse_po_string(self, value: str) -> str:
        return str(ast.literal_eval(value))

    def t(self, message: str, **kwargs: Any) -> str:
        translated = self._translations.gettext(message)
        if kwargs:
            return translated.format(**kwargs)
        return translated

    @property
    def email_subject(self) -> str:
        return str(self.get("email_subject", ""))

    def require(self, dotted_key: str) -> Any:
        value = self.get(dotted_key, _MISSING)
        if value is _MISSING:
            raise KeyError(f"Locale key not found: {dotted_key}")
        return value

    def get_prompt(self, step: str, role: str) -> str:
        return str(self.require(f"prompts.{step}.{role}"))

    def render_prompt(self, step: str, role: str, **kwargs: Any) -> str:
        return self.get_prompt(step, role).format(**kwargs)

    @property
    def theme_keywords(self) -> dict[str, list[str]]:
        value = self.get("theme_keywords", {})
        if not isinstance(value, dict):
            return {}
        result: dict[str, list[str]] = {}
        for key, items in value.items():
            if isinstance(items, list):
                result[str(key)] = [str(item) for item in items if str(item).strip()]
        return result

    @property
    def fallback_texts(self) -> dict[str, str]:
        value = self.get("fallback", {})
        if not isinstance(value, dict):
            return {}
        return {str(key): str(item) for key, item in value.items()}

    def get(self, dotted_key: str, default: Any = None) -> Any:
        if not dotted_key:
            return self._data

        value: Any = self._data
        for part in dotted_key.split("."):
            if isinstance(value, dict):
                if part not in value:
                    return default
                value = value[part]
                continue
            if isinstance(value, list) and part.isdigit():
                index = int(part)
                if 0 <= index < len(value):
                    value = value[index]
                    continue
            return default
        return value

    def install_jinja2(self, env: Any) -> None:
        env.install_gettext_translations(self._translations)
