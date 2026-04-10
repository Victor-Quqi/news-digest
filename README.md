# news-digest

[English](README.md) | [简体中文](README.zh-CN.md)

Daily RSS-to-email digest for teams or personal monitoring. The pipeline fetches RSS feeds, cleans and deduplicates articles, asks an OpenAI-compatible model to summarize, optionally suggest category candidates, categorize each item, generate the overview headline and detail lines, renders an HTML email, and sends it through SMTP.

## Highlights

- Parallel RSS fetching with per-source failure isolation
- AI pipeline: per-article summarization, optional category suggestion, per-article categorization, then split overview generation
- i18n support for Chinese and English UI copy
- Strict structured-output validation with retries
- Optional degraded raw-article email when AI output fails
- `--config`, `--dry-run`, `--send-html`, `--ai-debug`, `--ai-debug-dir`, `--log-level`, and `--timing`

## Quick Start

```bash
pip install -r requirements.txt
cp .env.example .env
cp config.yaml.example config.yaml
cp sources.yaml.example sources.yaml
python -m src.main --dry-run
```

Edit the three runtime files before the first real run:

- `.env`: API key, SMTP credentials, `RSSHUB_BASE_URL`
- `config.yaml`: locale, scheduling, AI behavior, filtering, logging
- `sources.yaml`: RSS source list

The repository does not track `.env`, `config.yaml`, `sources.yaml`, or `logs/`.

## CLI

```bash
python -m src.main --dry-run
python -m src.main --config config.yaml --dry-run --timing
python -m src.main --dry-run --ai-debug
python -m src.main --dry-run --ai-debug --ai-debug-dir logs/ai-debug
python -m src.main --dry-run --log-level DEBUG
python -m src.main
python -m src.main --send-html logs/news-digest-YYYYMMDD-HHMMSS.html
```

## Docker

Prepare runtime files first:

```bash
cp .env.example .env
cp config.yaml.example config.yaml
cp sources.yaml.example sources.yaml
docker compose up -d --build
```

The image only includes example config files. Real runtime files are mounted from the host, so secrets are not baked into the image. If RSSHub runs in another container or on another host, set `RSSHUB_BASE_URL` to an address reachable from inside the container.

## Configuration

Most behavior is controlled in `config.yaml`:

- `email`: recipients and subject, with config taking priority over locale defaults
- `schedule`: cron expression and timezone
- `ai`: model limits, structured-output policy, primary/backup retry targets, optional preferred category hints, categorization strictness, fallback behavior, debug logging
- `filter`: article time window, content length cap, and missing-RSS-time strictness
- `logging`: file path and rotation

The comments in [config.yaml.example](config.yaml.example) are the canonical reference.

Two config details worth knowing:

- If `preferred_categories` is empty, the model first proposes a compact set of category candidates for the current batch and then reuses them during per-article categorization.
- Model failover is driven by `summarization_retry_targets` and `overview_retry_targets`. To switch to a backup model, define a second target and set the matching `OPENAI_BACKUP_*` variables in `.env`.

## Validation

Run the tests:

```bash
python -m unittest discover -s tests -v
```

What this covers:

- example config loading
- `AIProcessor` focused unit checks
- HTML template rendering

It does not cover live RSS fetching, model calls, or SMTP delivery.

## Notes

- Core application code lives in `src/`
- Locale and prompt data live in `locales/`
- Future architecture decisions are recorded in `docs/architecture-roadmap.md`

## License

[Apache-2.0](LICENSE)
