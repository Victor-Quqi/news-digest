# news-digest

[English](README.md) | [简体中文](README.zh-CN.md)

一个面向个人或团队的信息订阅项目：抓取 RSS 新闻，清洗和去重内容，调用兼容 OpenAI 的模型做摘要与分类，生成 HTML 邮件并通过 SMTP 发送。

## 特性

- 并行抓取 RSS，单个源失败不会阻塞整体流程
- 两阶段 AI 处理：逐条摘要/分类，再生成日报总览
- 支持中英文界面文案
- 严格校验结构化输出并带重试
- AI 失败时可选降级为原文列表邮件
- 提供 `--dry-run`、`--send-html`、`--ai-debug` 方便验证和排障

## 快速开始

```bash
pip install -r requirements.txt
cp .env.example .env
cp config.yaml.example config.yaml
cp sources.yaml.example sources.yaml
python -m src.main --dry-run
```

首次运行前需要编辑三个运行时文件：

- `.env`：API Key、SMTP 凭证、`RSSHUB_BASE_URL`
- `config.yaml`：语言、调度、AI 行为、过滤和日志
- `sources.yaml`：RSS 源列表

仓库默认不会跟踪 `.env`、`config.yaml`、`sources.yaml` 和 `logs/`。

## CLI

```bash
python -m src.main --dry-run
python -m src.main --dry-run --ai-debug
python -m src.main
python -m src.main --send-html logs/news-digest-YYYYMMDD-HHMMSS.html
```

## Docker

先准备运行时文件：

```bash
cp .env.example .env
cp config.yaml.example config.yaml
cp sources.yaml.example sources.yaml
docker compose up -d --build
```

镜像内只带示例配置，不会把真实密钥烘焙进镜像。实际运行时通过挂载宿主机文件读取配置。如果 RSSHub 跑在其他容器或主机上，请把 `RSSHUB_BASE_URL` 改成容器内可访问的地址。

## 配置

主要配置集中在 `config.yaml`：

- `email`：收件人和邮件标题
- `schedule`：cron 表达式和时区
- `ai`：模型限制、结构化输出策略、重试、降级和调试日志
- `filter`：文章时间窗口和正文长度上限
- `logging`：日志文件和轮转

以 [config.yaml.example](config.yaml.example) 里的英文注释为准。

## 验证

```bash
python -m unittest discover -s tests -v
```

当前 smoke test 覆盖：

- 示例配置加载
- locale 初始化
- HTML 模板渲染

还没有覆盖实时 RSS 抓取、模型调用和 SMTP 发送。

## 其他

- 主体代码在 `src/`
- 多语言文案和 prompts 在 `locales/`
- 后续架构取舍记录在 `docs/architecture-roadmap.md`

## 许可证

[Apache-2.0](LICENSE)
