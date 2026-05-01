# @brutaldeluxe/pi-synthetic

[Synthetic.new](https://synthetic.new) provider extension for [Pi](https://github.com/mariozechner/pi-coding-agent) — access open-source models through an OpenAI-compatible API with quota tracking, model caching, and zero-data-retention web search.

## Install

```bash
pi install git:github.com/brutaldeluxe82/pi-synthetic
/reload
```

Then set your API key:

```bash
export SYNTHETIC_API_KEY=your-key-here
```

## What it does

- **Registers the `synthetic` provider** with all available models from the Synthetic.new API
- **Auto-refreshes** the model catalog on session start (every 6 hours or when cache is stale)
- **Caches models** locally so the extension works offline with a recent catalog
- **Tracks quotas** with a nice ANSI progress bar display
- **Web search** via `synthetic_web_search` tool — zero-data-retention, fresh results

## Commands

| Command | Description |
|---------|-------------|
| `/synthetic:quotas` | Show weekly token and rolling 5h usage quotas |
| `/synthetic:refresh` | Refresh the model catalog from the API |

## Tools

| Tool | Description |
|------|-------------|
| `synthetic_web_search` | Search the web via Synthetic's zero-data-retention API |

## Model features

- **Reasoning effort mapping** — maps Synthetic's `minimal`/`low`/`medium`/`high`/`xhigh` levels to Pi's internal effort levels
- **Per-model compat overrides** — handles model-specific API quirks (e.g. MiniMax-M2.5 needs `maxTokensField: "max_completion_tokens"`)
- **Diff tracking** — on refresh, shows which models were added, removed, or changed

## Configuration

Set `SYNTHETIC_API_KEY` in your environment. Get a key from [synthetic.new](https://synthetic.new/?referral=NDWw1u3UDWiFyDR).
