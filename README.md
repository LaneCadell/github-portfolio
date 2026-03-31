# Gambling Addiction Support Email Agent

A conversational AI agent built with [Pydantic AI](https://ai.pydantic.dev/) that helps users reach gambling addiction support resources by composing and sending a support request email on their behalf.

---

## Features

- Multi-turn conversation — the agent collects the user's name, email, and situation naturally across multiple messages
- Sends a formatted support email via SMTP with `Reply-To` set to the user so support can respond directly
- Provides helpline resources on demand
- Structured output via Pydantic models
- Works with OpenAI (GPT-4o) or Anthropic (Claude) models

---

## Requirements

- Python 3.10+
- An SMTP-capable email account (e.g. Gmail with an App Password)
- An OpenAI or Anthropic API key

---

## Installation

```bash
pip install pydantic-ai pydantic
```

---

## Configuration

Set the following environment variables before running:

| Variable | Description | Default |
|---|---|---|
| `SMTP_HOST` | SMTP server hostname | `smtp.gmail.com` |
| `SMTP_PORT` | SMTP server port | `587` |
| `SMTP_USER` | Your sending email address | — |
| `SMTP_PASS` | Your SMTP password or App Password | — |
| `OPENAI_API_KEY` | OpenAI API key (if using GPT-4o) | — |
| `ANTHROPIC_API_KEY` | Anthropic API key (if using Claude) | — |

### Gmail setup

If using Gmail, you must generate an [App Password](https://support.google.com/accounts/answer/185833) (standard passwords won't work with SMTP).

---

## Running the Agent

```bash
# Set credentials
export SMTP_USER="you@gmail.com"
export SMTP_PASS="your_16char_app_password"
export OPENAI_API_KEY="sk-..."

# Run
python agent.py
```

The agent will start an interactive session in your terminal. Type `quit` or `exit` to end the session.

---

## Switching to Claude

In `agent.py`, replace the model line:

```python
# Before (OpenAI)
from pydantic_ai.models.openai import OpenAIModel
model = OpenAIModel("gpt-4o")

# After (Anthropic)
from pydantic_ai.models.anthropic import AnthropicModel
model = AnthropicModel("claude-sonnet-4-20250514")
```

Then set your Anthropic API key:

```bash
export ANTHROPIC_API_KEY="sk-ant-..."
```

---

## Support Resources

If you or someone you know needs help right now:

- **National Problem Gambling Helpline** — 1-800-522-4700 *(24/7, free, confidential)*
- **National Council on Problem Gambling** — [ncpgambling.org](https://ncpgambling.org)
- **Gamblers Anonymous** — [gamblersanonymous.org](https://www.gamblersanonymous.org)
- **Gam-Anon** (for family & friends) — [gam-anon.org](https://www.gam-anon.org)
- **Crisis Text Line** — Text `HOME` to `741741`

---

## License

MIT
