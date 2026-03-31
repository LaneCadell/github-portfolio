"""
Gambling Addiction Support Email Agent
Built with Pydantic AI + smtplib

Install deps:
    pip install pydantic-ai pydantic

Usage:
    python agent.py
"""

import smtplib
import os
from email.mime.text import MIMEText
from email.mime.multipart import MIMEMultipart
from dataclasses import dataclass

from pydantic import BaseModel, EmailStr
from pydantic_ai import Agent, RunContext
from pydantic_ai.models.openai import OpenAIModel  # or AnthropicModel


# ─── Config ────────────────────────────────────────────────────────────────────

SMTP_HOST = os.getenv("SMTP_HOST", "smtp.gmail.com")
SMTP_PORT = int(os.getenv("SMTP_PORT", "587"))
SMTP_USER = os.getenv("SMTP_USER", "your_email@gmail.com")
SMTP_PASS = os.getenv("SMTP_PASS", "your_app_password")

SUPPORT_EMAIL = "support@ncpgambling.org"  # National Council on Problem Gambling


# ─── Dependencies (injected into tools via context) ────────────────────────────

@dataclass
class EmailDeps:
    smtp_host: str
    smtp_port: int
    smtp_user: str
    smtp_pass: str
    support_email: str


# ─── Structured output from agent ──────────────────────────────────────────────

class SupportEmailResult(BaseModel):
    sent: bool
    recipient: str
    subject: str
    summary: str  # short plain-language confirmation for the user


# ─── Agent definition ──────────────────────────────────────────────────────────

model = OpenAIModel("gpt-4o")  # swap for AnthropicModel("claude-sonnet-4-20250514")

agent = Agent(
    model=model,
    deps_type=EmailDeps,
    result_type=SupportEmailResult,
    system_prompt="""
    You are a compassionate support coordinator helping people reach resources
    for gambling addiction. Your job is to:

    1. Collect the user's name, email address, and a brief message describing
       their situation (ask for each if not provided).
    2. Compose a warm, empathetic support request email on their behalf.
    3. Use the `send_support_email` tool to send it.
    4. Return a SupportEmailResult confirming what was sent.

    Always be non-judgmental, supportive, and remind the user that help is
    available. Never minimise their situation.
    Resources to mention when relevant:
      - National Problem Gambling Helpline: 1-800-522-4700 (24/7)
      - National Council on Problem Gambling: ncpgambling.org
      - Gamblers Anonymous: gamblersanonymous.org
    """,
)


# ─── Tool: send email ──────────────────────────────────────────────────────────

@agent.tool
def send_support_email(
    ctx: RunContext[EmailDeps],
    sender_name: str,
    sender_email: str,
    user_message: str,
) -> dict:
    """
    Sends a support request email to the gambling addiction helpline on behalf
    of the user.

    Args:
        sender_name:   Full name of the person reaching out.
        sender_email:  Their email address (so support can reply).
        user_message:  The body of their support request.

    Returns:
        A dict with 'success' (bool) and 'error' (str | None).
    """
    deps = ctx.deps

    subject = f"Support Request from {sender_name}"

    body = f"""
Dear Support Team,

My name is {sender_name} and I am reaching out because I need help with a
gambling problem. Below is my message:

---
{user_message}
---

Please reply to me at: {sender_email}

Thank you for your time and support.

Warm regards,
{sender_name}

---
This email was sent via the Gambling Support Agent.
Resources:
  • National Problem Gambling Helpline: 1-800-522-4700
  • ncpgambling.org  |  gamblersanonymous.org
""".strip()

    msg = MIMEMultipart("alternative")
    msg["Subject"] = subject
    msg["From"]    = deps.smtp_user
    msg["To"]      = deps.support_email
    msg["Reply-To"] = sender_email

    msg.attach(MIMEText(body, "plain"))

    try:
        with smtplib.SMTP(deps.smtp_host, deps.smtp_port) as server:
            server.ehlo()
            server.starttls()
            server.login(deps.smtp_user, deps.smtp_pass)
            server.sendmail(deps.smtp_user, deps.support_email, msg.as_string())
        return {"success": True, "error": None}
    except Exception as e:
        return {"success": False, "error": str(e)}


# ─── Tool: get resources ───────────────────────────────────────────────────────

@agent.tool_plain
def get_support_resources() -> str:
    """Returns a formatted list of gambling addiction support resources."""
    return """
Key Support Resources:
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
🔹 National Problem Gambling Helpline
   📞 1-800-522-4700  (24/7, free, confidential)

🔹 National Council on Problem Gambling
   🌐 ncpgambling.org

🔹 Gamblers Anonymous
   🌐 gamblersanonymous.org  (peer support meetings)

🔹 Gam-Anon (for family/friends)
   🌐 gam-anon.org

🔹 Crisis Text Line
   💬 Text HOME to 741741
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
""".strip()


# ─── Main conversation loop ────────────────────────────────────────────────────

def main():
    deps = EmailDeps(
        smtp_host=SMTP_HOST,
        smtp_port=SMTP_PORT,
        smtp_user=SMTP_USER,
        smtp_pass=SMTP_PASS,
        support_email=SUPPORT_EMAIL,
    )

    print("=" * 60)
    print("  Gambling Addiction Support Agent")
    print("  Type 'quit' to exit at any time.")
    print("  National Helpline: 1-800-522-4700  (24/7)")
    print("=" * 60)
    print()

    message_history = []

    while True:
        user_input = input("You: ").strip()
        if not user_input:
            continue
        if user_input.lower() in ("quit", "exit"):
            print("\nTake care. Help is always available at 1-800-522-4700.\n")
            break

        result = agent.run_sync(
            user_input,
            deps=deps,
            message_history=message_history,
        )

        # Persist conversation history for multi-turn context
        message_history = result.all_messages()

        output = result.data
        print(f"\nAgent: {output.summary}")

        if output.sent:
            print(f"\n✅  Email sent to {output.recipient}")
            print(f"    Subject: {output.subject}\n")

        print()


if __name__ == "__main__":
    main()
