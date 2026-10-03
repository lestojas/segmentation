"""Notification channels. Every configured channel is tried; one failing never blocks the others."""
import os
import platform
import smtplib
import subprocess
import sys
from email.message import EmailMessage

import requests


def _log(msg):
    print(f"[notify] {msg}", file=sys.stderr)


def _ntfy(title, body, url):
    topic = os.environ.get("NTFY_TOPIC")
    if not topic:
        return False
    headers = {"Title": title.encode("ascii", "ignore").decode(), "Priority": "high", "Tags": "passport_control"}
    if url:
        headers["Click"] = url
    requests.post(f"https://ntfy.sh/{topic}", data=body.encode("utf-8"), headers=headers, timeout=20).raise_for_status()
    return True


def _email(title, body, url):
    user, pw = os.environ.get("GMAIL_USER"), os.environ.get("GMAIL_APP_PASSWORD")
    if not (user and pw):
        return False
    msg = EmailMessage()
    msg["Subject"], msg["From"], msg["To"] = title, user, os.environ.get("EMAIL_TO") or user
    msg.set_content(body + (f"\n\n{url}" if url else ""))
    with smtplib.SMTP_SSL("smtp.gmail.com", 465, timeout=30) as s:
        s.login(user, pw.replace(" ", ""))
        s.send_message(msg)
    return True


def _telegram(title, body, url):
    token, chat = os.environ.get("TELEGRAM_BOT_TOKEN"), os.environ.get("TELEGRAM_CHAT_ID")
    if not (token and chat):
        return False
    text = f"{title}\n\n{body}" + (f"\n\n{url}" if url else "")
    requests.post(f"https://api.telegram.org/bot{token}/sendMessage",
                  data={"chat_id": chat, "text": text}, timeout=20).raise_for_status()
    return True


def _sms(title, body, url):
    key, to = os.environ.get("SEMAPHORE_API_KEY"), os.environ.get("SMS_TO")
    if not (key and to):
        return False
    data = {"apikey": key, "number": to, "message": f"{title}\n{body}"[:450]}
    if os.environ.get("SEMAPHORE_SENDER"):
        data["sendername"] = os.environ["SEMAPHORE_SENDER"]
    requests.post("https://api.semaphore.co/api/v4/messages", data=data, timeout=20).raise_for_status()
    return True


def _mac(title, body, url):
    if platform.system() != "Darwin" or os.environ.get("MAC_NOTIFY", "1") != "1":
        return False
    esc = lambda s: s.replace("\\", "\\\\").replace('"', '\\"')
    subprocess.run(["osascript", "-e",
                    f'display notification "{esc(body[:200])}" with title "{esc(title)}" sound name "Glass"'],
                   check=True, timeout=10)
    return True


CHANNELS = [("ntfy", _ntfy), ("email", _email), ("telegram", _telegram), ("sms", _sms), ("mac", _mac)]


def send_all(title, body, url=None):
    """Returns the list of channels that delivered successfully."""
    ok = []
    for name, fn in CHANNELS:
        try:
            if fn(title, body, url):
                ok.append(name)
        except Exception as e:  # noqa: BLE001 - never let one channel kill the run
            _log(f"{name} failed: {e}")
    if not ok:
        _log("WARNING: no notification channel delivered. Check your .env")
    return ok
