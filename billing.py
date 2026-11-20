"""Premium payments with Razorpay Payment Links (works in test mode and live mode).

How it works
1. The app asks Razorpay to create a payment link for the Premium price.
2. The visitor pays on Razorpay's own secure page (UPI, cards, net banking, wallets).
3. The app asks Razorpay "is this link paid?" using your secret key, and only then unlocks Premium.

The secret key stays in environment variables on the server. Nothing here trusts the browser.
"""

from __future__ import annotations

import os
import time
import uuid

import requests

API = "https://api.razorpay.com/v1"


class BillingError(Exception):
    """An error whose message is safe to show to the visitor."""


def _creds() -> tuple[str, str]:
    return os.getenv("RAZORPAY_KEY_ID", "").strip(), os.getenv("RAZORPAY_KEY_SECRET", "").strip()


def configured() -> bool:
    key_id, secret = _creds()
    return bool(key_id and secret)


def is_test() -> bool:
    return _creds()[0].startswith("rzp_test_")


def _int(name: str, default: int) -> int:
    try:
        return int(os.getenv(name, str(default)))
    except ValueError:
        return default


def gst_rate() -> float:
    """GST percentage already included in the prices (0 means the prices show no tax)."""
    try:
        return max(0.0, float(os.getenv("GST_RATE", "0")))
    except ValueError:
        return 0.0


def plans() -> dict:
    """Paid plans. Prices are whole rupees and include tax when GST_RATE is set."""
    return {
        "pro": {
            "name": "Pro",
            "tagline": "Research, summarise and organise",
            "limit": _int("PREMIUM_MESSAGE_LIMIT", 300),
            "monthly": _int("PREMIUM_PRICE_INR", 149),
            "yearly": _int("PREMIUM_YEARLY_INR", 1490),
        },
        "max": {
            "name": "Max",
            "tagline": "Higher limits for heavy use",
            "limit": _int("MAX_MESSAGE_LIMIT", 1000),
            "monthly": _int("MAX_PRICE_INR", 499),
            "yearly": _int("MAX_YEARLY_INR", 4990),
        },
    }


def period_days(period: str) -> int:
    return 365 if period == "yearly" else _int("PREMIUM_DAYS", 30)


def price_for(plan: str, period: str) -> int:
    return plans()[plan]["yearly" if period == "yearly" else "monthly"]


def savings_percent(plan: str) -> int:
    p = plans()[plan]
    return max(0, round((1 - p["yearly"] / (p["monthly"] * 12)) * 100))


def tax_split(total: int) -> tuple[float, float]:
    """(price before tax, tax) for a tax-inclusive total."""
    rate = gst_rate()
    if rate <= 0:
        return float(total), 0.0
    before = round(total / (1 + rate / 100), 2)
    return before, round(total - before, 2)


# kept for older callers: the Pro monthly price and length
def price() -> int:
    return price_for("pro", "monthly")


def days() -> int:
    return period_days("monthly")


def _call(method: str, path: str, **kwargs) -> dict:
    key_id, secret = _creds()
    try:
        response = requests.request(method, API + path, auth=(key_id, secret), timeout=15, **kwargs)
    except requests.RequestException:
        raise BillingError("Could not reach the payment service. Check your internet connection and try again.")
    if response.status_code == 401:
        raise BillingError("The payment keys were rejected. Check RAZORPAY_KEY_ID and RAZORPAY_KEY_SECRET.")
    if response.status_code in (400, 404) and method == "GET":
        raise BillingError("No payment found for that code.")
    if not response.ok:
        raise BillingError(f"The payment service returned an error ({response.status_code}). Please try again.")
    return response.json()


def create_payment_link(plan: str = "pro", period: str = "monthly", name: str = "", email: str = "", extra=None) -> dict:
    """Returns {"id": receipt code, "short_url": page to pay on}. `extra` is saved as notes on the payment."""
    if plan not in plans() or period not in ("monthly", "yearly"):
        raise BillingError("That plan is not available.")
    notes = {"plan": plan, "period": period}
    for key, value in (extra or {}).items():
        value = str(value).strip()
        if value:
            notes[str(key)] = value[:250]
    body: dict = {
        "amount": price_for(plan, period) * 100,  # Razorpay counts in paise
        "currency": "INR",
        "accept_partial": False,
        "description": f"PDF RAG Chatbot {plans()[plan]['name']} ({period_days(period)} days)",
        "reference_id": f"prm_{int(time.time())}_{uuid.uuid4().hex[:6]}",
        "reminder_enable": False,
        "notes": notes,
    }
    customer = {k: v.strip() for k, v in (("name", name), ("email", email)) if v and v.strip()}
    if customer:
        body["customer"] = customer
        if "email" in customer:
            body["notify"] = {"email": True, "sms": False}
    data = _call("POST", "/payment_links", json=body)
    return {"id": data["id"], "short_url": data["short_url"]}


def fetch_link(link_id: str) -> dict:
    link_id = (link_id or "").strip()
    if not link_id.startswith("plink_"):
        raise BillingError("That does not look like a receipt code. It starts with plink_.")
    return _call("GET", f"/payment_links/{link_id}")


def activation(link: dict):
    """{"plan", "period", "until"} for a paid link, or None if it has not been paid."""
    if link.get("status") != "paid":
        return None
    notes = link.get("notes")
    notes = notes if isinstance(notes, dict) else {}  # Razorpay sends [] when there are no notes
    plan = notes.get("plan") if notes.get("plan") in plans() else "pro"
    period = "yearly" if notes.get("period") == "yearly" else "monthly"
    started = float(link.get("created_at") or time.time())
    return {"plan": plan, "period": period, "until": started + period_days(period) * 86400}


def premium_until(link: dict):
    """When Premium ends for this payment link, or None if it has not been paid."""
    info = activation(link)
    return info["until"] if info else None
