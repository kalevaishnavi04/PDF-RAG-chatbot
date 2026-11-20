"""Run with:  python -m pytest -q   (or: python tests/test_billing.py). No network needed."""
import os
import sys
import time

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import billing  # noqa: E402


class FakeResponse:
    def __init__(self, status, data):
        self.status_code, self._data, self.ok = status, data, 200 <= status < 300

    def json(self):
        return self._data


def setup(status=200, data=None, error=None):
    os.environ.update({"RAZORPAY_KEY_ID": "rzp_test_x", "RAZORPAY_KEY_SECRET": "s", "PREMIUM_PRICE_INR": "149", "PREMIUM_DAYS": "30"})
    seen = {}

    def fake(method, url, auth=None, timeout=None, json=None, **kw):
        seen.update(method=method, url=url, auth=auth, json=json)
        if error:
            raise error
        return FakeResponse(status, data)

    billing.requests.request = fake
    return seen


def test_configuration_flags():
    setup()
    assert billing.configured() and billing.is_test() and billing.price() == 149 and billing.days() == 30
    os.environ["RAZORPAY_KEY_ID"] = "rzp_live_x"
    assert not billing.is_test()
    os.environ["RAZORPAY_KEY_ID"] = ""
    assert not billing.configured()


def test_create_link_sends_paise_customer_and_notes():
    seen = setup(data={"id": "plink_1", "short_url": "https://rzp.io/i/1"})
    link = billing.create_payment_link("pro", "yearly", "Asha", "a@b.com", {"gstin": "22AAAAA0000A1Z5", "address": ""})
    assert link == {"id": "plink_1", "short_url": "https://rzp.io/i/1"}
    body = seen["json"]
    assert body["amount"] == 149000 and body["currency"] == "INR"
    assert body["customer"] == {"name": "Asha", "email": "a@b.com"} and body["notify"]["email"] is True
    assert body["notes"] == {"plan": "pro", "period": "yearly", "gstin": "22AAAAA0000A1Z5"}  # empty values are dropped
    assert seen["auth"] == ("rzp_test_x", "s") and seen["url"].endswith("/payment_links")
    billing.create_payment_link("max", "monthly")
    assert seen["json"]["amount"] == 49900 and "customer" not in seen["json"]
    try:
        billing.create_payment_link("gold", "monthly")
    except billing.BillingError:
        pass
    else:
        raise AssertionError("unknown plan must be refused")


def test_plans_prices_and_savings():
    setup()
    plans = billing.plans()
    assert plans["pro"]["monthly"] == 149 and plans["pro"]["yearly"] == 1490 and plans["max"]["monthly"] == 499
    assert billing.savings_percent("pro") == 17 and billing.savings_percent("max") == 17
    assert billing.period_days("yearly") == 365 and billing.period_days("monthly") == 30


def test_tax_split_matches_a_gst_inclusive_price():
    os.environ["GST_RATE"] = "18"
    assert billing.tax_split(2399) == (2033.05, 365.95)
    os.environ["GST_RATE"] = "0"
    assert billing.tax_split(149) == (149.0, 0.0)


def test_activation_reads_plan_and_period_from_notes():
    setup()
    now = time.time()
    assert billing.activation({"status": "created"}) is None
    info = billing.activation({"status": "paid", "created_at": now, "notes": {"plan": "max", "period": "yearly"}})
    assert info["plan"] == "max" and abs(info["until"] - (now + 365 * 86400)) < 1
    fallback = billing.activation({"status": "paid", "created_at": now, "notes": []})  # Razorpay sends [] for no notes
    assert fallback["plan"] == "pro" and fallback["period"] == "monthly"
    assert billing.activation({"status": "paid", "created_at": now, "notes": {"plan": "gold"}})["plan"] == "pro"


def test_friendly_errors():
    for kwargs, text in (
        ({"status": 401, "data": {}}, "keys were rejected"),
        ({"status": 500, "data": {}}, "(500)"),
    ):
        setup(**kwargs)
        try:
            billing.fetch_link("plink_1")
        except billing.BillingError as exc:
            assert text in str(exc)
        else:
            raise AssertionError("expected BillingError")
    setup(status=400, data={})
    try:
        billing.fetch_link("plink_nope")
    except billing.BillingError as exc:
        assert "No payment found" in str(exc)
    try:
        billing.fetch_link("abc")
    except billing.BillingError as exc:
        assert "plink_" in str(exc)
    setup(error=billing.requests.ConnectionError("down"))
    try:
        billing.fetch_link("plink_1")
    except billing.BillingError as exc:
        assert "Could not reach" in str(exc)


if __name__ == "__main__":
    tests = [v for k, v in sorted(globals().items()) if k.startswith("test_")]
    for t in tests:
        t()
    print(f"{len(tests)} billing tests passed")
