# Premium payments (Razorpay)

The app can sell **Premium** (more messages) through Razorpay, which supports UPI, cards, net banking and wallets in India. This app never sees card or UPI details: the visitor pays on Razorpay’s own page.

## How the flow works

1. Visitor opens **Upgrade plan** (the **Plans** page), picks **Pro** or **Max**, monthly or yearly, and reaches **Configure your plan**.
2. They enter their name and email (address, business name and GST number are optional), tick the agreement and press **Continue to payment**. The app asks Razorpay to create a payment link, saves the billing details as notes on that payment, and shows **Open secure payment page** plus a **receipt code** (`plink_…`).
3. The visitor pays on Razorpay’s page (a new tab).
4. Back in the app they press **I have paid, activate**. The app asks Razorpay “is this link paid?” with your secret key, and only then unlocks Premium for the number of days you set.
5. After a page refresh they can restore Premium by pasting the receipt code.

## Turn it on in test mode first (no real money)

1. Create an account at razorpay.com and stay in **Test Mode**.
2. Dashboard > Account & Settings > **API Keys** > Generate. Copy the **Key ID** (starts with `rzp_test_`) and the **Key Secret**.
3. Add to your `.env` (never commit this file):
```
RAZORPAY_KEY_ID=rzp_test_xxxxxxxx
RAZORPAY_KEY_SECRET=xxxxxxxxxxxxxxxx
PREMIUM_PRICE_INR=149
PREMIUM_YEARLY_INR=1490
PREMIUM_MESSAGE_LIMIT=300
MAX_PRICE_INR=499
MAX_YEARLY_INR=4990
MAX_MESSAGE_LIMIT=1000
```
The yearly prices above give “Save 17%”. `PREMIUM_DAYS` (default 30) is the length of a monthly plan; yearly is 365 days.

**GST:** only set `GST_RATE=18` if your prices already include GST (you are GST-registered). The checkout then shows “Subtotal” and “Tax (GST 18%)” inside the total, like a tax invoice. With the default `0`, no tax line is shown.
4. Restart the app, open **Upgrade plan**, choose a plan and press **Continue to payment**. On Razorpay’s test page use the test UPI ID or test card from Razorpay’s documentation (“Test card details” and “Test UPI IDs”). The app shows “Test mode: no real money is charged”.
5. Return to the app and press **I have paid, activate**.

**For a portfolio or interview demo, stay in test mode.** Everyone can see the full payment flow and nobody is charged.

## Paying with UPI and PhonePe

You do not integrate PhonePe separately. Razorpay’s payment page already offers **UPI**, and every UPI app works through it:

- **On a phone:** the page lists UPI apps (PhonePe, Google Pay, Paytm, BHIM). Tap PhonePe and approve with your UPI PIN.
- **On a laptop:** the page shows a **QR code** (scan it with PhonePe) or lets you type a **UPI ID** such as `name@ybl` and approve the request in the PhonePe app.

To make sure UPI is available:
1. In Razorpay, open **Settings > Configuration > Payment Methods** and check that **UPI** is switched on.
2. In **Test Mode** it is already on, but no real app is involved: choose UPI on the page and use the test UPI ID from Razorpay’s docs (`success@razorpay` is listed as the success case, `failure@razorpay` as the failure case). PhonePe itself only works in **Live Mode**.
3. In **Live Mode** UPI becomes available once your Razorpay account is activated (KYC done).

**First live test:** set `PREMIUM_PRICE_INR=1`, pay ₹1 from PhonePe, confirm Premium unlocks, then refund the ₹1 from the Razorpay dashboard (Transactions > Payments > Refund) and put the real price back.

PhonePe also has its own separate merchant gateway. You only need it if you do not want Razorpay; this app is built for Razorpay.

## Going live (real money)

- Complete Razorpay KYC (PAN, bank account, business details). Individuals and sole proprietors can usually register, but Razorpay decides.
- Razorpay normally expects public pages for **Terms and Conditions**, **Privacy Policy**, **Refund/Cancellation policy** and a **contact** address on your website.
- Generate **Live** keys (`rzp_live_…`) and set them as environment variables on Render, not in git.
- Razorpay charges a fee per payment (check their current pricing). Set the price so it covers the Gemini cost of 300 messages plus fees.
- Tax rules (income tax, GST) depend on your situation. Ask a qualified person; this is not legal or tax advice.

## What this version does not do yet

- **No user accounts.** The receipt code is the licence, so someone could share it. Premium is per browser session unless the code is pasted again.
- **No recurring subscription.** One payment buys one period (30 days or 365 days) and the checkout says so; there is no auto-renewal. Use Razorpay Subscriptions for that.
- **Billing details** (address, business name, GST number) are saved only as notes on the Razorpay payment, so you can see them in the Razorpay dashboard. The app has no database of its own.
- **No refunds or webhooks handling** inside the app (use the Razorpay dashboard).

## Next step when you want it production-grade

1. Add login with `st.login` (Google).
2. Store each user’s Premium end date in a database (SQLite or Postgres).
3. Receive Razorpay **webhooks** in a small FastAPI service so Premium activates by itself.
4. Use Razorpay **Subscriptions** for monthly renewal.
