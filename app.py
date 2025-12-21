import json
import joblib
import pandas as pd
import gradio as gr

# =======================
# Load model artifacts
# =======================
model = joblib.load("price_model.joblib")
range_info = joblib.load("price_range.joblib")

LOW_Q = range_info["low_q"]
HIGH_Q = range_info["high_q"]

with open("year_badge.json", "r") as f:
    YEAR_BADGE = {int(k): v for k, v in json.load(f).items()}

with open("year_spec.json", "r") as f:
    YEAR_SPEC = {int(k): v for k, v in json.load(f).items()}

# =======================
# Static choices
# =======================
BODY_STYLES = ["hatch", "sedan"]
SELLER_TYPES = ["dealer", "private"]
STATES = ["NSW", "VIC", "QLD", "WA", "SA", "ACT", "TAS", "NT"]
YEARS = sorted(YEAR_BADGE.keys())

FEATURE_COLS = [
    "badge", "spec", "body_style",
    "kms", "seller_type", "state", "year"
]

# =======================
# Prediction logic
# =======================
def predict_range_and_judge(
    year, badge, spec, body_style, kms, seller_type, state, user_price
):
    X_row = pd.DataFrame([{
        "badge": badge,
        "spec": spec,
        "body_style": body_style,
        "kms": kms,
        "seller_type": seller_type,
        "state": state,
        "year": int(year)
    }], columns=FEATURE_COLS)

    pred = float(model.predict(X_row)[0])

    lo = pred + LOW_Q
    hi = pred + HIGH_Q

    lo_r = round(lo, -2)
    hi_r = round(hi, -2)

    if lo_r > hi_r:
        lo_r, hi_r = hi_r, lo_r

    est_range = f"${lo_r:,.0f} – ${hi_r:,.0f}"

    label = ""
    if isinstance(user_price, str) and user_price.strip() != "":
        try:
            p = float(user_price.replace(",", "").replace("$", "").strip())
            p_r = round(p, -2)

            if p_r < lo_r:
                label = "Below market price 🟢"
            elif p_r > hi_r:
                label = "Above market price 🔴"
            else:
                label = "Around market price 🟡"
        except Exception:
            label = "Invalid price input"

    return est_range, label

# =======================
# Dynamic dropdown update
# =======================
def update_badge_and_spec(year):
    year = int(year)
    badges = YEAR_BADGE.get(year, [])
    specs = YEAR_SPEC.get(year, [])

    return (
        gr.Dropdown(choices=badges, value=badges[0] if badges else None),
        gr.Dropdown(choices=specs, value=specs[0] if specs else None),
    )

# =======================
# Custom CSS
# =======================
CUSTOM_CSS = """
.gradio-container {
    max-width: 900px;
    margin: auto;
    padding: 32px;
    border-radius: 14px;
    box-shadow: 0 10px 35px rgba(0, 0, 0, 0.18);
}
"""

# =======================
# Gradio UI
# =======================
with gr.Blocks(title="Toyota Corolla Price Estimator (AU)") as demo:

    gr.HTML("<h1>Toyota Corolla Price Estimator (AU) 🚙</h1>")

    gr.HTML("<p>Enter car details to get an estimated valuation. Optionally enter a listed price to flag listing as above, around or below market price.</p>")

    gr.Textbox(label="Transmission", value="Automatic", interactive=False)

    with gr.Row():
        year = gr.Dropdown(label="Year", choices=YEARS, value=YEARS[-1])
        body_style = gr.Dropdown(label="Body style", choices=BODY_STYLES, value="hatch")

    with gr.Row():
        badge = gr.Dropdown(
            label="Badge",
            choices=YEAR_BADGE[YEARS[-1]],
            value=YEAR_BADGE[YEARS[-1]][0]
        )
        spec = gr.Dropdown(
            label="Spec",
            choices=YEAR_SPEC[YEARS[-1]],
            value=YEAR_SPEC[YEARS[-1]][0]
        )

    with gr.Row():
        kms = gr.Number(label="Kilometres", value=50000, minimum=0)
        seller_type = gr.Dropdown(label="Seller type", choices=SELLER_TYPES, value="dealer")
        state = gr.Dropdown(label="State", choices=STATES, value="NSW")

    user_price = gr.Textbox(
        label="Optional: Enter listed price (Excl. Est. Govt. Charges)",
        placeholder="Leave blank to only see estimated range"
    )

    btn = gr.Button("Estimate")

    est_range_out = gr.Textbox(label="Estimated selling range (Excl. Est. Govt. Charges)")
    label_out = gr.Textbox(label="Valuation vs market")

    year.change(fn=update_badge_and_spec, inputs=[year], outputs=[badge, spec])

    btn.click(
        fn=predict_range_and_judge,
        inputs=[year, badge, spec, body_style, kms, seller_type, state, user_price],
        outputs=[est_range_out, label_out]
    )

    gr.HTML(
        """
        <hr style="margin-top: 24px; margin-bottom: 12px;">
        <div style="text-align:center; font-size:13px; font-weight:600; color:#555; line-height:1.4;">
            To understand the assumptions and limitations of this model, please visit the <a href="https://github.com/sri-bandara/Implementing-a-Pricing-Model-for-Toyota-Corolla-Based-off-Listings-Scraped-from-carsales.com.au" target="_blank" style="color:#0366d6; text-decoration:none;">GitHub repository</a>.
        </div>
        """
    )

if __name__ == "__main__":
    demo.launch(css=CUSTOM_CSS)