"""
AI EMERGENCY INTELLIGENCE — Humanitarian Information Triage & Classification
==============================================================================
UI-only redesign. The model, tokenizer, and inference logic are UNCHANGED
from the original app.py. Only the Gradio interface layer has been rebuilt
to look like an operational crisis-information console instead of a
default Gradio demo.

Run:
    python app.py

Requirements (same as before, nothing new added):
    pip install gradio torch transformers
"""

import torch
import gradio as gr
from transformers import AutoTokenizer, AutoModelForSequenceClassification

# ------------------------------------------------------------------
# 1. MODEL LOADING — real fine-tuned model, loaded at runtime.
#    No placeholder, no mock weights. This must be run from the
#    project directory that contains ./emergency_model.
# ------------------------------------------------------------------
MODEL_PATH = "./emergency_model"

tokenizer = AutoTokenizer.from_pretrained(MODEL_PATH)
model = AutoModelForSequenceClassification.from_pretrained(MODEL_PATH)
model.eval()

# HumAID / QCRI event-type labels, in the model's id2label order.
# NOTE: this list is only used for the "Category Reference" panel text.
# The actual predicted label always comes from model.config.id2label,
# so if your fine-tuned label set differs, predictions stay correct.
CATEGORY_DESCRIPTIONS = {
    "caution_and_advice": "Warnings, safety guidance, or preventive instructions shared with the public.",
    "displaced_people_and_evacuations": "Reports of people relocating, sheltering, or being evacuated.",
    "infrastructure_and_utility_damage": "Damage to roads, buildings, power, water, or other infrastructure.",
    "injured_or_dead_people": "Reports referencing casualties, injuries, or fatalities.",
    "missing_or_found_people": "Reports of missing persons or people who have been located.",
    "not_humanitarian": "Text unrelated to humanitarian or emergency response needs.",
    "other_relevant_information": "Relevant emergency context that doesn't fit other categories.",
    "requests_or_urgent_needs": "Explicit requests for help, supplies, or urgent assistance.",
    "rescue_volunteering_or_donation_effort": "Mentions of rescue operations, volunteering, or donations.",
    "sympathy_and_support": "Expressions of sympathy, solidarity, or emotional support.",
}

EXAMPLE_MESSAGES = [
    "Flood waters have entered the ground floor of the community center, several families are trapped on the roof and need immediate rescue.",
    "Please avoid Route 9 near the river crossing, the bridge has structural damage and is not safe to cross.",
    "We are collecting blankets and bottled water for evacuees at the downtown shelter, drop-off open until 8pm.",
    "My neighbor has not been seen since the earthquake, last known location was near the old market street.",
]

# ------------------------------------------------------------------
# 2. INFERENCE (unchanged logic, just returns richer structured data)
# ------------------------------------------------------------------
def classify_incident(text: str):
    """
    Runs the fine-tuned DistilBERT model on the input text.
    Returns:
        - status message
        - top label + confidence (for the headline result)
        - interpretation sentence
        - top-3 (label, probability) pairs for the bar chart
        - raw dict for the summary table
    """
    if not text or not text.strip():
        empty_msg = "⚠ No input provided. Enter an incident report to analyze."
        return (
            empty_msg,
            "—",
            "—",
            "Awaiting input before classification can run.",
            gr.update(value=None),
            "—",
            "—",
            "AWAITING INPUT",
        )

    inputs = tokenizer(
        text, return_tensors="pt", truncation=True, padding=True, max_length=128
    )

    with torch.no_grad():
        logits = model(**inputs).logits
        probs = torch.softmax(logits, dim=-1)[0]

    id2label = model.config.id2label
    ranked = sorted(
        ((id2label[i], float(probs[i])) for i in range(len(probs))),
        key=lambda x: x[1],
        reverse=True,
    )

    top_label, top_conf = ranked[0]
    top3 = ranked[:3]

    readable_label = top_label.replace("_", " ").title()
    interpretation = (
        f"The input most closely matches the pattern of "
        f"'{readable_label}' reports, with the model {top_conf * 100:.1f}% "
        f"confident in this classification based on learned language patterns "
        f"from the HumAID training data."
    )

    # Bar-chart-friendly structure: {label: probability}
    chart_data = {lbl.replace("_", " ").title(): prob for lbl, prob in top3}

    status_msg = "✅ ANALYSIS COMPLETE"

    return (
        status_msg,
        readable_label,
        f"{top_conf * 100:.1f}%",
        interpretation,
        gr.update(value=chart_data),
        readable_label,
        f"{top_conf * 100:.1f}%",
        "COMPLETE",
    )


# ------------------------------------------------------------------
# 3. CUSTOM CSS — dark command-center visual system
# ------------------------------------------------------------------
CUSTOM_CSS = """
:root {
    --bg-void:      #0a0e14;
    --bg-panel:     #10161f;
    --bg-panel-alt: #141b26;
    --border-line:  #22303f;
    --text-primary: #e6edf3;
    --text-muted:   #7c8b9c;
    --accent-amber: #e8a33d;
    --accent-red:   #d6543f;
    --accent-teal:  #3fb6a8;
    --status-good:  #3fb68a;
}

.gradio-container {
    background: var(--bg-void) !important;
    color: var(--text-primary) !important;
    font-family: 'Inter', 'Segoe UI', system-ui, sans-serif !important;
}

/* ---------- Header ---------- */
#header-row {
    border-bottom: 1px solid var(--border-line);
    padding-bottom: 14px;
    margin-bottom: 18px;
}
#system-title {
    font-size: 1.5rem;
    font-weight: 700;
    letter-spacing: 0.02em;
    color: var(--text-primary);
    margin: 0;
}
#system-subtitle {
    color: var(--text-muted);
    font-size: 0.92rem;
    margin-top: 2px;
}
#status-chip {
    border: 1px solid var(--status-good);
    color: var(--status-good);
    border-radius: 4px;
    padding: 6px 12px;
    font-size: 0.82rem;
    font-weight: 600;
    text-align: center;
    background: rgba(63, 182, 138, 0.08);
}
#model-chip {
    border: 1px solid var(--border-line);
    color: var(--text-muted);
    border-radius: 4px;
    padding: 6px 12px;
    font-size: 0.78rem;
    text-align: center;
    margin-top: 6px;
}

/* ---------- Panels ---------- */
.op-panel {
    background: var(--bg-panel) !important;
    border: 1px solid var(--border-line) !important;
    border-radius: 6px !important;
    padding: 16px !important;
}
.panel-label {
    color: var(--text-muted);
    font-size: 0.78rem;
    font-weight: 600;
    letter-spacing: 0.06em;
    margin-bottom: 8px;
    border-left: 2px solid var(--accent-teal);
    padding-left: 8px;
}

/* ---------- Input ---------- */
#incident-input textarea {
    background: var(--bg-panel-alt) !important;
    border: 1px solid var(--border-line) !important;
    color: var(--text-primary) !important;
    font-size: 0.95rem !important;
}

#analyze-btn {
    background: var(--accent-red) !important;
    color: #fff !important;
    font-weight: 700 !important;
    border: none !important;
    letter-spacing: 0.03em;
}
#analyze-btn:hover {
    background: #c04430 !important;
}

/* ---------- Result readouts ---------- */
.result-value {
    font-size: 1.4rem;
    font-weight: 700;
    color: var(--accent-amber);
}
.result-sub {
    color: var(--text-muted);
    font-size: 0.85rem;
}

/* ---------- Summary table ---------- */
#summary-table table {
    font-size: 0.88rem !important;
}

/* ---------- Category reference ---------- */
.cat-item {
    border-bottom: 1px solid var(--border-line);
    padding: 6px 0;
    font-size: 0.85rem;
}
.cat-name {
    color: var(--accent-teal);
    font-weight: 600;
}
.cat-desc {
    color: var(--text-muted);
}

/* ---------- Footer ---------- */
#footer-note {
    text-align: center;
    color: var(--text-muted);
    font-size: 0.78rem;
    border-top: 1px solid var(--border-line);
    padding-top: 12px;
    margin-top: 20px;
}
"""

# ------------------------------------------------------------------
# 4. UI LAYOUT
# ------------------------------------------------------------------
with gr.Blocks(css=CUSTOM_CSS, title="AI Emergency Intelligence") as demo:

    # ---- HEADER ----
    with gr.Row(elem_id="header-row"):
        with gr.Column(scale=4):
            gr.Markdown(
                '<p id="system-title">AI EMERGENCY INTELLIGENCE</p>'
                '<p id="system-subtitle">Humanitarian Information Triage &amp; Classification</p>'
            )
        with gr.Column(scale=1, min_width=160):
            gr.Markdown('<div id="status-chip">● MODEL ONLINE</div>')
            gr.Markdown('<div id="model-chip">ENGINE: DistilBERT (fine-tuned)</div>')

    # ---- MAIN INPUT PANEL ----
    with gr.Group(elem_classes="op-panel"):
        gr.Markdown('<div class="panel-label">INCIDENT REPORT INPUT</div>')
        incident_text = gr.Textbox(
            elem_id="incident-input",
            label=None,
            placeholder="Paste or type an incident report, social media message, or field report to classify...",
            lines=6,
            show_label=False,
        )
        with gr.Row():
            gr.Examples(examples=EXAMPLE_MESSAGES, inputs=incident_text, label="Example incident messages")
        analyze_btn = gr.Button("▶ ANALYZE INCIDENT", elem_id="analyze-btn", size="lg")

    status_line = gr.Markdown("STATUS: STANDING BY")

    # ---- ANALYSIS RESULT + INTELLIGENCE SUMMARY ----
    with gr.Row():
        with gr.Column(scale=1):
            with gr.Group(elem_classes="op-panel"):
                gr.Markdown('<div class="panel-label">ANALYSIS RESULT</div>')
                pred_label = gr.Markdown('<span class="result-value">—</span>')
                pred_conf = gr.Markdown('<span class="result-sub">Confidence: —</span>')
                interpretation = gr.Markdown("Run an analysis to see interpretation here.")

        with gr.Column(scale=1):
            with gr.Group(elem_classes="op-panel"):
                gr.Markdown('<div class="panel-label">TOP 3 PREDICTIONS</div>')
                top3_chart = gr.Label(num_top_classes=3, label=None, show_label=False)

    # ---- INTELLIGENCE SUMMARY TABLE ----
    with gr.Group(elem_classes="op-panel"):
        gr.Markdown('<div class="panel-label">INTELLIGENCE SUMMARY</div>')
        summary_table = gr.Dataframe(
            headers=["Field", "Value"],
            value=[
                ["Classification", "—"],
                ["Confidence", "—"],
                ["Analysis Status", "AWAITING INPUT"],
                ["Model Used", "DistilBERT (fine-tuned, HumAID event-type)"],
            ],
            interactive=False,
            elem_id="summary-table",
        )

    # ---- CATEGORY REFERENCE ----
    with gr.Group(elem_classes="op-panel"):
        gr.Markdown('<div class="panel-label">CATEGORY REFERENCE — 10 SUPPORTED CLASSES</div>')
        cat_md = "\n".join(
            f'<div class="cat-item"><span class="cat-name">{name.replace("_", " ").title()}</span><br>'
            f'<span class="cat-desc">{desc}</span></div>'
            for name, desc in CATEGORY_DESCRIPTIONS.items()
        )
        gr.Markdown(cat_md)

    # ---- FOOTER ----
    gr.Markdown(
        '<div id="footer-note">AI-assisted humanitarian information classification &bull; Decision-support prototype</div>'
    )

    # ---- WIRE UP EVENT ----
    def run_analysis(text):
        (
            status_msg,
            label,
            conf,
            interp,
            chart_update,
            summary_label,
            summary_conf,
            summary_status,
        ) = classify_incident(text)

        new_table = [
            ["Classification", summary_label],
            ["Confidence", summary_conf],
            ["Analysis Status", summary_status],
            ["Model Used", "DistilBERT (fine-tuned, HumAID event-type)"],
        ]

        return (
            f"STATUS: {status_msg}",
            f'<span class="result-value">{label}</span>',
            f'<span class="result-sub">Confidence: {conf}</span>',
            interp,
            chart_update,
            new_table,
        )

    analyze_btn.click(
        fn=run_analysis,
        inputs=incident_text,
        outputs=[status_line, pred_label, pred_conf, interpretation, top3_chart, summary_table],
    )

# ------------------------------------------------------------------
# 5. LAUNCH
# ------------------------------------------------------------------
if __name__ == "__main__":
    demo.launch()