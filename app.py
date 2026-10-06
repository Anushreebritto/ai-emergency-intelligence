"""
AI EMERGENCY INTELLIGENCE
Humanitarian Information Triage & Classification

ONNX Runtime deployment version.
The model is loaded from Hugging Face.

Run locally:
    python app.py
"""

import os
import numpy as np
import gradio as gr

from transformers import AutoTokenizer
from optimum.onnxruntime import ORTModelForSequenceClassification


# ------------------------------------------------------------------
# 1. MODEL LOADING
# ------------------------------------------------------------------

MODEL_PATH = "Anushreebritto2/emergency-intelligence-distilbert"
MODEL_SUBFOLDER = "optimized_model"

print("Loading ONNX model from Hugging Face...")

tokenizer = AutoTokenizer.from_pretrained(
    MODEL_PATH,
    subfolder=MODEL_SUBFOLDER
)

model = ORTModelForSequenceClassification.from_pretrained(
    MODEL_PATH,
    subfolder=MODEL_SUBFOLDER
)

print("ONNX model loaded successfully!")


# ------------------------------------------------------------------
# 2. CATEGORY DESCRIPTIONS
# ------------------------------------------------------------------

CATEGORY_DESCRIPTIONS = {
    "caution_and_advice":
        "Warnings, safety guidance, or preventive instructions shared with the public.",

    "displaced_people_and_evacuations":
        "Reports of people relocating, sheltering, or being evacuated.",

    "infrastructure_and_utility_damage":
        "Damage to roads, buildings, power, water, or other infrastructure.",

    "injured_or_dead_people":
        "Reports referencing casualties, injuries, or fatalities.",

    "missing_or_found_people":
        "Reports of missing persons or people who have been located.",

    "not_humanitarian":
        "Text unrelated to humanitarian or emergency response needs.",

    "other_relevant_information":
        "Relevant emergency context that doesn't fit other categories.",

    "requests_or_urgent_needs":
        "Explicit requests for help, supplies, or urgent assistance.",

    "rescue_volunteering_or_donation_effort":
        "Mentions of rescue operations, volunteering, or donations.",

    "sympathy_and_support":
        "Expressions of sympathy, solidarity, or emotional support.",
}


# ------------------------------------------------------------------
# 3. EXAMPLE INCIDENTS
# ------------------------------------------------------------------

EXAMPLE_MESSAGES = [
    "Flood waters have entered the ground floor of the community center, several families are trapped on the roof and need immediate rescue.",

    "Please avoid Route 9 near the river crossing, the bridge has structural damage and is not safe to cross.",

    "We are collecting blankets and bottled water for evacuees at the downtown shelter, drop-off open until 8pm.",

    "My neighbor has not been seen since the earthquake, last known location was near the old market street.",
]


# ------------------------------------------------------------------
# 4. SOFTMAX
# ------------------------------------------------------------------

def softmax(logits):
    logits = np.asarray(logits, dtype=np.float32)
    logits = logits - np.max(logits)
    probabilities = np.exp(logits)
    return probabilities / np.sum(probabilities)


# ------------------------------------------------------------------
# 5. INFERENCE
# ------------------------------------------------------------------

def classify_incident(text: str):

    if not text or not text.strip():

        empty_msg = (
            "⚠ No input provided. Enter an incident report to analyze."
        )

        return (
            empty_msg,
            "—",
            "—",
            "Awaiting input before classification can run.",
            "Awaiting input before top-3 predictions can run.",
            "—",
            "—",
            "AWAITING INPUT",
        )

    # Tokenize input
    inputs = tokenizer(
        text,
        return_tensors="np",
        truncation=True,
        padding=True,
        max_length=128,
    )

    # ONNX Runtime inference
    outputs = model(**inputs)

    logits = outputs.logits

    # Convert logits to probabilities
    probs = softmax(logits[0])

    # Model label mapping
    id2label = model.config.id2label

    # Rank predictions
    ranked = sorted(
        (
            (
                id2label[i],
                float(probs[i])
            )
            for i in range(len(probs))
        ),
        key=lambda x: x[1],
        reverse=True,
    )

    # Top prediction
    top_label, top_conf = ranked[0]

    # Top 3
    top3 = ranked[:3]

    readable_label = top_label.replace("_", " ").title()

    interpretation = (
        f"The input most closely matches the pattern of "
        f"'{readable_label}' reports, with the model "
        f"{top_conf * 100:.1f}% confident in this classification "
        f"based on learned language patterns from the HumAID training data."
    )

    top3_lines = [
        f"**{i}. {lbl.replace('_', ' ').title()}** — {prob * 100:.1f}%"
        for i, (lbl, prob) in enumerate(top3, start=1)
    ]

    top3_text = "\n\n".join(top3_lines)

    status_msg = "✅ ANALYSIS COMPLETE"

    return (
        status_msg,
        readable_label,
        f"{top_conf * 100:.1f}%",
        interpretation,
        top3_text,
        readable_label,
        f"{top_conf * 100:.1f}%",
        "COMPLETE",
    )


# ------------------------------------------------------------------
# 6. CUSTOM CSS
# ------------------------------------------------------------------

CUSTOM_CSS = """
:root {
    --bg-void: #0a0e14;
    --bg-panel: #10161f;
    --bg-panel-alt: #141b26;
    --border-line: #22303f;
    --text-primary: #e6edf3;
    --text-muted: #7c8b9c;
    --accent-amber: #e8a33d;
    --accent-red: #d6543f;
    --accent-teal: #3fb6a8;
    --status-good: #3fb68a;
}

.gradio-container {
    background: var(--bg-void) !important;
    color: var(--text-primary) !important;
    font-family: 'Inter', 'Segoe UI', system-ui, sans-serif !important;
}

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

.result-value {
    font-size: 1.4rem;
    font-weight: 700;
    color: var(--accent-amber);
}

.result-sub {
    color: var(--text-muted);
    font-size: 0.85rem;
}

#summary-table table {
    font-size: 0.88rem !important;
}

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
# 7. UI
# ------------------------------------------------------------------

with gr.Blocks(
    css=CUSTOM_CSS,
    title="AI Emergency Intelligence"
) as demo:

    # Header
    with gr.Row(elem_id="header-row"):

        with gr.Column(scale=4):

            gr.Markdown(
                '<p id="system-title">AI EMERGENCY INTELLIGENCE</p>'
                '<p id="system-subtitle">'
                'Humanitarian Information Triage &amp; Classification'
                '</p>'
            )

        with gr.Column(scale=1, min_width=160):

            gr.Markdown(
                '<div id="status-chip">● MODEL ONLINE</div>'
            )

            gr.Markdown(
                '<div id="model-chip">'
                'ENGINE: DistilBERT (ONNX Runtime)'
                '</div>'
            )


    # Input
    with gr.Group(elem_classes="op-panel"):

        gr.Markdown(
            '<div class="panel-label">INCIDENT REPORT INPUT</div>'
        )

        incident_text = gr.Textbox(
            elem_id="incident-input",
            label=None,
            placeholder=(
                "Paste or type an incident report, social media message, "
                "or field report to classify..."
            ),
            lines=6,
            show_label=False,
        )

        with gr.Row():

            gr.Examples(
                examples=EXAMPLE_MESSAGES,
                inputs=incident_text,
                label="Example incident messages"
            )

        analyze_btn = gr.Button(
            "▶ ANALYZE INCIDENT",
            elem_id="analyze-btn",
            size="lg"
        )

    status_line = gr.Markdown("STATUS: STANDING BY")


    # Results
    with gr.Row():

        with gr.Column(scale=1):

            with gr.Group(elem_classes="op-panel"):

                gr.Markdown(
                    '<div class="panel-label">ANALYSIS RESULT</div>'
                )

                pred_label = gr.Markdown(
                    '<span class="result-value">—</span>'
                )

                pred_conf = gr.Markdown(
                    '<span class="result-sub">Confidence: —</span>'
                )

                interpretation = gr.Markdown(
                    "Run an analysis to see interpretation here."
                )


        with gr.Column(scale=1):

            with gr.Group(elem_classes="op-panel"):

                gr.Markdown(
                    '<div class="panel-label">TOP 3 PREDICTIONS</div>'
                )

                top3_chart = gr.Markdown(
                    "Run an analysis to see ranked predictions here."
                )


    # Intelligence summary
    with gr.Group(elem_classes="op-panel"):

        gr.Markdown(
            '<div class="panel-label">INTELLIGENCE SUMMARY</div>'
        )

        summary_table = gr.Dataframe(
            headers=["Field", "Value"],
            value=[
                ["Classification", "—"],
                ["Confidence", "—"],
                ["Analysis Status", "AWAITING INPUT"],
                [
                    "Model Used",
                    "DistilBERT (fine-tuned, HumAID event-type)"
                ],
            ],
            interactive=False,
            elem_id="summary-table",
        )


    # Category reference
    with gr.Group(elem_classes="op-panel"):

        gr.Markdown(
            '<div class="panel-label">'
            'CATEGORY REFERENCE — 10 SUPPORTED CLASSES'
            '</div>'
        )

        cat_md = "\n".join(
            f'<div class="cat-item">'
            f'<span class="cat-name">'
            f'{name.replace("_", " ").title()}'
            f'</span><br>'
            f'<span class="cat-desc">{desc}</span>'
            f'</div>'
            for name, desc in CATEGORY_DESCRIPTIONS.items()
        )

        gr.Markdown(cat_md)


    # Footer
    gr.Markdown(
        '<div id="footer-note">'
        'AI-assisted humanitarian information classification '
        '&bull; Decision-support prototype'
        '</div>'
    )


    # Event handler
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
            [
                "Model Used",
                "DistilBERT (fine-tuned, HumAID event-type)"
            ],
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
        outputs=[
            status_line,
            pred_label,
            pred_conf,
            interpretation,
            top3_chart,
            summary_table,
        ],
    )


# ------------------------------------------------------------------
# 8. LAUNCH
# ------------------------------------------------------------------

if __name__ == "__main__":

    port = int(os.environ.get("PORT", 7860))

    demo.launch(
        server_name="0.0.0.0",
        server_port=port
    )