import os
os.environ["NO_PROXY"] = "localhost,127.0.0.1,0.0.0.0"
os.environ["no_proxy"] = "localhost,127.0.0.1,0.0.0.0"

import csv
from datetime import datetime

import torch
import gradio as gr
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM

# --- Model ---
model_checkpoint = "./finetuned-en-sw-final"
tokenizer = AutoTokenizer.from_pretrained(model_checkpoint)
model = AutoModelForSeq2SeqLM.from_pretrained(model_checkpoint)

FLAG_FILE = "flagged_translations.csv"


def translate(english_text):
    if not english_text.strip():
        return ""
    inputs = tokenizer(english_text, return_tensors="pt", truncation=True, max_length=512)
    with torch.no_grad():
        ids = model.generate(**inputs, max_length=512, num_beams=4)
    return tokenizer.decode(ids[0], skip_special_tokens=True)


def flag_translation(english_text, swahili_text, reason):
    """Save a flagged (input, output, reason) triple to a CSV for later fine-tuning."""
    if not english_text.strip():
        return "⚠️ Translate something first, then flag it."
    if not swahili_text.strip():
        return "⚠️ No translation to flag — click Translate first."

    file_exists = os.path.exists(FLAG_FILE)
    with open(FLAG_FILE, "a", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        if not file_exists:
            writer.writerow(["timestamp", "english", "swahili", "reason"])
        writer.writerow([
            datetime.now().isoformat(timespec="seconds"),
            english_text.strip(),
            swahili_text.strip(),
            reason,
        ])
    return f"✅ Flagged as **{reason}** — saved to `{FLAG_FILE}`. Asante!"


# --- Branding CSS ---
CUSTOM_CSS = """
.gradio-container {
    max-width: 1000px !important;
    margin: auto !important;
    font-family: 'Segoe UI', Tahoma, sans-serif !important;
}

/* Centered gradient title */
#app-title h1 {
    background: linear-gradient(90deg, #0284c7 0%, #06b6d4 50%, #14b8a6 100%);
    -webkit-background-clip: text;
    -webkit-text-fill-color: transparent;
    font-weight: 800;
    font-size: 2rem;
    text-align: center;
    margin: 0.5rem 0 1.25rem 0 !important;
}

/* Gradient translate button */
#translate-btn {
    background: linear-gradient(90deg, #0284c7, #06b6d4) !important;
    border: none !important;
    color: white !important;
    font-weight: 600 !important;
    box-shadow: 0 4px 14px rgba(6, 182, 212, 0.30) !important;
    transition: transform 0.15s ease, box-shadow 0.15s ease !important;
}
#translate-btn:hover {
    transform: translateY(-1px);
    box-shadow: 0 6px 20px rgba(6, 182, 212, 0.45) !important;
}

/* Flag button — purple to stand out from the blue Translate button */
#flag-btn {
    background: linear-gradient(90deg, #7c3aed, #a855f7) !important;
    border: none !important;
    color: white !important;
    font-weight: 600 !important;
    box-shadow: 0 4px 14px rgba(124, 58, 237, 0.30) !important;
    transition: transform 0.15s ease, box-shadow 0.15s ease !important;
}
#flag-btn:hover {
    transform: translateY(-1px);
    box-shadow: 0 6px 20px rgba(124, 58, 237, 0.45) !important;
}

/* Textbox rounding */
#english-box, #swahili-box {
    border-radius: 12px !important;
}

/* Custom footer */
#app-footer {
    text-align: center;
    color: #64748b;
    font-size: 0.9rem;
    margin-top: 1.5rem;
    padding-top: 1rem;
    border-top: 1px solid #e2e8f0;
}
#app-footer p {
    margin: 0 !important;
}

/* Hide the default Gradio footer */
footer { display: none !important; }
"""


# --- App layout ---
with gr.Blocks(title="English → Swahili Translator") as demo:

    gr.Markdown("# English → Swahili Translator", elem_id="app-title")

    # Side-by-side input/output
    with gr.Row():
        with gr.Column():
            english_input = gr.Textbox(
                label="English",
                placeholder="Type English here...",
                lines=5,
                elem_id="english-box",
            )
        with gr.Column():
            swahili_output = gr.Textbox(
                label="Swahili",
                lines=5,
                interactive=False,
                elem_id="swahili-box",
            )

    # Translate / Clear
    with gr.Row():
        translate_btn = gr.Button("Translate ➡️", variant="primary",
                                  elem_id="translate-btn", scale=2)
        clear_btn = gr.Button("Clear", variant="secondary", scale=1)

    # --- Flag section for fine-tuning data collection ---
    gr.Markdown("---")
    gr.Markdown(
        "### 🚩 Flag a bad translation\n"
        "Help improve the model. Flagged examples are saved to "
        "`flagged_translations.csv` for future fine-tuning."
    )
    with gr.Row():
        flag_reason = gr.Dropdown(
            choices=["Bad translation", "Wrong language", "Incorrect grammar",
                     "Unnatural phrasing", "Missing content", "Other"],
            value="Bad translation",
            label="Reason",
            scale=2,
        )
        flag_btn = gr.Button("Flag for fine-tuning", elem_id="flag-btn", scale=1)

    flag_status = gr.Markdown("")

    # --- Footer ---
    gr.Markdown(
        "Created by **Wendy Okoth**",
        elem_id="app-footer",
    )

    # --- Wiring ---
    translate_btn.click(fn=translate, inputs=english_input, outputs=swahili_output)
    english_input.submit(fn=translate, inputs=english_input, outputs=swahili_output)
    clear_btn.click(fn=lambda: ("", ""), inputs=None,
                    outputs=[english_input, swahili_output])

    flag_btn.click(
        fn=flag_translation,
        inputs=[english_input, swahili_output, flag_reason],
        outputs=flag_status,
    )


# --- Launch (Gradio 6.0: theme + css go here) ---
demo.launch(
    server_name="127.0.0.1",
    theme=gr.themes.Ocean(),
    css=CUSTOM_CSS,
)