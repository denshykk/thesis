import gradio as gr
import torch
from peft import PeftModel
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM, pipeline

MODEL_PATH = "src/mt5-ukrainian-summarizer-lora"
BASE_MODEL_PATH = "google/mt5-large"


def load_model():
    """Load the trained model"""
    print("Loading model...")

    tokenizer = AutoTokenizer.from_pretrained(MODEL_PATH)

    base_model = AutoModelForSeq2SeqLM.from_pretrained(BASE_MODEL_PATH)
    model = PeftModel.from_pretrained(base_model, MODEL_PATH)

    ppln = pipeline(
        "summarization",
        model=model,
        tokenizer=tokenizer,
        device=0 if torch.cuda.is_available() else -1,
        num_beams=4,
        no_repeat_ngram_size=2,
        clean_up_tokenization_spaces=True,
    )

    print("Model loaded successfully!")
    return ppln, tokenizer


ppln, tokenizer = load_model()


def summarize_gradio(text, min_len, max_len):
    """Summary function with parameters"""
    if not text.strip():
        return "Будь ласка, введіть текст для реферування."

    if len(text.strip()) < 50:
        return "Текст занадто короткий. Введіть принаймні 50 символів."

    try:
        result = ppln(
            text,
            min_length=min_len,
            max_length=max_len,
            do_sample=False
        )

        summary = result[0]["summary_text"]

        input_words = len(text.split())
        summary_words = len(summary.split())
        compression_ratio = round(input_words / summary_words, 1) if summary_words > 0 else 0

        stats = f"\n\nСтатистика:\n• Вхідний текст: {input_words} слів\n• Реферат: {summary_words} слів\n• Коефіцієнт стиснення: {compression_ratio}:1"

        return summary + stats

    except Exception as e:
        return f"Помилка: {str(e)}"


chat_ui = gr.Interface(
    fn=summarize_gradio,
    inputs=[
        gr.Textbox(
            label="Enter Ukrainian text to summarize",
            lines=5,
            placeholder="Введіть текст тут..."
        ),
        gr.Slider(
            minimum=32,
            maximum=256,
            value=64,
            step=8,
            label="Minimum Summary Length"
        ),
        gr.Slider(
            minimum=64,
            maximum=512,
            value=256,
            step=8,
            label="Maximum Summary Length"
        ),
    ],
    outputs=gr.Textbox(label="Summary"),
    title="Ukrainian Summarizer (mT5 + LoRA)",
    description="A lightweight summarization model fine-tuned with LoRA on XLSum. Use sliders to control summary length."
)

if __name__ == "__main__":
    print("Launching Gradio interface...")
    chat_ui.launch(share=True)
