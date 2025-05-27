import html
import re
import unicodedata

from datasets import load_dataset


def clean_text(text):
    """Clean and normalize text"""
    if not text:
        return ""

    # Remove HTML tags and unescape HTML entities
    text = html.unescape(text)
    text = re.sub(r'<[^>]+>', '', text)

    # Normalize Unicode
    text = unicodedata.normalize('NFKC', text)

    # Normalize Ukrainian apostrophes
    text = re.sub(r'[''`]', "'", text)

    # Remove extra whitespace
    text = re.sub(r'\s+', ' ', text)

    # Convert to lowercase
    text = text.lower().strip()

    return text


def preprocess_dataset():
    """Load and preprocess XLSum Ukrainian dataset"""
    print("Loading XLSum Ukrainian dataset...")
    dataset = load_dataset("csebuetnlp/xlsum", "ukrainian")

    def preprocess_examples(examples):
        """Preprocess batch of examples"""
        cleaned_texts = [clean_text(text) for text in examples["text"]]
        cleaned_summaries = [clean_text(summary) for summary in examples["summary"]]

        return {
            "text": cleaned_texts,
            "summary": cleaned_summaries
        }

    print("🧹 Preprocessing dataset...")
    processed_dataset = dataset.map(
        preprocess_examples,
        batched=True,
        desc="Preprocessing"
    )

    print("Saving preprocessed dataset...")
    processed_dataset.save_to_disk("./preprocessed_xlsum_ukrainian")

    print("Dataset preprocessing completed!")
    print(f"Train samples: {len(processed_dataset['train'])}")
    print(f"Validation samples: {len(processed_dataset['validation'])}")
    print(f"Test samples: {len(processed_dataset['test'])}")

    return processed_dataset


if __name__ == "__main__":
    preprocess_dataset()
