import torch
from bert_score import score as bert_score
from datasets import load_from_disk
from peft import PeftModel
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM


def generate_summary(model, tokenizer, text, device):
    """Generate summary for a single text"""
    input_text = f"summarize: {text}"
    inputs = tokenizer(
        input_text,
        return_tensors="pt",
        max_length=1024,
        truncation=True,
        padding="max_length"
    ).to(device)

    with torch.no_grad():
        outputs = model.generate(
            input_ids=inputs["input_ids"],
            attention_mask=inputs["attention_mask"],
            max_length=128,
            num_beams=4,
            early_stopping=True,
            no_repeat_ngram_size=2
        )

    summary = tokenizer.decode(outputs[0], skip_special_tokens=True)
    return summary


def main():
    print("Starting Model Evaluation")

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Using device: {device}")

    print("Loading trained model...")
    model_path = "./mt5-ukrainian-summarizer-lora"

    tokenizer = AutoTokenizer.from_pretrained(model_path)
    base_model = AutoModelForSeq2SeqLM.from_pretrained("google/mt5-large")
    model = PeftModel.from_pretrained(base_model, model_path)
    model.to(device)
    model.eval()

    print("Loading test dataset...")
    dataset = load_from_disk("./preprocessed_xlsum_ukrainian")
    test_dataset = dataset["test"]

    print("Generating summaries...")
    predictions = []
    references = []

    for i, example in enumerate(test_dataset):
        if i % 10 == 0:
            print(f"Progress: {i}/{len(test_dataset)}")

        pred = generate_summary(model, tokenizer, example["text"], device)
        predictions.append(pred)
        references.append(example["summary"])

    print("Computing ROUGE scores...")
    rouge = evaluate.load("rouge")
    rouge_result = rouge.compute(predictions=predictions, references=references)

    print("Computing BERTScore...")
    P, R, F1 = bert_score(
        cands=predictions,
        refs=references,
        lang="uk",
        verbose=False
    )

    bertscore_result = {
        'precision': P.mean().item(),
        'recall': R.mean().item(),
        'f1': F1.mean().item()
    }

    print("\n" + "=" * 50)
    print("EVALUATION RESULTS")
    print("=" * 50)

    print("\nROUGE Scores:")
    print(f"ROUGE-1: {rouge_result['rouge1']:.4f}")
    print(f"ROUGE-2: {rouge_result['rouge2']:.4f}")
    print(f"ROUGE-L: {rouge_result['rougeL']:.4f}")

    print("\nBERTScore:")
    print(f"Precision: {bertscore_result['precision']:.4f}")
    print(f"Recall: {bertscore_result['recall']:.4f}")
    print(f"F1: {bertscore_result['f1']:.4f}")

    results = {
        'rouge': rouge_result,
        'bertscore': bertscore_result,
        'sample_predictions': predictions[:5],
        'sample_references': references[:5]
    }

    import json
    with open("evaluation_results.json", "w", encoding="utf-8") as f:
        json.dump(results, f, ensure_ascii=False, indent=2)

    print("\nResults saved to evaluation_results.json")
    print("Evaluation completed!")


if __name__ == "__main__":
    main()
