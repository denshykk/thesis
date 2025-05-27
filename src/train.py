from datasets import load_from_disk
from peft import LoraConfig, get_peft_model, TaskType
from transformers import (
    AutoTokenizer,
    AutoModelForSeq2SeqLM,
    TrainingArguments,
    Trainer,
    DataCollatorForSeq2Seq
)


def preprocess_function(examples, tokenizer):
    """Preprocess examples for training"""
    model_inputs = tokenizer(
        [f"summarize: {text}" for text in examples["text"]],
        max_length=1024,
        truncation=True,
        padding="max_length"
    )

    targets = tokenizer(
        text_target=examples["summary"],
        max_length=128,
        truncation=True,
        padding="max_length"
    )

    model_inputs["labels"] = targets["input_ids"]
    return model_inputs


def main():
    print("Starting Ukrainian Summarization Training")

    print("Loading preprocessed dataset...")
    dataset = load_from_disk("./preprocessed_xlsum_ukrainian")
    train_dataset = dataset["train"]
    val_dataset = dataset["validation"]

    print(f"Training samples: {len(train_dataset)}")
    print(f"Validation samples: {len(val_dataset)}")

    model_name = "google/mt5-large"
    print(f"Loading model: {model_name}")

    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModelForSeq2SeqLM.from_pretrained(model_name)

    print("Configuring LoRA...")
    peft_config = LoraConfig(
        r=8,
        lora_alpha=16,
        target_modules=["q", "v"],
        lora_dropout=0.1,
        bias="none",
        task_type=TaskType.SEQ_2_SEQ_LM
    )

    model = get_peft_model(model, peft_config)
    model.print_trainable_parameters()

    print("Preprocessing datasets...")
    tokenized_train = train_dataset.map(
        lambda x: preprocess_function(x, tokenizer),
        batched=True,
        remove_columns=train_dataset.column_names
    )

    tokenized_val = val_dataset.map(
        lambda x: preprocess_function(x, tokenizer),
        batched=True,
        remove_columns=val_dataset.column_names
    )

    training_args = TrainingArguments(
        output_dir="./mt5-ukrainian-summarizer-lora",
        per_device_train_batch_size=4,
        per_device_eval_batch_size=4,
        num_train_epochs=3,
        learning_rate=5e-4,
        eval_strategy="epoch",
        save_strategy="epoch",
        logging_steps=50,
        fp16=False,
        save_total_limit=2,
        report_to="none"
    )

    data_collator = DataCollatorForSeq2Seq(tokenizer, model=model)

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=tokenized_train,
        eval_dataset=tokenized_val,
        tokenizer=tokenizer,
        data_collator=data_collator
    )

    print("Starting training...")
    trainer.train()

    print("Saving model...")
    trainer.save_model()
    tokenizer.save_pretrained("./mt5-ukrainian-summarizer-lora")

    print("Training completed successfully!")


if __name__ == "__main__":
    main()
