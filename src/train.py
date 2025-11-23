import random, os, torch
from pathlib import Path
from datasets import load_dataset
from transformers import (AutoTokenizer, AutoModelForSeq2SeqLM,
                          Seq2SeqTrainer, Seq2SeqTrainingArguments,
                          DataCollatorForSeq2Seq)
from src.config import config

def set_seed(seed):
    random.seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)
    import numpy as np
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

def preprocess(examples, tokenizer):
    inputs = tokenizer(examples["input"], max_length=config.max_input_length, truncation=True)
    with tokenizer.as_target_tokenizer():
        labels = tokenizer(examples["target"], max_length=config.max_target_length, truncation=True)
    inputs["labels"] = labels["input_ids"]
    return inputs

def main():
    set_seed(config.seed)

    ds = load_dataset("json", data_files=config.processed_data_path)["train"]
    ds = ds.train_test_split(test_size=0.1, seed=config.seed)

    tokenizer = AutoTokenizer.from_pretrained(config.model_name)
    model = AutoModelForSeq2SeqLM.from_pretrained(config.model_name)

    train_ds = ds["train"].map(lambda x: preprocess(x, tokenizer), batched=True)
    eval_ds = ds["test"].map(lambda x: preprocess(x, tokenizer), batched=True)

    args = Seq2SeqTrainingArguments(
        output_dir=config.output_dir,
        evaluation_strategy="epoch",
        per_device_train_batch_size=config.train_batch_size,
        per_device_eval_batch_size=config.eval_batch_size,
        num_train_epochs=config.epochs,
        learning_rate=config.lr,
        predict_with_generate=True,
        save_total_limit=2,
        remove_unused_columns=False,
        fp16=torch.cuda.is_available()
    )

    data_collator = DataCollatorForSeq2Seq(tokenizer, model)

    trainer = Seq2SeqTrainer(
        model=model,
        args=args,
        train_dataset=train_ds,
        eval_dataset=eval_ds,
        tokenizer=tokenizer,
        data_collator=data_collator,
    )

    trainer.train()
    trainer.save_model(config.output_dir)
    tokenizer.save_pretrained(config.output_dir)

if __name__ == "__main__":
    main()
