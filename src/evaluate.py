from datasets import load_dataset, load_metric
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM
from src.config import config

def evaluate():
    ds = load_dataset("json", data_files=config.processed_data_path)["train"]
    ds = ds.train_test_split(test_size=0.1)["test"]

    tokenizer = AutoTokenizer.from_pretrained(config.output_dir)
    model = AutoModelForSeq2SeqLM.from_pretrained(config.output_dir)

    rouge = load_metric("rouge")

    preds, refs = [], []
    for sample in ds:
        inp = tokenizer(sample["input"], return_tensors="pt", truncation=True)
        out = model.generate(**inp, max_length=config.max_target_length)
        pred = tokenizer.decode(out[0], skip_special_tokens=True)
        preds.append(pred)
        refs.append(sample["target"])

    result = rouge.compute(predictions=preds, references=refs)
    print(result)

if __name__ == "__main__":
    evaluate()
