from transformers import AutoTokenizer, AutoModelForSeq2SeqLM
from src.config import config

_cache = {}

def load():
    if "model" not in _cache:
        tokenizer = AutoTokenizer.from_pretrained(config.output_dir)
        model = AutoModelForSeq2SeqLM.from_pretrained(config.output_dir)
        _cache["model"] = (tokenizer, model)
    return _cache["model"]

def summarize(text, mode="paragraph"):
    tokenizer, model = load()
    mode = mode.strip().lower()
    prompt = f"mode: {mode}\ntext: {text}"
    inp = tokenizer(prompt, return_tensors="pt", truncation=True)
    out = model.generate(**inp, max_length=config.max_target_length)
    return tokenizer.decode(out[0], skip_special_tokens=True)
