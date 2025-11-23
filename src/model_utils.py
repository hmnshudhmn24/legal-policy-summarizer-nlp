from transformers import AutoTokenizer, AutoModelForSeq2SeqLM

def load_model_and_tokenizer(path):
    tok = AutoTokenizer.from_pretrained(path)
    mod = AutoModelForSeq2SeqLM.from_pretrained(path)
    return tok, mod
