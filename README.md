# 🏛️ Legal / Policy Text Summarizer NLP

An advanced Transformer-based NLP model that simplifies **legal, governmental, and policy documents** into three easy-to-understand summary formats:

### 🔹 3-line summary
### 🔹 1-paragraph summary
### 🔹 Bullet points (3–7 bullets)

This project includes the full ML pipeline: preprocessing, PDF extraction, dataset creation, training, evaluation, inference, FastAPI deployment, Gradio UI, tests, and a HuggingFace model card.

## 🚀 Features
✔ Summarizes long policies, laws, govt documents  
✔ Output styles: 3line, paragraph, bullets  
✔ Full training/evaluation pipeline  
✔ Works with PDFs  
✔ Built on `google/flan-t5-base`  
✔ Apache 2.0 licensed  
✔ HuggingFace-ready metadata  

## 📁 Project Structure
```
legal-policy-summarizer-nlp/
├── data/
├── src/
├── tests/
├── app/
├── notebooks/
├── huggingface/
├── model/
├── README.md
├── LICENSE
└── requirements.txt
```

## 📦 Installation
```
python -m venv venv
source venv/bin/activate
pip install -r requirements.txt
```

## 📄 Dataset Preprocessing
```
python -m src.dataset_preprocessing --input data/raw/dataset.csv --output data/processed/dataset_clean.jsonl
```

## 🏋️ Training
```
python -m src.train
```

## 🧪 Evaluation
```
python -m src.evaluate
```

## 🤖 Inference
```python
from src.inference import summarize
print(summarize("policy text...", mode="paragraph"))
```

## 🌐 API (FastAPI)
```
uvicorn app.api:app --reload --port 8000
```

## 🎨 Gradio UI
```
python app/ui.py
```

## ⚖️ License
Apache License 2.0.

