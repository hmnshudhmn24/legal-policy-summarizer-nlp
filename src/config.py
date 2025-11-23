from dataclasses import dataclass

@dataclass
class Config:
    model_name: str = "google/flan-t5-base"
    output_dir: str = "model/checkpoints/best-model"
    processed_data_path: str = "data/processed/dataset_clean.jsonl"
    train_batch_size: int = 4
    eval_batch_size: int = 4
    epochs: int = 3
    lr: float = 3e-4
    max_input_length: int = 1024
    max_target_length: int = 256
    seed: int = 42
    device: str = "cuda"

config = Config()
