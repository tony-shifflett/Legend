import torch
import yaml
import os
from transformers import pipeline, BitsAndBytesConfig  # <--- Added BitsAndBytesConfig

class LegendKeeper:
    def __init__(self, model_id="google/gemma-3-4b-it"):
        print(f"--- 🕯️ Awakening the Legend Keeper ({model_id}) ---")
        
        self.config_path = os.path.join("configs", "prompts.yaml")
        self.config = self._load_config()
        self.history = []
        
        # 1. Define the 4-bit configuration explicitly
        quantization_config = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_compute_dtype=torch.bfloat16,
            bnb_4bit_quant_type="nf4",
            bnb_4bit_use_double_quant=True
        )

        # 2. Initialize the pipeline with the config object
        self.pipe = pipeline(
            "text-generation",
            model=model_id,
            model_kwargs={
                "quantization_config": quantization_config, # <--- Pass the object here
                "device_map": "auto"
            }
        )