import torch
import yaml
import os
from transformers import pipeline, BitsAndBytesConfig  # <--- Essential Import

class LegendKeeper:
    def __init__(self, model_id="google/gemma-3-4b-it"):
        print(f"--- 🕯️ Awakening the Legend Keeper ({model_id}) ---")
        
        # 1. Load the Persona Config
        self.config_path = os.path.join("configs", "prompts.yaml")
        self.config = self._load_config()
        self.history = []
        
        # 2. Quantization Logic (The "Secret Sauce" for Colab)
        # This belongs here because it's part of the 'Engine' setup
        quantization_config = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_compute_dtype=torch.bfloat16,
            bnb_4bit_quant_type="nf4",
            bnb_4bit_use_double_quant=True
        )

        # 3. Model Pipeline
        self.pipe = pipeline(
            "text-generation",
            model=model_id,
            model_kwargs={
                "quantization_config": quantization_config,
                "device_map": "auto"
            }
        )

    def _load_config(self):
        """Helper to read the YAML file."""
        with open(self.config_path, 'r') as file:
            return yaml.safe_load(file)

    def generate_backstory(self, name, char_class, race, vibe):
        """Initializes a NEW story and clears history."""
        self.reset_chat()
        
        raw_prompt = self.config['personas']['backstory']
        system_content = raw_prompt.format(
            name=name, 
            char_class=char_class, 
            race=race, 
            vibe=vibe
        )
        
        # In Gemma 3, we treat the persona as the first 'system' message
        self.history.append({"role": "system", "content": system_content})
        
        return self.chat("Begin the chronicle of my character.")

    def chat(self, user_input):
        """Main interactive loop with memory."""
        self.history.append({"role": "user", "content": user_input})

        outputs = self.pipe(
            self.history, 
            max_new_tokens=350, 
            do_sample=True, 
            temperature=0.8,
            top_p=0.9
        )
        
        ai_msg = outputs[0]['generated_text'][-1]['content']
        self.history.append({"role": "assistant", "content": ai_msg})
        
        return ai_msg

    def reset_chat(self):
        """Clears the history."""
        self.history = []
        print("--- The Grimoire has been reset. ---")