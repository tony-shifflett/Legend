import torch
import yaml
import os
from transformers import pipeline

class LegendKeeper:
    def __init__(self, model_id="google/gemma-3-4b-it"):
        print(f"--- 🕯️ Awakening the Legend Keeper ({model_id}) ---")
        
        # 1. Load the Persona Config from your YAML
        self.config_path = os.path.join("configs", "prompts.yaml")
        self.config = self._load_config()
        
        # 2. Initialize Conversation History
        # We start empty; the first 'chat' call will inject the system prompt
        self.history = []
        
        # 3. Initialize the 2026 AI Pipeline
        # We use 4-bit quantization to ensure it fits on Colab's T4 GPU
        self.pipe = pipeline(
            "text-generation",
            model=model_id,
            model_kwargs={
                "torch_dtype": torch.bfloat16,
                "load_in_4bit": True,
                "device_map": "auto"
            }
        )

    def _load_config(self):
        """Helper to read the YAML file."""
        with open(self.config_path, 'r') as file:
            return yaml.safe_load(file)

    def generate_backstory(self, name, char_class, race, vibe):
        """
        Initializes a NEW story. This clears history and 
        starts the collaboration.
        """
        self.reset_chat()
        
        # Format the guarded system prompt from YAML
        raw_system_content = self.config['personas']['backstory']
        system_content = raw_system_content.format(
            name=name, 
            char_class=char_class, 
            race=race, 
            vibe=vibe
        )
        
        # Start the history with the System Persona
        self.history.append({"role": "system", "content": system_content})
        
        # Trigger the first response
        return self.chat("Begin the chronicle of my character.")

    def chat(self, user_input):
        """
        Main interactive loop. Maintains memory of the conversation.
        """
        # Add the user's new message/question to memory
        self.history.append({"role": "user", "content": user_input})

        # Generate response using the ENTIRE history
        # Gemma 3 automatically uses its chat template via the pipeline
        outputs = self.pipe(
            self.history, 
            max_new_tokens=350, 
            do_sample=True, 
            temperature=0.8,
            top_p=0.9
        )
        
        # Extract only the newly generated message
        ai_msg = outputs[0]['generated_text'][-1]['content']
        
        # Save AI's response to history so it remembers for the next turn
        self.history.append({"role": "assistant", "content": ai_msg})
        
        return ai_msg

    def reset_chat(self):
        """Clears the history for a fresh start."""
        self.history = []
        print("--- The Grimoire has been reset. ---")