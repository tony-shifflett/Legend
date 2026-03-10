Project Title: The Legend Keeper’s Grimoire
Phase: Iteration 1 (The "Crawl")



1. Problem Statement
In the world of Tabletop Role-Playing Games (TTRPGs), players and Dungeon Masters (DMs) often face "Creative Burnout" and "The Blank Page Syndrome." Creating a character involves a difficult balance between complex mechanical statistics and compelling narrative backstories.

Current AI tools are often too generalized, providing generic "fantasy" advice that lacks the specific tone of a dedicated D&D campaign or the mechanical accuracy required by seasoned players. There is a need for a specialized, persona-driven tool that streamlines the creative process while maintaining high-quality, thematic consistency.

2. Importance
The Legend Keeper’s Grimoire serves as a creative force-multiplier for the TTRPG community.

For Players: It transforms a few basic "vibes" or ideas into a rich, playable history, increasing emotional investment in their characters.

For Dungeon Masters: It allows for the rapid generation of "Lore-Heavy" NPCs (Non-Player Characters) to populate a world on short notice.

For the Industry: It demonstrates how Large Language Models (LLMs) can be safely and effectively constrained to a specific domain (Fantasy/D&D) using specialized prompting and architectural guardrails.

3. Tools and Technology
To ensure this project is modern and "job-ready," it utilizes the following 2026 AI Tech Stack:

Core Model: Gemma 3 (4B-Instruct) — An open-source, high-efficiency LLM from Google, capable of complex reasoning within a fantasy context.

Library: Hugging Face Transformers — For model loading, 4-bit quantization (to optimize Colab GPU usage), and inference.

Orchestration: LangChain — To structure the interaction between user inputs and the model output.

Interface: Google Colab — Acts as the interactive frontend, providing free access to T4 GPU compute for model execution.

Version Control: GitHub — Used to host the modular logic (src/), ensuring the project follows professional software engineering standards.

4. Current Scope (Iteration 1: Legendary Origins)
The first iteration focuses on the core "Backstory Engine":

User Inputs: Name, Race, Class, and "Vibe" (e.g., "Grumpy but protective").

AI Persona: The "Legend Keeper," a consistent narrative voice that generates immersive, under-200-word backstories.

Modular Codebase: A clean separation between the user interface (Colab) and the generation logic (GitHub).

5. Future Goals
The Grimoire is built to evolve from a storyteller into a full campaign assistant:

Phase 2: The "Walk" (RAG Integration)
Retrieval-Augmented Generation (RAG): Integrating ChromaDB to allow the AI to reference specific PDF rulebooks or campaign setting guides.

Mechanical Accuracy: Ensuring character stats and abilities align perfectly with official D&D 5e rules.

Phase 3: The "Run" (Campaign Architect)
Campaign Management: Expanding the engine to generate interconnected quest hooks, town descriptions, and historical lore.

Persistence: Implementing a SQLite database to save and export generated characters for long-term use.

Mobile Access: Transitioning the logic into a FastAPI backend to support a mobile UI (e.g., Flutter).