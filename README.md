## Legend Keeper

Legend Keeper is a D&D-focused backstory and lore assistant. It combines a local/vector lore search layer with an instruction-tuned language model so responses stay immersive, in-character, and grounded in campaign context.

### What this project does

- Generates character backstories from simple inputs (name, class, race, vibe).
- Retrieves relevant lore snippets from a persistent ChromaDB collection.
- Keeps a narrative persona ("The Legend Keeper") with strict guardrails.
- Supports follow-up chat while preserving story continuity.

### How the code works

#### 1) `LoreLibrarian` handles retrieval

- File: `src/librarian.py`
- `LoreLibrarian` connects to a persistent ChromaDB path (`data/lore_db` by default, or a user-provided path).
- It binds to collection `dnd_lore` (or falls back to the only available collection).
- `search()` runs semantic lookup and returns top lore passages as plain text context.

#### 2) `LegendKeeper` handles generation

- File: `src/generator.py`
- `LegendKeeper` loads persona rules from `configs/prompts.yaml`.
- It initializes a 4-bit quantized text-generation pipeline (`transformers` + `bitsandbytes`) for efficient local/Colab inference.
- `generate_backstory()` starts a new guarded story session and injects retrieved lore.
- `chat()` appends user input, injects context each turn, generates with continuation safeguards, and trims incomplete endings.

#### 3) Prompt configuration controls behavior

- File: `configs/prompts.yaml`
- Centralizes identity, safety constraints, formatting, tone, and interaction rules.
- Keeps behavior editable without changing Python logic.

### Current dependencies (high level)

- LLM/runtime: `transformers`, `torch`, `accelerate`, `bitsandbytes`
- Retrieval/DB: `chromadb`, `sentence-transformers`
- Supporting tools: `pyyaml`, `python-dotenv`, and LangChain modular packages

### AI tooling disclosure

This project used AI tools during development (including coding assistants and LLM support) to help draft, refactor, and document code. Final project structure, validation, and behavioral decisions were reviewed and curated by the maintainer.

### System architecture

```mermaid
%%{init: {'theme':'base','themeVariables':{
    'fontFamily':'Segoe UI, Arial, sans-serif',
    'textColor':'#111827',
    'lineColor':'#334155',
    'primaryTextColor':'#111827',
    'primaryBorderColor':'#1f2937'
}}}%%
graph TD
    subgraph "Persistent Storage (The Shelf)"
        A[(ChromaDB: './chroma_db')]
    style A fill:#dbeafe,stroke:#1e3a8a,stroke-width:3px,color:#0b1f44
    end

    subgraph "Phase 1: Knowledge Retrieval"
        User((User Query)) --> B{"Embedding Model <br/>(all-MiniLM-L6-v2)"}
        B -->|Search Vector| A
        A -->|Official 5e Snippets| C[Prompt Construction]
    end

    subgraph "Phase 2: The Legend Keeper"
        User --> C
        C --> D{{"Gemma 3 Flash <br/>(Reasoning Engine)"}}
        D --> E[In-Character Answer]
    end

    subgraph "The Current Limitation"
        F[New Character/Homebrew] -.->|Context Only| D
        D -.->|NOT SAVED| A
    end

    E --> Interface([Gradio UI])

    style User fill:#f3f4f6,stroke:#111827,stroke-width:2px,color:#111827
    style B fill:#0f766e,stroke:#134e4a,stroke-width:2px,color:#ffffff
    style C fill:#fef3c7,stroke:#92400e,stroke-width:2px,color:#1f2937
    style D fill:#4c1d95,stroke:#2e1065,stroke-width:2px,color:#ffffff
    style E fill:#dcfce7,stroke:#166534,stroke-width:2px,color:#14532d
    style F fill:#fee2e2,stroke:#991b1b,stroke-width:2px,color:#7f1d1d
    style Interface fill:#e0f2fe,stroke:#0c4a6e,stroke-width:2px,color:#082f49
    linkStyle default stroke:#334155,stroke-width:2px,color:#111827
```