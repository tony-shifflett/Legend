graph TD
    subgraph "Persistent Storage (The Shelf)"
        A[(ChromaDB: './chroma_db')]
        style A fill:#e1f5fe,stroke:#01579b,stroke-width:3px
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
        D -.X|NOT SAVED| A
    end

    E --> Interface([Gradio UI])

    style D fill:#f9f,stroke:#333,stroke-width:2px
    style B fill:#bbf,stroke:#333,stroke-width:2px