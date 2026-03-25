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