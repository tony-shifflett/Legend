# Master System Architecture v2

## Audit Notes

1. The repository does not currently contain a `main.py` or any Gradio source file. The UI layer in the diagram is modeled as the intended interface boundary described by the README and by your requested architecture.
2. The implemented persistent ChromaDB path in `LoreLibrarian` defaults to `data/lore_db`, not `./chroma_db`. The diagram reflects the implemented path and calls out the mismatch.
3. The repository currently contains two executable logic modules: `src/generator.py` for orchestration and `src/librarian.py` for retrieval.
4. The README describes a Gradio UI, while the proposal describes a Colab frontend. The diagram prioritizes the implemented Python runtime and marks the UI as an external presentation layer.

## Mermaid

```mermaid
flowchart TB
    %% Master system model built from src/generator.py and src/librarian.py

    subgraph UI[UI Layer - Gradio Boundary (modeled, not present in repo)]
        U1[Player]
        U2[main.py / Gradio Form<br/>captures Name, Race, Class, Vibe]
        U3[Chat Input<br/>follow-up prompt]
        U4[Chronicle Output<br/>rendered backstory or reply]
    end

    subgraph ORCH[Orchestration Layer - LegendKeeper Class]
        O1[LegendKeeper.__init__<br/>load prompts.yaml<br/>init LoreLibrarian<br/>init Gemma pipeline]
        O2[generate_backstory(name, class, race, vibe)]
        O3[reset_chat()] 
        O4[Build system prompt<br/>persona + lore context + guardrails]
        O5[self.history<br/>session JSON messages]
        O6[chat(user_input)]
        O7[working_history = list(self.history)]
        O8[Generation loop<br/>attempt <= max_continuations]
        O9[_extract_assistant_content(outputs)]
        O10[assembled response buffer]
        O11{_is_complete_response<br/>sentence and paragraph closed?}
        O12[Append continuation turn<br/>assistant chunk + continue prompt]
        O13[_trim_to_complete_boundary]
        O14[Append final assistant reply<br/>to self.history]
    end

    subgraph INTEL[Intelligence Layer - Gemma 3 Pipeline]
        I1[Gemma 3 as Query Expander]
        I2[Gemma 3 as Storyteller]
    end

    subgraph DATA[Data and Retrieval Layer - LoreLibrarian + ChromaDB]
        D1[LoreLibrarian.search(query, llm_pipe, n_results)]
        D2[expand_query(query, llm_pipe)]
        D3[Expanded Query Set<br/>original + up to 3 variants]
        D4[(Persistent ChromaDB<br/>default path: data/lore_db<br/>requested path: ./chroma_db)]
        D5[Candidate Pool<br/>documents + distances]
        D6[Distance and keyword filter<br/>_keyword_overlap + threshold]
        D7[CrossEncoder Re-Ranker<br/>cross-encoder/ms-marco-MiniLM-L-6-v2]
        D8[Top 3 lore snippets]
        D9[dnd_lore collection binding]
    end

    PNOTE["**Persistent**<br/>- configs/prompts.yaml<br/>- ChromaDB files on disk<br/>- dnd_lore collection"]
    VNOTE["**Volatile**<br/>- self.history<br/>- working_history<br/>- assembled response<br/>- expanded queries<br/>- candidate scores"]
    ANOTE["**Audit Gap**<br/>- no main.py in repo<br/>- UI layer modeled from README/request"]

    U1 -->|Character inputs| U2
    U2 -->|Name, Race, Class, Vibe| O2
    U3 -->|User follow-up text| O6
    O1 -->|Loaded persona config| O2
    O2 -->|Reset session state| O3
    O3 -->|Empty list []| O5
    O2 -->|Lore query string from vibe or name| D1
    O2 -->|Prompt variables + lore context| O4
    O4 -->|System message JSON| O5
    O2 -->|Seed prompt: Begin the chronicle...| O6
    O6 -->|User query text| D1
    D1 -->|Query expansion prompt| D2
    D2 -->|Prompt messages| I1
    I1 -->|Alternative search strings| D3
    D1 -->|Collection lookup target| D9
    D9 -->|Bound collection handle| D4
    D3 -->|query_texts[]| D4
    D4 -->|documents + distances| D5
    D5 -->|Candidate snippets| D6
    D6 -->|Filtered query-doc pairs| D7
    D7 -->|Scored snippets| D8
    D8 -->|Lore context string| O4
    D8 -->|Context Note string| O6
    O6 -->|Guarded user message JSON| O5
    O5 -->|Copied JSON history| O7
    O7 -->|JSON History| O8
    O8 -->|Message list + generation params| I2
    I2 -->|generated_text output| O9
    O9 -->|Assistant chunk text| O10
    O10 -->|Accumulated response text| O11
    O11 -->|No: incomplete response| O12
    O12 -->|Updated working_history JSON| O8
    O11 -->|Yes: complete response| O13
    O13 -->|Trimmed chronicle text| O14
    O14 -->|Assistant message JSON| O5
    O14 -->|Chronicle text| U4

    PNOTE -.->|On-disk state| O1
    PNOTE -.->|Persistent vectors and documents| D4
    VNOTE -.->|In-memory runtime state| O5
    VNOTE -.->|Transient retrieval state| D3
    ANOTE -.->|Modeled external boundary| U2

    style UI fill:#eef6ff,stroke:#1d4ed8,stroke-width:2px
    style ORCH fill:#f8fafc,stroke:#334155,stroke-width:2px
    style INTEL fill:#fff7ed,stroke:#c2410c,stroke-width:2px
    style DATA fill:#ecfdf5,stroke:#047857,stroke-width:2px
    style PNOTE fill:#dbeafe,stroke:#1d4ed8,stroke-dasharray: 5 5
    style VNOTE fill:#fef3c7,stroke:#b45309,stroke-dasharray: 5 5
    style ANOTE fill:#fee2e2,stroke:#b91c1c,stroke-dasharray: 5 5
```