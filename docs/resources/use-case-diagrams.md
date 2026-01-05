# Use Case Diagrams

The following diagrams illustrate the NeMo Guardrails architecture for each use case.

---

## 1. Content Safety

```{mermaid}
%%{init: {'theme': 'neutral', 'themeVariables': { 'background': 'transparent' }}}%%

flowchart LR
  A("Application")

  subgraph NemoGuardrails["NeMo Guardrails"]
    direction TB
    B1("Input Content Check")
    B2("Output Content Check")
  end

  subgraph Safety["Content Safety Options"]
    direction TB
    C1("LLM Self-Check")
    C2("Llama 3.1 NemoGuard<br/>Content Safety NIM")
    C3("LlamaGuard")
    C4("Fiddler")
    C5("ActiveFence")
    C6("AI Defense")
  end

  D("LLM")

  A --> NemoGuardrails
  NemoGuardrails --> Safety
  Safety --> D
  D --> Safety
  Safety --> NemoGuardrails
  NemoGuardrails --> A

  style A fill:#d8d8e8,stroke:#999
  style NemoGuardrails fill:#f0f7e6,stroke:#76b900,stroke-width:2px
  style Safety fill:#e8f4d9,stroke:#76b900
  style D fill:#d8d8e8,stroke:#999
```

*Content safety guardrails check both user inputs and LLM outputs for harmful content.*

---

## 2. Jailbreak Protection

```{mermaid}
%%{init: {'theme': 'neutral', 'themeVariables': { 'background': 'transparent' }}}%%

flowchart LR
  A("Application")

  subgraph NemoGuardrails["NeMo Guardrails"]
    direction TB
    B1("Input Jailbreak Detection")
  end

  subgraph Jailbreak["Jailbreak Detection Options"]
    direction TB
    C1("Self-Check Detection")
    C2("Heuristic Detection")
    C3("NemoGuard Jailbreak<br/>Detection NIM")
    C4("Prompt Security")
    C5("Pangea AI Guard")
  end

  D("LLM")

  A --> NemoGuardrails
  NemoGuardrails --> Jailbreak
  Jailbreak --> D
  D --> NemoGuardrails
  NemoGuardrails --> A

  style A fill:#d8d8e8,stroke:#999
  style NemoGuardrails fill:#f0f7e6,stroke:#76b900,stroke-width:2px
  style Jailbreak fill:#ffe6e6,stroke:#cc0000
  style D fill:#d8d8e8,stroke:#999
```

*Jailbreak protection prevents adversarial attempts from bypassing safety measures.*

---

## 3. Topic Control

```{mermaid}
%%{init: {'theme': 'neutral', 'themeVariables': { 'background': 'transparent' }}}%%

flowchart LR
  A("Application")

  subgraph NemoGuardrails["NeMo Guardrails"]
    direction TB
    B1("Topic Validation")
    B2("Dialog Flow Control")
  end

  subgraph TopicControl["Topic Control Options"]
    direction TB
    C1("Dialog Rails<br/>(Colang)")
    C2("Topical Rails")
    C3("NemoGuard Topic<br/>Control NIM")
  end

  D("LLM")

  A --> NemoGuardrails
  NemoGuardrails --> TopicControl
  TopicControl --> D
  D --> NemoGuardrails
  NemoGuardrails --> A

  style A fill:#d8d8e8,stroke:#999
  style NemoGuardrails fill:#f0f7e6,stroke:#76b900,stroke-width:2px
  style TopicControl fill:#e6f0ff,stroke:#0066cc
  style D fill:#d8d8e8,stroke:#999
```

*Topic control ensures conversations stay within predefined subject boundaries.*

---

## 4. PII Detection and Masking

```{mermaid}
%%{init: {'theme': 'neutral', 'themeVariables': { 'background': 'transparent' }}}%%

flowchart LR
  A("Application")

  subgraph NemoGuardrails["NeMo Guardrails"]
    direction TB
    B1("Input PII Detection")
    B2("Output PII Masking")
  end

  subgraph PII["PII Detection Options"]
    direction TB
    C1("Microsoft Presidio")
    C2("Private AI")
    C3("AutoAlign PII")
    C4("GuardrailsAI Validators")
  end

  D("LLM")

  A --> NemoGuardrails
  NemoGuardrails --> PII
  PII --> D
  D --> PII
  PII --> NemoGuardrails
  NemoGuardrails --> A

  style A fill:#d8d8e8,stroke:#999
  style NemoGuardrails fill:#f0f7e6,stroke:#76b900,stroke-width:2px
  style PII fill:#fff3e6,stroke:#ff9900
  style D fill:#d8d8e8,stroke:#999
```

*PII detection protects user privacy by detecting and masking sensitive data.*

---

## 5. Agentic Security

```{mermaid}
%%{init: {'theme': 'neutral', 'themeVariables': { 'background': 'transparent' }}}%%

flowchart TB
  A("Application")

  subgraph NemoGuardrails["NeMo Guardrails"]
    direction TB
    B1("Pre-Tool Validation")
    B2("Post-Tool Validation")
    B3("Action Monitoring")
  end

  subgraph Agent["LLM Agent"]
    direction LR
    D1("LLM")
    D2("Tool Calls")
  end

  subgraph Tools["External Tools & APIs"]
    direction TB
    E1("Database")
    E2("File System")
    E3("External APIs")
  end

  A --> NemoGuardrails
  NemoGuardrails --> Agent
  Agent --> NemoGuardrails
  D2 --> Tools
  Tools --> D2
  NemoGuardrails --> A

  style A fill:#d8d8e8,stroke:#999
  style NemoGuardrails fill:#f0f7e6,stroke:#76b900,stroke-width:2px
  style Agent fill:#e6e6fa,stroke:#6600cc
  style Tools fill:#ffe6cc,stroke:#cc6600
```

*Agentic security provides guardrails for LLM agents using tools and external systems.*

---

## 6. Custom and Third-Party Guardrails

```{mermaid}
%%{init: {'theme': 'neutral', 'themeVariables': { 'background': 'transparent' }}}%%

flowchart LR
  A("Application")

  subgraph NemoGuardrails["NeMo Guardrails"]
    direction TB
    B1("Custom Input Rails")
    B2("Custom Output Rails")
  end

  subgraph Custom["Custom Guardrail Options"]
    direction TB
    C1("Python Actions")
    C2("LangChain Tools")
    C3("Third-Party APIs")
    C4("Custom Scripts")
  end

  D("LLM")

  A --> NemoGuardrails
  NemoGuardrails --> Custom
  Custom --> D
  D --> Custom
  Custom --> NemoGuardrails
  NemoGuardrails --> A

  style A fill:#d8d8e8,stroke:#999
  style NemoGuardrails fill:#f0f7e6,stroke:#76b900,stroke-width:2px
  style Custom fill:#e6ffe6,stroke:#009900
  style D fill:#d8d8e8,stroke:#999
```

*Build custom guardrails using Python actions, LangChain tools, or third-party APIs.*

---

## 7. Integration Options

```{mermaid}
%%{init: {'theme': 'neutral', 'themeVariables': { 'background': 'transparent' }}}%%

flowchart TB
  subgraph SDK["Python SDK Integration"]
    direction LR
    A1("Python App")
    B1("LLMRails")
    C1("LLM")
    A1 --> B1 --> C1
  end

  subgraph Server["Server API Integration"]
    direction LR
    A2("Any Application")
    B2("NeMo Guardrails<br/>Server :8000")
    C2("LLM")
    A2 -->|"HTTP API"| B2 --> C2
  end

  subgraph OpenAI["OpenAI-Compatible Endpoint"]
    direction LR
    A3("Existing OpenAI<br/>Integration")
    B3("/v1/chat/completions")
    A3 --> B3
  end

  style SDK fill:#f0f7e6,stroke:#76b900
  style Server fill:#e6f0ff,stroke:#0066cc
  style OpenAI fill:#fff3e6,stroke:#ff9900
```

*Multiple integration options: Python SDK, HTTP Server API, or OpenAI-compatible endpoint.*

---

## Combined Architecture Overview

```{mermaid}
%%{init: {'theme': 'neutral', 'themeVariables': { 'background': 'transparent' }}}%%

flowchart TB
  A("Application Code")

  subgraph NemoGuardrails["NVIDIA NeMo Guardrails Library"]
    direction TB

    subgraph InputRails["Input Rails"]
      I1("Content Safety")
      I2("Jailbreak Detection")
      I3("PII Detection")
      I4("Topic Validation")
    end

    subgraph OutputRails["Output Rails"]
      O1("Content Check")
      O2("PII Masking")
      O3("Response Validation")
    end
  end

  subgraph NIMs["NemoGuard NIMs"]
    direction TB
    N1("Content Safety NIM")
    N2("Jailbreak Detection NIM")
    N3("Topic Control NIM")
  end

  subgraph ThirdParty["Third-Party Integrations"]
    direction TB
    T1("Presidio / Private AI")
    T2("ActiveFence / AI Defense")
    T3("LlamaGuard / Fiddler")
  end

  C("Large Language Model")

  A <--> InputRails
  InputRails <--> NIMs
  InputRails <--> ThirdParty
  NIMs <--> C
  ThirdParty <--> C
  C <--> OutputRails
  OutputRails <--> A

  style A fill:#d8d8e8,stroke:#999
  style NemoGuardrails fill:#f0f7e6,stroke:#76b900,stroke-width:2px
  style InputRails fill:#e8f4d9,stroke:#76b900
  style OutputRails fill:#e8f4d9,stroke:#76b900
  style NIMs fill:#d9edf7,stroke:#31708f
  style ThirdParty fill:#fcf8e3,stroke:#8a6d3b
  style C fill:#d8d8e8,stroke:#999
```

*Complete architecture showing all guardrail types and integration options.*
