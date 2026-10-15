# Project J.A.R.V.I.S. – System Architecture

> **Just A Rather Very Intelligent System**  
> An autonomous, 100% local-first desktop AI agent designed for Linux/Wayland (Hyprland) environments to perceive screen state, control input devices, manage applications, run system workflows, and communicate through natural voice.

---

## 1. High-Level System Architecture

The following flowchart illustrates the end-to-end architecture of the J.A.R.V.I.S. ecosystem, spanning **Sensory Ingestion**, **Cognitive Reasoning & Planning**, **Safety Gates**, and **Effectors & Environmental Control**.

```mermaid
flowchart TD
    %% User Inputs & Ingestion
    subgraph Inputs ["1. Input Interfaces & Senses"]
        UserCLI["Terminal CLI (Rich UI / Prompts)"]
        MicIn["Microphone Audio Stream (PipeWire)"]
        WakeWord["Wake-Word Detector ('Jarvis' / 'Hey Jarvis')"]
        VAD["Voice Activity Detection (RMS Energy / Silence Cutoff)"]
        LocalSTT["Speech-to-Text Engine"]
        ScreenFeed["Wayland/Hyprland Display (grim / RAM Disk)"]
        SysMetrics["Linux Telemetry (psutil / PipeWire / Network)"]
    end

    MicIn --> WakeWord
    WakeWord --> VAD
    VAD --> LocalSTT

    %% Reflex vs Cognitive Routing
    subgraph Routing ["2. Ingestion & Fast-Path Routing"]
        InputDispatcher{"Intent Classification & Dispatch"}
        FastPath["FastPath Reflex Router (<15ms Regex Match)"]
        ReflexExec["Direct Tool Reflex (Time, Mute, Volume, Telemetry)"]
    end

    UserCLI --> InputDispatcher
    LocalSTT --> InputDispatcher
    InputDispatcher -- "Deterministic Match" --> FastPath
    FastPath --> ReflexExec
    InputDispatcher -- "Complex / Unknown Goal" --> ReActEngine

    %% Cognitive Core
    subgraph CognitiveCore ["3. Cognitive Core (Local Brain)"]
        ReActEngine["Jarvis Agent Orchestrator (ReAct Loop)"]
        PlanGenerator["Workflow Planner (Deterministic Templates / LLM Fallback)"]
        ContextPruner["Context Window Manager (<14 Recent Turns)"]
        PromptBuilder["Prompt Builder (Persona + Recalled Facts + System State)"]
        ModelManager["Centralized Ollama Model Manager (VRAM Lifecycle)"]
        MemoryBank[("Persistent Memory Store (Long-Term Facts + Profile)")]
        ActiveState["Active Goal & Task State Tracker"]
    end

    ReActEngine <--> ActiveState
    ReActEngine --> PlanGenerator
    ReActEngine <--> MemoryBank
    ReActEngine --> PromptBuilder
    PromptBuilder --> ContextPruner
    ContextPruner --> ModelManager

    %% Model Inference Engines (100% Local GPU via Ollama)
    subgraph ModelPool ["4. Local Neural Inference (Ollama GPU)"]
        OllamaReasoning["Local Reasoning Core: Qwen (e.g. qwen3:8b)"]
        OllamaVision["Local Vision Core: Qwen-VL (e.g. qwen2.5vl:3b)"]
    end

    ModelManager <--> |Mutual Exclusion / VRAM Swap| OllamaReasoning
    ModelManager <--> |Mutual Exclusion / VRAM Swap| OllamaVision

    %% Safety & Authorization
    subgraph SafetyGate ["5. Safety Engine & Guardrails"]
        RiskClassifier{"Risk Tier Classifier"}
        TierSafe["SAFE: Telemetry, Read-Only, Verification"]
        TierModerate["MODERATE: App Launch, Volume, Edits"]
        TierDangerous["DANGEROUS: Shell Execution, Deletion, Kill"]
        PromptGuard["Untrusted Screen/Web Prompt Injection Filter"]
        ConfirmPrompt{"User Authorization Gate (CLI / Voice Prompt)"}
    end

    ReActEngine -- "Tool Call Request" --> PromptGuard
    PromptGuard --> RiskClassifier
    RiskClassifier --> TierSafe
    RiskClassifier --> TierModerate
    RiskClassifier --> TierDangerous
    TierSafe --> ToolRegistry
    TierModerate --> ToolRegistry
    TierDangerous --> ConfirmPrompt
    ConfirmPrompt -- "Approved" --> ToolRegistry
    ConfirmPrompt -- "Rejected" --> ReActEngine

    %% Tool Registry & Domain Toolsets
    subgraph ToolRegistry ["6. Active Tool Registry"]
        AppTools["App Control (hyprctl, gio, xdg-open)"]
        FileTools["File Operations (Read, Edit Diff, Trash)"]
        ShellTools["Shell Runner (Sandboxed Bash Execution)"]
        DOMTools["Browser DOM Driver (Playwright Chrome Driver - Primary)"]
        SysTools["System Controls (Volume, Brightness, Notifications)"]
        NavTools["Closed-Loop Screen Navigator (Visual Fallback)"]
    end

    %% Effectors & Desktop Interaction
    subgraph Effectors ["7. Hands & Effectors (Execution)"]
        WorkspaceMgr["Hyprland Workspace Manager"]
        WaylandPointer["Native Wayland Pointer (wlr-virtual-pointer)"]
        PyAutoGUI["Keyboard & Mouse Automation (XKB Keymap)"]
        TTS["Edge-TTS Synthesis Engine (en-GB-RyanNeural)"]
        AudioOut["Audio Playback (pw-play / ffplay / paplay)"]
    end

    DOMTools -. "DOM-First Browser Actions" .-> Effectors
    NavTools --> WorkspaceMgr
    NavTools --> WaylandPointer
    NavTools --> PyAutoGUI
    ReflexExec --> AudioOut
    ReActEngine --> TTS
    TTS --> AudioOut

    %% Feedback Loop
    subgraph VisionPipeline ["8. Visual Perception & Verification (Eyes)"]
        CoordTrans["Standardized CoordinateTransformer (Model <-> Screen Scaling)"]
        GrimCapture["Screen Capture Engine (RAM Disk /dev/shm & Bounded Retention)"]
        GroundingEngine["Visual Grounding Engine (In-BBox Point Validation)"]
        VisualDiff["Visual Diff Engine (Pixel Threshold & UI Change Detection)"]
    end

    ScreenFeed --> GrimCapture
    GrimCapture --> GroundingEngine
    GroundingEngine <--> OllamaVision
    GroundingEngine --> CoordTrans
    CoordTrans --> NavTools
    WaylandPointer -. "Action Dispatched" .-> ScreenFeed
    GrimCapture --> VisualDiff
    VisualDiff -. "Verification Feedback" .-> NavTools
```

---

## 2. ReAct Cognitive Loop & Decision Pipeline

The central reasoning engine executes an iterative **Reason + Act** cycle with deterministic fast-path routing, context pruning, tool deduplication, and visual feedback collection.

```mermaid
flowchart TD
    Start(["User Utterance Received"]) --> IntentCheck{"Deterministic Intent?"}

    IntentCheck -- Yes --> FastPathRoute["FastPathRouter (<15ms)"]
    FastPathRoute --> ExecFastTool["Direct Registry Tool Execution"]
    ExecFastTool --> ReturnFast["Return Timed Response to User"]

    IntentCheck -- No --> ActiveGoalCheck{"Active Goal in Progress?"}
    ActiveGoalCheck -- Yes --> AugmentGoal["Augment Prompt with Active Goal Context"]
    ActiveGoalCheck -- No --> SetNewGoal["Record Active Goal"]

    AugmentGoal --> GenPlan["Generate Workflow Plan (Templates / LLM)"]
    SetNewGoal --> GenPlan
    GenPlan --> EnrichMemory["Recall Semantically Relevant Long-Term Memories"]
    EnrichMemory --> AssemblePrompt["Build System Prompt with Memory & Telemetry"]

    AssemblePrompt --> PruneCtx["Prune Context Window (<14 Recent Messages)"]
    PruneCtx --> PrepareLLM["ModelManager: Prepare Reasoning Core (qwen3:8b)"]
    PrepareLLM --> QueryLLM["Invoke Ollama Chat API (Streaming User-Facing Path)"]

    QueryLLM --> ParseResp{"Contains Tool Calls?"}

    ParseResp -- No --> FinalReply["Format Final Assistant Message"]
    FinalReply --> ReturnOutput(["Deliver Text & Speech to User"])

    ParseResp -- Yes --> ToolLoop["Iterate Tool Calls"]
    ToolLoop --> DedupCheck{"Consecutive Duplicate Call?"}

    DedupCheck -- Yes --> SuppressCall["Suppress Repeated Execution with Error"]
    SuppressCall --> AppendToolResult["Append Tool Error to Context"]

    DedupCheck -- No --> SafetyEval["SafetyEngine: Evaluate Risk Tier & Prompt Injections"]
    SafetyEval --> RiskCheck{"Risk Tier Classification"}

    RiskCheck -- "Dangerous / Shell / Kill" --> AskConfirm{"Request Confirmation (CLI / Voice)"}
    AskConfirm -- Denied --> ToolRejected["Return Rejection Error to LLM"]
    AskConfirm -- Granted --> FocusWorkspace["Ensure Active Hyprland Workspace"]

    RiskCheck -- "Safe / Moderate" --> FocusWorkspace
    ToolRejected --> AppendToolResult

    FocusWorkspace --> ExecTool["Execute Tool Implementation"]
    ExecTool --> PerfTimer["Record Per-Stage Execution Latency"]
    PerfTimer --> AppendToolResult

    AppendToolResult --> MaxIterCheck{"Max Iterations Reached? (10)"}
    MaxIterCheck -- No --> PruneCtx
    MaxIterCheck -- Yes --> AbortLoop["Synthesize Failure Summary to User"]
    AbortLoop --> ReturnOutput
```

---

## 3. Closed-Loop GUI "Computer Use" Workflow

For graphical user interface automation, J.A.R.V.I.S. prioritizes **DOM/Accessibility-first** interaction when controlling browsers, falling back to a robust closed-loop **Observe ➔ Locate ➔ Act ➔ Verify** visual pipeline for desktop apps.

```mermaid
flowchart LR
    subgraph Step1 ["1. Observe"]
        Capture["ScreenCaptureEngine (grim)"]
        RAMDisk["Save to /dev/shm (or /tmp) with max_frames=20"]
        Capture --> RAMDisk
    end

    subgraph Step2 ["2. Locate"]
        VLMManager["ModelManager: Load Vision Core (qwen2.5vl:3b)"]
        VLM["Ollama Vision Model"]
        Parser["Parser: Independent BBox & Point Extraction"]
        Transformer["CoordinateTransformer (Model Scale -> Screen Resolution)"]
        RAMDisk --> VLMManager
        VLMManager --> VLM
        VLM --> Parser
        Parser --> Transformer
    end

    subgraph Step3 ["3. Act"]
        FocusWS["Switch & Verify Hyprland Workspace"]
        SafetyCheck["Verify Point Strictly Inside Bounding Box"]
        PointerDispatch["Dispatch Action (wlr-virtual-pointer / PyAutoGUI)"]
        Transformer --> FocusWS
        FocusWS --> SafetyCheck
        SafetyCheck --> PointerDispatch
    end

    subgraph Step4 ["4. Verify"]
        PostCapture["Post-Action Capture (persistent=False)"]
        DiffEngine["VisualDiffEngine (pixel_threshold=15 Filtering)"]
        StateCheck{"UI Changed?"}
        PointerDispatch --> PostCapture
        PostCapture --> DiffEngine
        DiffEngine --> StateCheck
    end

    StateCheck -- Yes --> SuccessOutcome(["Action Confirmed Successful"])
    StateCheck -- No --> Recover(["Report Failure: Suppress Dangerous Clicks"])
```

---

## 4. Subsystem Breakdown & Tech Stack

| Subsystem | Key Modules | Primary Technologies | Key Functions & Responsibilities |
|---|---|---|---|
| **Voice In** | `voice/wake_word.py`<br>`voice/listener.py` | PipeWire (`pw-record`), Whisper STT, RMS Energy VAD | Wake-word detection ("Jarvis"), voice activity detection, streaming speech-to-text pipeline. |
| **Voice Out** | `voice/speaker.py` | Microsoft Edge-TTS (`en-GB-RyanNeural`), `pw-play` / `ffplay` | British gentleman speech synthesis, streaming sentence pipelining, instant barge-in cut-off. |
| **Cognitive Core** | `core/agent.py`<br>`core/fast_path.py`<br>`core/prompts.py` | Python 3, Pydantic, Custom ReAct Loop, Deterministic Workflow Templates | Orchestrates reasoning, maintains active goal tracking, token context pruning, reflex execution routing. |
| **Model Manager** | `core/model_manager.py`<br>`core/llm_client.py` | Local Ollama API, Centralized Residency Manager | Exclusive VRAM residency on laptop GPUs (RTX 3050 4GB), mutual exclusion between reasoning and vision, configurable idle timeout. |
| **Perception (Vision)** | `vision/capture.py`<br>`vision/coordinates.py`<br>`vision/grounding.py`<br>`vision/analyzer.py`<br>`vision/visual_diff.py` | `grim` (Wayland), Pillow, Qwen 2.5-VL via Ollama | Zero-disk-I/O RAM screenshot capture with bounded retention (20 frames), bidirectional coordinate scaling/clamping, pixel-thresholded UI diff detection. |
| **GUI Automation** | `gui_driver/navigator.py`<br>`gui_driver/mouse_keyboard.py`<br>`core/workspace.py` | Wayland `wlr-virtual-pointer`, PyAutoGUI, `hyprctl` | Smooth cursor motion, Wayland C pointer clicks, keyboard shortcuts, verified workspace isolation. |
| **Browser DOM** | `tools/browser_dom.py` | Playwright, Chromium Headless/Headed | Primary browser interface: sub-15ms DOM inspection, element query, text entry, and link traversal without vision overhead. |
| **Safety Engine** | `core/safety.py` | Python Regex, Risk Tiers (`SAFE`, `MODERATE`, `DANGEROUS`) | Block dangerous shell patterns (fork bombs, `rm -rf /`), gate destructive actions behind user confirmation, filter untrusted screen prompts. |
| **Persistent Memory** | `memory/store.py`<br>`tools/memory_ops.py` | Atomic JSON Storage, Semantic Keyword Scoring | Cross-session user preferences, profile attributes, conversation facts, and proactive prompt context. |
| **System Tools** | `tools/app_control.py`<br>`tools/file_ops.py`<br>`tools/system_ctl.py`<br>`tools/system_info.py` | `hyprctl`, `psutil`, `gio`, `xdg-open`, `notify-send`, PipeWire | Launch/focus apps via Hyprland, safe file edits with diff preview, audio volume, hardware telemetry. |

---

## 5. Architectural Invariants & Design Principles

1. **100% Local-First & Zero Cloud Leakage**:
   - Primary reasoning (`qwen3:8b`) and visual grounding (`qwen2.5vl:3b`) run completely locally on GPU via Ollama.
   - User desktop pixels, shell commands, and audio streams never leave the local machine.

2. **Centralized VRAM Lifecycle Management**:
   - Laptop GPUs (such as RTX 3050 4GB VRAM) cannot co-host an 8B LLM and a 3B VLM simultaneously.
   - `OllamaModelManager` enforces strict mutual exclusion: loading the vision model unloads the reasoning model, and vice versa.
   - Configurable idle timeouts prevent unnecessary GPU power draw and thermal throttling while avoiding redundant model reloads during active multi-turn tasks.

3. **Standardized Coordinates & Safe Grounding**:
   - Resized VLM images (e.g. max 1280px) are mapped strictly back to native screen resolution via `CoordinateTransformer`.
   - Never fall back to center-screen clicks on visual grounding failure; failures return `None, None` to abort unsafe actions immediately.
   - Click points are verified to lie strictly inside the detected bounding box before cursor dispatch.

4. **Transient Capture Bounded Retention**:
   - Perception and verification loops use temporary capture directories (`/dev/shm` RAM disk with `/tmp` fallback) with `persistent=False`.
   - Bounded retention (`_cleanup_old_ram_frames`) automatically purges obsolete frames past 20 captures, preventing RAM/disk bloat.
   - Permanent storage (`workspace/screenshots/`) is reserved strictly for explicit user-requested screenshots.

5. **DOM-First, Vision as Fallback**:
   - For web and browser interactions, deterministic DOM accessibility trees (`tools/browser_dom.py`) are prioritized for sub-15ms reliability.
   - Vision-based closed-loop grounding is used for native desktop applications and canvas elements where accessibility trees are inaccessible.

6. **Ultra-Low Latency Dual-Track Processing**:
   - **Track 1 (Fast-Path Reflex)**: Deterministic, regex-matched commands execute directly in `<15ms`, bypassing LLM round-trips entirely.
   - **Track 2 (Cognitive ReAct)**: Complex, exploratory tasks run through the multi-turn ReAct loop with arrow-chain workflow pre-planning.

7. **Multi-Tiered Safety Guardrails & Untrusted Content Defense**:
   - Destructive operations (`safe_delete_file`, `terminate_process`, `execute_shell_command`) are gated by explicit interactive confirmation (CLI `y/N` or Voice `yes/no`).
   - Untrusted text captured from screens or web pages is filtered to prevent indirect prompt injections from hijacking the agent.
