# J.A.R.V.I.S. Local-First Multi-Layer Memory Architecture 🧠

This document provides a comprehensive technical overview of the local-first, multi-layer memory system implemented for Project J.A.R.V.I.S.

---

## 1. Architectural Overview

Jarvis utilizes a **5-layer cognitive memory architecture** designed for fast, local-first retrieval on laptop hardware (NVIDIA RTX 3050 6GB VRAM, 16GB RAM, Arch Linux). Memory storage, retrieval, and reasoning are completely decoupled: the local LLM (`qwen3:4b`) does not re-read entire conversation histories, but instead receives token-budgeted, highly relevant recalled context assembled dynamically per turn.

```
+-------------------------------------------------------------------------+
|                  User Request / Computer Observation                    |
+-------------------------------------------------------------------------+
                                    |
                                    v
+-------------------------------------------------------------------------+
|                         Memory Extractor (Qwen3)                        |
|        - Asynchronous, non-blocking background extraction pool          |
|        - Categorizes: semantic, episodic, procedural, project           |
|        - Zero-LLM regex/keyword heuristic fallback                      |
+-------------------------------------------------------------------------+
                                    |
                                    v
+-------------------------------------------------------------------------+
|                    Unified SQLite Store (WAL Mode)                      |
|                                                                         |
|  [Working Memory]     -> task_checkpoints (resumable steps & errors)    |
|  [Semantic Memory]    -> memories & user_profile (facts, preferences)   |
|  [Episodic Memory]    -> memories (notable milestones & error events)   |
|  [Project Memory]     -> projects (stack, branch, architecture)         |
|  [Procedural Memory]  -> procedures (verified workflows & fixes)        |
+-------------------------------------------------------------------------+
          |                                               |
          v                                               v
+------------------------+                     +------------------------+
| SQLite FTS5 (BM25)     |                     | Ollama Vector Embeddings|
| - Full-text keyword    |                     | - embeddinggemma:300m  |
| - Active-only triggers |                     | - 768-dim binary BLOBs |
| - Sub-millisecond scan |                     | - Cosine similarity    |
+------------------------+                     +------------------------+
          \                                               /
           \                                             /
            v                                           v
       +-----------------------------------------------------+
       |   Hybrid Retrieval: Reciprocal Rank Fusion (RRF)    |
       |   Score(d) = sum( weight / (k + rank) )             |
       +-----------------------------------------------------+
                                 |
                                 v
+-------------------------------------------------------------------------+
|                       ContextBuilder (<1200 Tokens)                     |
|  1. User Profile Attributes (Name, core preferences)                    |
|  2. Resumed Task Checkpoint (Completed steps, pending, last error)     |
|  3. Active Project Metadata (Tech stack, architecture)                 |
|  4. Recalled Reusable Procedures & Relevant Semantic Facts              |
|  5. General Long-Term Facts & Preferences                              |
+-------------------------------------------------------------------------+
                                 |
                                 v
+-------------------------------------------------------------------------+
|                      J.A.R.V.I.S. Autonomous Agent                      |
|        - Dispatches tools, executes actions, updates checkpoints        |
+-------------------------------------------------------------------------+
```

---

## 2. The Five Memory Layers

| Layer | Responsibility | Persistence Target | Retrieval Method |
| :--- | :--- | :--- | :--- |
| **Working Memory** | Active goal, completed steps, pending steps, last action result, error state | `task_checkpoints` table | `get_latest_active_checkpoint()` |
| **Semantic Memory** | Stable user preferences, facts, profile attributes, tool configs | `memories` table (`layer='semantic'`), `user_profile` table | Hybrid RRF / FTS5 BM25 |
| **Episodic Memory** | Notable past events, decisions, completed tasks, error analyses | `memories` table (`layer='episodic'`) | FTS5 BM25 & category filters |
| **Project Memory** | Project architecture, framework stack, root directory, blockers | `projects` table | Project ID lookup |
| **Procedural Memory**| Reusable troubleshooting workflows, multi-step system fixes | `procedures` table & `memories` (`layer='procedural'`) | `recall_procedures()` & FTS5 |

---

## 3. SQLite Database Schema & Indexing

Located at `data/jarvis.db` by default (configurable via `JARVIS_DB_PATH`).

### Key Tables

1. **`memories`**: Core multi-layer memory repository.
   - Fields: `id`, `layer`, `category`, `subject`, `predicate`, `content`, `tags`, `project_id`, `importance`, `confidence`, `status`, `source`, `provenance`, `created_at`, `updated_at`, `last_accessed`, `access_count`, `expires_at`, `superseded_by`.
   - Constraints: Foreign key `superseded_by REFERENCES memories(id) ON DELETE SET NULL`.
   - Indexes: `layer`, `category`, `status`, `project_id`, `importance DESC`, `updated_at DESC`.

2. **`memories_fts`**: SQLite FTS5 Virtual Table.
   - Tokenizer: `porter unicode61`.
   - Synchronized automatically via SQLite triggers:
     - `trg_memories_ai`: Indexes new records when `status = 'active'`.
     - `trg_memories_au`: Keeps index in sync on updates, removing superseded or inactive memories.
     - `trg_memories_ad`: Automatically removes deleted memories from index.

3. **`memory_embeddings`**: Local Vector Storage.
   - Stores normalized 768-dimensional float vectors as compact binary BLOBs (`struct.pack`).
   - Foreign key: `memory_id REFERENCES memories(id) ON DELETE CASCADE`.

4. **`task_checkpoints`**: Resumable Working Memory.
   - Tracks `task_id`, `user_request`, `goal`, `status` (`in_progress`, `completed`, `failed`, `interrupted`, `cancelled`), `completed_steps` (JSON), `pending_steps` (JSON), `last_tool_name`, `last_tool_result`, `error_summary`.

5. **`procedures`**: Verified Workflows & Troubleshooting Fixes.
   - Tracks `id`, `title`, `trigger_intent`, `preconditions` (JSON), `steps` (JSON), `expected_outcome`, `verification_method`, `failure_modes` (JSON), `confidence`, `verified`, `success_count`, `last_executed`.

6. **`user_profile`**: Direct key-value store for user personal traits (name, role, default terminal, browser).

---

## 4. Hybrid Retrieval with Reciprocal Rank Fusion (RRF)

When searching memory (`recall_hybrid`), Jarvis queries both FTS5 full-text search and local vector embeddings concurrently, then fuses their results:

$$\text{RRF\_Score}(d) = \sum_{m \in \{\text{FTS}, \text{VEC}\}} \frac{w_m}{k + \text{rank}_m(d)}$$

Where:
- $k = 60$ (smoothing constant preventing high-rank domination)
- $w_{\text{FTS}} = 0.5$ (keyword precision weight)
- $w_{\text{VEC}} = 0.5$ (semantic similarity weight)

### Graceful Fallback Strategy
1. **Primary**: Fast FTS5 keyword retrieval (<1ms) is used for default agent turns, keeping latency imperceptible.
2. **Semantic / Hybrid**: If enabled, `EmbeddingManager` queries local Ollama `embeddinggemma:300m`.
3. **Offline / Busy Fallback**: If Ollama is offline or times out, retrieval falls back to pure FTS5 BM25 search without throwing exceptions or blocking the user.
4. **Token Containment Fallback**: If FTS5 yields no results, Python token overlap matching catches partial matches.

---

## 5. Fact Supersession & Conflict Resolution

When Jarvis learns new information that contradicts an older memory with the same `(subject, predicate)` (e.g. changing preferred editor from Neovim to VS Code):
1. Deduplication verifies whether it is an exact duplicate or updated phrasing.
2. If conflicting content exists for the same `(subject, predicate)`:
   - The new memory is inserted with `status = 'active'`.
   - The older conflicting memory is updated to `status = 'superseded'` and `superseded_by = <new_memory_id>`.
   - FTS5 trigger `trg_memories_au` automatically evicts the superseded memory from the full-text search index.
   - Active recall queries exclusively return the updated preference.

---

## 6. Resumable Working Memory & Task Checkpoints

When Jarvis executes a multi-step desktop task:
1. `save_checkpoint()` creates or updates a checkpoint at each tool step.
2. If an operation fails (e.g. database lock file, network drop, missing window), the checkpoint records `status = 'failed'`, the list of already-completed steps, and the exact error.
3. On the next turn, if the user says *"continue"*, *"try again"*, or gives guidance:
   - `ContextBuilder` detects the active checkpoint and injects it into Tier 2 of the system prompt.
   - Jarvis sees which actions already succeeded and executes only the remaining pending actions, preventing redundant execution loops.
4. When the goal succeeds, `update_checkpoint_status(task_id, status='completed')` cleanly clears the active checkpoint.

---

## 7. Backup & Disaster Recovery

### Automatic Backups
- **Legacy Migration**: Upon first startup, if `data/jarvis_memory.json` is detected, a lossless backup is created in `data/backups/jarvis_memory_backup_<timestamp>.json` before populating SQLite tables.
- **SQLite Snapshots**: `DatabaseManager.create_backup(label)` creates timestamped point-in-time copies in `data/backups/jarvis_backup_<timestamp>_<label>.db`.

### Manual Backup
To take an immediate backup:
```python
from memory.db import DatabaseManager
mgr = DatabaseManager()
backup_file = mgr.create_backup(label="manual_snapshot")
print(f"Backup created at: {backup_file}")
```

### Manual Restoration
To restore from a backup:
```bash
# 1. Stop Jarvis
# 2. Copy the desired backup over the active database:
cp data/backups/jarvis_backup_20261015_143000_manual_snapshot.db data/jarvis.db
```
