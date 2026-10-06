import json
import os
import re
import threading
import uuid
from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional, Set

from config.settings import settings


@dataclass
class MemoryEntry:
    id: str
    content: str
    category: str = "fact"  # "fact", "preference", "instruction", "profile", "task"
    tags: List[str] = field(default_factory=list)
    importance: int = 3  # 1 (low) to 5 (critical)
    created_at: str = field(default_factory=lambda: datetime.now(timezone.utc).isoformat())
    updated_at: str = field(default_factory=lambda: datetime.now(timezone.utc).isoformat())
    access_count: int = 0
    last_accessed: str = field(default_factory=lambda: datetime.now(timezone.utc).isoformat())

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "MemoryEntry":
        return cls(
            id=data.get("id", f"mem_{uuid.uuid4().hex[:8]}"),
            content=data.get("content", ""),
            category=data.get("category", "fact"),
            tags=data.get("tags", []),
            importance=int(data.get("importance", 3)),
            created_at=data.get("created_at", datetime.now(timezone.utc).isoformat()),
            updated_at=data.get("updated_at", datetime.now(timezone.utc).isoformat()),
            access_count=int(data.get("access_count", 0)),
            last_accessed=data.get("last_accessed", datetime.now(timezone.utc).isoformat()),
        )


class MemoryStore:
    """Persistent, thread-safe memory and profile store for J.A.R.V.I.S.
    
    Persists across sessions to atomic JSON storage. Supports semantic keyword search,
    importance weighting, access tracking, user profiling, and prompt context generation.
    """

    def __init__(self, file_path: Optional[Path | str] = None):
        self.file_path = Path(file_path) if file_path else settings.MEMORY_FILE_PATH
        self._lock = threading.RLock()
        self.memories: Dict[str, MemoryEntry] = {}
        self.profile: Dict[str, Any] = {}
        self._load()

    def _load(self):
        with self._lock:
            if not self.file_path.exists():
                return

            try:
                with open(self.file_path, "r", encoding="utf-8") as f:
                    data = json.load(f)
                    self.profile = data.get("profile", {})
                    raw_entries = data.get("memories", [])
                    self.memories = {
                        entry["id"]: MemoryEntry.from_dict(entry)
                        for entry in raw_entries
                        if isinstance(entry, dict) and "content" in entry
                    }
            except Exception as e:
                # Corrupted or unreadable file: keep state clean but do not crash
                pass

    def _save(self):
        with self._lock:
            try:
                parent = self.file_path.parent
                parent.mkdir(parents=True, exist_ok=True)

                data = {
                    "version": "1.0",
                    "updated_at": datetime.now(timezone.utc).isoformat(),
                    "profile": self.profile,
                    "memories": [entry.to_dict() for entry in self.memories.values()],
                }

                # Atomic write via temporary file
                tmp_path = self.file_path.with_suffix(".tmp")
                with open(tmp_path, "w", encoding="utf-8") as f:
                    json.dump(data, f, indent=2, ensure_ascii=False)
                os.replace(tmp_path, self.file_path)
            except Exception:
                pass

    def _tokenize(self, text: str) -> Set[str]:
        words = re.findall(r"\b[a-zA-Z0-9_\-\.]+\b", text.lower())
        stop_words = {
            "a", "an", "the", "and", "or", "in", "on", "at", "to", "for", "with",
            "is", "was", "are", "were", "of", "it", "that", "this", "my", "your",
            "me", "you", "i", "sir", "jarvis", "please", "be", "have", "has", "do"
        }
        return {w for w in words if len(w) > 1 and w not in stop_words}

    def remember(
        self,
        content: str,
        category: str = "fact",
        tags: Optional[List[str]] = None,
        importance: int = 3,
    ) -> MemoryEntry:
        """Stores a new fact, preference, or instruction into persistent memory.
        If an identical or near-duplicate memory exists, updates it cleanly."""
        with self._lock:
            clean_content = content.strip()
            if not clean_content:
                raise ValueError("Memory content cannot be empty.")

            clean_tags = [t.strip().lower() for t in (tags or []) if t.strip()]
            importance = max(1, min(5, int(importance)))
            now = datetime.now(timezone.utc).isoformat()

            # Automatic profile extraction for common user attributes
            lower = clean_content.lower()
            name_match = re.search(r"\b(?:my name is|call me|i am|i'm|user(?:'s)? name is)\s+([a-zA-Z]+)\b", lower)
            if name_match:
                name = name_match.group(1).capitalize()
                self.profile["name"] = name
                category = "profile"
                importance = 5
                if "name" not in clean_tags:
                    clean_tags.append("name")

            # Check for existing memory match (by exact text or very high token overlap)
            content_tokens = self._tokenize(clean_content)
            existing_match: Optional[MemoryEntry] = None

            for entry in self.memories.values():
                if entry.content.lower() == clean_content.lower():
                    existing_match = entry
                    break
                entry_tokens = self._tokenize(entry.content)
                if content_tokens and entry_tokens:
                    intersection = len(content_tokens & entry_tokens)
                    union = len(content_tokens | entry_tokens)
                    if union > 0 and (intersection / union) > 0.8:
                        existing_match = entry
                        break

            if existing_match:
                existing_match.content = clean_content
                existing_match.category = category
                existing_match.importance = max(existing_match.importance, importance)
                existing_match.updated_at = now
                existing_match.tags = list(set(existing_match.tags + clean_tags))
                self._save()
                return existing_match

            new_id = f"mem_{uuid.uuid4().hex[:8]}"
            entry = MemoryEntry(
                id=new_id,
                content=clean_content,
                category=category,
                tags=clean_tags,
                importance=importance,
                created_at=now,
                updated_at=now,
                access_count=0,
                last_accessed=now,
            )
            self.memories[new_id] = entry
            self._save()
            return entry

    def recall(
        self,
        query: str,
        category: Optional[str] = None,
        limit: int = 5,
    ) -> List[MemoryEntry]:
        """Searches long-term memory for entries matching the query."""
        with self._lock:
            q_clean = query.strip().lower()
            if not q_clean:
                return self.list_all(category=category, limit=limit)

            q_tokens = self._tokenize(q_clean)
            scored: List[tuple[float, MemoryEntry]] = []
            now = datetime.now(timezone.utc).isoformat()

            for entry in self.memories.values():
                if category and entry.category.lower() != category.lower():
                    continue

                score = 0.0
                entry_lower = entry.content.lower()

                # Exact phrase match
                if q_clean in entry_lower:
                    score += 15.0

                # Token matching
                entry_tokens = self._tokenize(entry.content)
                if q_tokens and entry_tokens:
                    overlap = len(q_tokens & entry_tokens)
                    score += overlap * 3.0

                # Tag matching
                for tag in entry.tags:
                    if tag in q_tokens or tag in q_clean:
                        score += 5.0

                if score > 0:
                    # Weight by importance (scale 1.0 to 2.0)
                    score *= (1.0 + (entry.importance - 1) * 0.25)
                    scored.append((score, entry))

            scored.sort(key=lambda x: x[0], reverse=True)
            results = [item[1] for item in scored[:limit]]

            # Update access statistics
            for res in results:
                res.access_count += 1
                res.last_accessed = now

            if results:
                self._save()

            return results

    def list_all(
        self,
        category: Optional[str] = None,
        limit: int = 50,
    ) -> List[MemoryEntry]:
        """Returns stored memories sorted by importance and update date."""
        with self._lock:
            items = list(self.memories.values())
            if category:
                items = [m for m in items if m.category.lower() == category.lower()]
            items.sort(key=lambda m: (m.importance, m.updated_at), reverse=True)
            return items[:limit]

    def forget(self, query_or_id: str) -> Optional[MemoryEntry]:
        """Deletes a memory matching ID or content substring."""
        with self._lock:
            target_key = query_or_id.strip()

            # Exact ID lookup
            if target_key in self.memories:
                removed = self.memories.pop(target_key)
                self._save()
                return removed

            # Substring match
            target_lower = target_key.lower()
            for k, entry in list(self.memories.items()):
                if target_lower in entry.content.lower():
                    removed = self.memories.pop(k)
                    self._save()
                    return removed

            return None

    def set_profile_attribute(self, key: str, value: Any):
        """Sets a persistent user profile attribute."""
        with self._lock:
            self.profile[key] = value
            self._save()

    def get_profile_attribute(self, key: str, default: Any = None) -> Any:
        with self._lock:
            return self.profile.get(key, default)

    def get_profile(self) -> Dict[str, Any]:
        with self._lock:
            return dict(self.profile)

    def clear(self, category: Optional[str] = None) -> int:
        """Clears memories, optionally scoped to category."""
        with self._lock:
            if category:
                to_delete = [k for k, m in self.memories.items() if m.category.lower() == category.lower()]
                for k in to_delete:
                    del self.memories[k]
                self._save()
                return len(to_delete)
            else:
                count = len(self.memories)
                self.memories.clear()
                self._save()
                return count

    def get_context_summary(self, max_items: int = 5) -> str:
        """Constructs a compact context block of profile and highest-priority memories
        suitable for direct injection into system prompt context."""
        with self._lock:
            lines = []

            # Profile attributes
            if self.profile:
                prof_items = []
                if "name" in self.profile:
                    prof_items.append(f"Name: {self.profile['name']}")
                for k, v in self.profile.items():
                    if k != "name":
                        prof_items.append(f"{k.capitalize()}: {v}")
                if prof_items:
                    lines.append(f"• User Profile: {', '.join(prof_items)}")

            # Top memories by importance
            top_memories = self.list_all(limit=max_items)
            if top_memories:
                lines.append("• Stored Long-Term Memories:")
                for m in top_memories:
                    tag_str = f" [{', '.join(m.tags)}]" if m.tags else ""
                    lines.append(f"  - ({m.category.capitalize()}) {m.content}{tag_str}")

            if not lines:
                return ""

            return "[PERSISTENT LONG-TERM MEMORY & USER PROFILE]\n" + "\n".join(lines)


# Global singleton instance
memory_store = MemoryStore()
