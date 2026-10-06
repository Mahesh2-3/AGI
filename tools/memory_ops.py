from typing import Any, Dict, List, Optional
from memory.store import memory_store
from tools.base import registry


@registry.register(
    description="Stores a persistent memory, fact, user preference, contact, or instruction that Jarvis must remember across sessions forever."
)
def remember_fact(
    fact: str,
    category: str = "fact",
    tags: Optional[List[str]] = None,
    importance: int = 3,
) -> Dict[str, Any]:
    try:
        entry = memory_store.remember(
            content=fact,
            category=category,
            tags=tags or [],
            importance=importance,
        )
        return {
            "success": True,
            "id": entry.id,
            "fact": entry.content,
            "category": entry.category,
            "importance": entry.importance,
            "message": f"Successfully stored in long-term memory: '{entry.content}'",
        }
    except Exception as e:
        return {"success": False, "error": f"Failed to commit memory: {str(e)}"}


@registry.register(
    description="Searches persistent long-term memory for stored facts, preferences, contacts, or past instructions matching a query."
)
def recall_memory(
    query: str,
    category: Optional[str] = None,
    limit: int = 5,
) -> Dict[str, Any]:
    try:
        entries = memory_store.recall(query=query, category=category, limit=limit)
        results = [
            {
                "id": m.id,
                "content": m.content,
                "category": m.category,
                "importance": m.importance,
                "tags": m.tags,
            }
            for m in entries
        ]
        return {
            "success": True,
            "query": query,
            "count": len(results),
            "memories": results,
            "formatted_summary": "\n".join([f"- [{m['category'].upper()}] {m['content']}" for m in results]) if results else "No matching memories found.",
        }
    except Exception as e:
        return {"success": False, "error": f"Failed to recall memories: {str(e)}"}


@registry.register(
    description="Lists stored memories, user preferences, and facts, optionally filtered by category ('fact', 'preference', 'profile', 'task')."
)
def list_memories(
    category: Optional[str] = None,
    limit: int = 25,
) -> Dict[str, Any]:
    try:
        entries = memory_store.list_all(category=category, limit=limit)
        results = [
            {
                "id": m.id,
                "content": m.content,
                "category": m.category,
                "importance": m.importance,
                "tags": m.tags,
            }
            for m in entries
        ]
        return {
            "success": True,
            "total": len(results),
            "memories": results,
        }
    except Exception as e:
        return {"success": False, "error": f"Failed to list memories: {str(e)}"}


@registry.register(
    description="Removes or deletes a specific memory or fact from persistent memory by matching its ID or content text."
)
def forget_memory(query_or_id: str) -> Dict[str, Any]:
    try:
        removed = memory_store.forget(query_or_id)
        if removed:
            return {
                "success": True,
                "id": removed.id,
                "deleted_content": removed.content,
                "message": f"Forgotten: '{removed.content}'",
            }
        return {
            "success": False,
            "error": f"No memory found matching '{query_or_id}'.",
        }
    except Exception as e:
        return {"success": False, "error": f"Failed to delete memory: {str(e)}"}


@registry.register(
    description="Stores or updates a persistent user profile attribute or preference (e.g. favorite browser, code directory, theme)."
)
def update_user_preference(key: str, value: str) -> Dict[str, Any]:
    try:
        clean_key = key.strip().lower().replace(" ", "_")
        memory_store.set_profile_attribute(clean_key, value)
        # Also register as a memory entry
        memory_store.remember(
            content=f"User preference {clean_key}: {value}",
            category="preference",
            tags=["preference", clean_key],
            importance=4,
        )
        return {
            "success": True,
            "key": clean_key,
            "value": value,
            "message": f"Stored preference: {clean_key} = {value}",
        }
    except Exception as e:
        return {"success": False, "error": f"Failed to update preference: {str(e)}"}


@registry.register(
    description="Returns an overview of persistent memory status, user profile attributes, and memory counts."
)
def get_memory_summary() -> Dict[str, Any]:
    try:
        profile = memory_store.get_profile()
        all_memories = memory_store.list_all(limit=100)
        return {
            "success": True,
            "profile": profile,
            "total_memories": len(all_memories),
            "categories": list({m.category for m in all_memories}),
        }
    except Exception as e:
        return {"success": False, "error": f"Failed to retrieve memory summary: {str(e)}"}
