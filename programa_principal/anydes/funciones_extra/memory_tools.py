from __future__ import annotations

from anydes.memoria.manager import MemoryManager
from anydes.nucleo.config import Config
from anydes.nucleo.config import config


def _manager() -> MemoryManager:
    return MemoryManager.from_config(config)


def handlers_for_config(bound_config: Config) -> dict[str, object]:
    def manager() -> MemoryManager:
        return MemoryManager.from_config(bound_config)

    def permanent_read(key: str) -> dict[str, object]:
        return _memory_permanent_read(manager(), key)

    def permanent_write(argument: str) -> dict[str, object]:
        return _memory_permanent_write(manager(), argument)

    def temporary_read(key: str) -> dict[str, object]:
        return _memory_temporary_read(manager(), key)

    def temporary_write(argument: str) -> dict[str, object]:
        return _memory_temporary_write(manager(), argument)

    def temporary_clear(_: str = "") -> dict[str, object]:
        manager().temporary.clear()
        return {"cleared": True, "tier": "temporary"}

    def protected_write(argument: str) -> dict[str, object]:
        return _memory_protected_write(manager(), argument)

    def protected_read(key: str) -> dict[str, object]:
        return _memory_protected_read(manager(), key)

    def profile(_: str = "") -> dict[str, object]:
        return _memory_profile(manager())

    def conversation_search(query: str) -> dict[str, object]:
        query = query.strip()
        if not query:
            return {"found": False, "error": "missing_query"}
        return {"query": query, "summary": manager().search_conversation(query)}

    def recent(_: str = "") -> dict[str, object]:
        return {"summary": manager().recent_conversation_summary()}

    return {
        "memory_profile": profile,
        "memory_conversation_search": conversation_search,
        "memory_recent": recent,
        "memory_permanent_read": permanent_read,
        "memory_permanent_write": permanent_write,
        "memory_temporary_read": temporary_read,
        "memory_temporary_write": temporary_write,
        "memory_temporary_clear": temporary_clear,
        "memory_protected_read": protected_read,
        "memory_protected_write": protected_write,
        "memory_read": permanent_read,
        "memory_write": permanent_write,
    }


def memory_permanent_read(key: str) -> dict[str, object]:
    return _memory_permanent_read(_manager(), key)


def _memory_permanent_read(memory_manager: MemoryManager, key: str) -> dict[str, object]:
    key = key.strip()
    memory = memory_manager.permanent
    if not key:
        items = memory.all()
        return {
            "tier": "permanent",
            "key": "",
            "found": bool(items),
            "items": dict(sorted(items.items())),
        }

    value = memory.get(key)
    if value is None:
        return {"tier": "permanent", "key": key, "found": False}
    return {"tier": "permanent", "key": key, "found": True, "value": value}


def memory_permanent_write(argument: str) -> dict[str, object]:
    return _memory_permanent_write(_manager(), argument)


def _memory_permanent_write(memory_manager: MemoryManager, argument: str) -> dict[str, object]:
    key, separator, value = argument.partition("=")
    key = key.strip()
    value = value.strip()
    if not separator or not key:
        return {"saved": False, "tier": "permanent", "error": "expected_key_value_format"}
    result = memory_manager.set_permanent(key, value)
    return {
        "saved": result.ok,
        "tier": "permanent",
        "key": key,
        "sensitive_rejected": not result.ok,
    }


def memory_temporary_read(key: str) -> dict[str, object]:
    return _memory_temporary_read(_manager(), key)


def _memory_temporary_read(memory_manager: MemoryManager, key: str) -> dict[str, object]:
    key = key.strip()
    memory = memory_manager.temporary
    if not key:
        items = memory.all()
        return {
            "tier": "temporary",
            "key": "",
            "found": bool(items),
            "items": dict(sorted(items.items())),
        }
    value = memory.get(key)
    if value is None:
        return {"tier": "temporary", "key": key, "found": False}
    return {"tier": "temporary", "key": key, "found": True, "value": value}


def memory_temporary_write(argument: str) -> dict[str, object]:
    return _memory_temporary_write(_manager(), argument)


def _memory_temporary_write(memory_manager: MemoryManager, argument: str) -> dict[str, object]:
    key, separator, value = argument.partition("=")
    key = key.strip()
    value = value.strip()
    if not separator or not key:
        return {"saved": False, "tier": "temporary", "error": "expected_key_value_format"}
    result = memory_manager.set_temporary(key, value)
    return {
        "saved": result.ok,
        "tier": "temporary",
        "key": key,
        "sensitive_rejected": not result.ok,
    }


def memory_temporary_clear(_: str = "") -> dict[str, object]:
    _manager().temporary.clear()
    return {"cleared": True, "tier": "temporary"}


def memory_protected_write(argument: str) -> dict[str, object]:
    return _memory_protected_write(_manager(), argument)


def _memory_protected_write(memory_manager: MemoryManager, argument: str) -> dict[str, object]:
    key, separator, value = argument.partition("=")
    key = key.strip()
    value = value.strip()
    if not separator or not key:
        return {"saved": False, "tier": "protected", "error": "expected_key_value_format"}
    result = memory_manager.set_protected(key, value)
    return {
        "saved": result.ok,
        "tier": "protected",
        "key": key,
        "protected_store_enabled": memory_manager.protected.enabled,
    }


def memory_protected_read(key: str) -> dict[str, object]:
    return _memory_protected_read(_manager(), key)


def _memory_protected_read(memory_manager: MemoryManager, key: str) -> dict[str, object]:
    key = key.strip()
    if not key:
        return {"tier": "protected", "found": False, "error": "missing_key"}
    result = memory_manager.get_protected(key)
    payload: dict[str, object] = {
        "tier": "protected",
        "key": key,
        "found": result.ok,
        "protected_store_enabled": memory_manager.protected.enabled,
    }
    if result.ok:
        payload["value"] = result.message
    return payload


def memory_profile(_: str = "") -> dict[str, object]:
    return _memory_profile(_manager())


def memory_conversation_search(query: str) -> dict[str, object]:
    query = query.strip()
    if not query:
        return {"found": False, "error": "missing_query"}
    return {"query": query, "summary": _manager().search_conversation(query)}


def memory_recent(_: str = "") -> dict[str, object]:
    return {"summary": _manager().recent_conversation_summary()}


def _memory_profile(memory_manager: MemoryManager) -> dict[str, object]:
    permanent = memory_manager.permanent.all()
    temporary = memory_manager.temporary.all()
    protected_keys = memory_manager.protected.keys()
    classified = memory_manager.conversation.counts()
    return {
        "summary": memory_manager.profile_summary(),
        "permanent_count": len(permanent),
        "temporary_count": len(temporary),
        "protected_key_count": len(protected_keys),
        "conversation_counts": {
            "normal": classified.get("Normal", 0),
            "personal": classified.get("Personal", 0),
            "secret": classified.get("Secretos", 0),
        },
    }
