from backend.logger import get_logger

logger = get_logger("langchain.memory")

# Simple in-memory conversation store
# Each session_id maps to a list of {"role": "user"/"assistant", "content": "..."}
_memory_store: dict[str, list[dict]] = {}

MAX_TURNS = 10  # remember last 10 exchanges


def get_memory(session_id: str = "default") -> list[dict]:
    if session_id not in _memory_store:
        _memory_store[session_id] = []
        logger.info(f"Created new memory | session='{session_id}'")
    return _memory_store[session_id]


def save_to_memory(session_id: str, question: str, answer: str):
    memory = get_memory(session_id)
    memory.append({"role": "user", "content": question})
    memory.append({"role": "assistant", "content": answer})

    # Keep only last MAX_TURNS exchanges (each exchange = 2 messages)
    if len(memory) > MAX_TURNS * 2:
        _memory_store[session_id] = memory[-(MAX_TURNS * 2):]

    logger.debug(f"Saved to memory | session='{session_id}' | total_messages={len(_memory_store[session_id])}")


def get_history_as_text(session_id: str) -> str:
    memory = get_memory(session_id)
    if not memory:
        return ""

    lines = []
    for msg in memory:
        role = "User" if msg["role"] == "user" else "Assistant"
        lines.append(f"{role}: {msg['content']}")

    history = "\n".join(lines)
    logger.debug(f"History loaded | session='{session_id}' | messages={len(memory)}")
    return history


def reset_memory(session_id: str = "default"):
    if session_id in _memory_store:
        del _memory_store[session_id]
        logger.info(f"Memory cleared | session='{session_id}'")