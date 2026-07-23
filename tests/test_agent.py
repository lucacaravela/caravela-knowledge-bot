"""Unit tests for agent helpers."""

from types import SimpleNamespace

import agent


def test_compact_history_drops_tool_traffic_and_merges():
    messages = [
        {"role": "user", "content": "quais fintechs vimos?"},
        # Assistant turn with thinking text + tool call (SDK-object style).
        {
            "role": "assistant",
            "content": [
                SimpleNamespace(type="text", text="Vou buscar no pipeline."),
                SimpleNamespace(type="tool_use", id="t1", name="search_pipeline", input={}),
            ],
        },
        # Tool result turn (dict style) — must be dropped entirely.
        {
            "role": "user",
            "content": [
                {"type": "tool_result", "tool_use_id": "t1", "content": "big dump" * 500}
            ],
        },
        # Final assistant answer.
        {
            "role": "assistant",
            "content": [SimpleNamespace(type="text", text="Encontrei 3 fintechs: A, B, C.")],
        },
        {"role": "user", "content": "e a segunda?"},
    ]

    compacted = agent.compact_history(messages)

    assert compacted == [
        {"role": "user", "content": "quais fintechs vimos?"},
        {
            "role": "assistant",
            "content": "Vou buscar no pipeline.\n\nEncontrei 3 fintechs: A, B, C.",
        },
        {"role": "user", "content": "e a segunda?"},
    ]
    # No tool dumps survive.
    assert all("big dump" not in m["content"] for m in compacted)


def test_compact_history_alternates_roles():
    messages = [
        {"role": "user", "content": "oi"},
        {"role": "assistant", "content": [SimpleNamespace(type="text", text="ola")]},
    ]
    compacted = agent.compact_history(messages)
    roles = [m["role"] for m in compacted]
    assert roles == ["user", "assistant"]
