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


def test_make_conversation_title_strips_quotes():
    class FakeMessages:
        def create(self, **kwargs):
            assert kwargs["model"] == agent.TITLE_MODEL
            return SimpleNamespace(
                content=[SimpleNamespace(type="text", text='"Receita da Celes"\n')]
            )

    client = SimpleNamespace(messages=FakeMessages())
    assert agent.make_conversation_title(client, "qual a receita da celes?") == "Receita da Celes"


def test_make_conversation_title_rejects_sentence_like_answers():
    class ChattyMessages:
        def create(self, **kwargs):
            return SimpleNamespace(
                content=[
                    SimpleNamespace(
                        type="text",
                        text="I don't have any information about conversations you've had before this one.",
                    )
                ]
            )

    client = SimpleNamespace(messages=ChattyMessages())
    assert agent.make_conversation_title(client, "o que falamos antes?") is None


def test_make_conversation_title_returns_none_on_failure():
    class BrokenMessages:
        def create(self, **kwargs):
            raise RuntimeError("api down")

    client = SimpleNamespace(messages=BrokenMessages())
    assert agent.make_conversation_title(client, "oi") is None


def test_compact_history_alternates_roles():
    messages = [
        {"role": "user", "content": "oi"},
        {"role": "assistant", "content": [SimpleNamespace(type="text", text="ola")]},
    ]
    compacted = agent.compact_history(messages)
    roles = [m["role"] for m in compacted]
    assert roles == ["user", "assistant"]
