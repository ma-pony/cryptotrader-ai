"""Hashes describe the actual request; response identity never falls back to configuration."""

from unittest.mock import patch

from langchain_core.messages import AIMessage, HumanMessage, SystemMessage
from langchain_core.outputs import ChatGeneration, LLMResult

from cryptotrader.agents.base import create_runtime_llm_factory
from cryptotrader.runtime_config.models import LlmConfig


def test_runtime_factory_preserves_request_prompt_and_actual_model_evidence():
    from cryptotrader.backtest.evidence import capture_model_evidence

    with capture_model_evidence() as evidence, patch("cryptotrader.agents.base.ChatOpenAI") as chat:
        factory = create_runtime_llm_factory(LlmConfig(), api_key="NEVER-PERSIST-SECRET")
        factory(model="requested-model", role="technical", with_fallback=False)
        callbacks = chat.call_args.kwargs["callbacks"]
        for callback in callbacks:
            if callback is evidence:
                callback.on_chat_model_start(
                    {},
                    [[SystemMessage(content="Actual system"), HumanMessage(content="Historical bar")]],
                    run_id="call-1",
                    invocation_params={"model": "requested-model"},
                )
        for callback in callbacks:
            callback.on_llm_end(
                LLMResult(
                    generations=[
                        [
                            ChatGeneration(
                                message=AIMessage(content="long", response_metadata={"model_name": "actual-model-1"})
                            )
                        ]
                    ]
                ),
                run_id="call-1",
            )
    assert len(evidence.records) == 1
    record = evidence.records[0]
    assert record["requested_model"] == "requested-model"
    assert record["actual_model"] == "actual-model-1"
    assert len(record["prompt_hash"]) == 64
    assert "Actual system" not in str(record)
    assert "NEVER-PERSIST-SECRET" not in str(record)


def test_unknown_response_identity_remains_explicit():
    from cryptotrader.backtest.evidence import capture_model_evidence

    with capture_model_evidence() as evidence:
        evidence.on_chat_model_start(
            {}, [[HumanMessage(content="bar")]], run_id="call-1", invocation_params={"model": "requested-model"}
        )
        evidence.on_llm_end(LLMResult(generations=[]), run_id="call-1")
    assert evidence.records[0]["actual_model"] is None
    assert evidence.records[0]["actual_model_reason"] == "response_identity_unavailable"


def test_transport_url_and_non_string_response_are_not_safe_model_identities():
    from cryptotrader.backtest.evidence import capture_model_evidence

    with capture_model_evidence() as evidence:
        evidence.on_chat_model_start(
            {},
            [[HumanMessage(content="bar")]],
            run_id="call-unsafe",
            invocation_params={"model": "https://example.invalid?token=secret"},
        )
        evidence.on_llm_end(LLMResult(generations=[], llm_output={"model": {"token": "secret"}}), run_id="call-unsafe")
    assert evidence.records[0]["requested_model"] is None
    assert evidence.records[0]["actual_model"] is None
    assert "secret" not in str(evidence.records)


async def test_concurrent_runs_keep_separate_request_evidence():
    import asyncio

    from cryptotrader.backtest.evidence import capture_model_evidence, current_model_evidence

    async def run(identity):
        with capture_model_evidence() as evidence:
            await asyncio.sleep(0)
            current_model_evidence().on_chat_model_start(
                {}, [[HumanMessage(content=identity)]], run_id=identity, invocation_params={"model": identity}
            )
            await asyncio.sleep(0)
            return evidence.records

    left, right = await asyncio.gather(run("left-model"), run("right-model"))
    assert [item["requested_model"] for item in left] == ["left-model"]
    assert [item["requested_model"] for item in right] == ["right-model"]
    assert left[0]["prompt_hash"] != right[0]["prompt_hash"]
