"""Manual thinking must not downgrade max/ultra to the medium default.

GLM on zai-indirect uses the Anthropic Messages manual-budget route, unlike
modern Claude's adaptive route. The request budget is a ceiling, not a promise
that a provider will consume that many reasoning tokens.
"""

import pytest

from agent.anthropic_adapter import build_anthropic_kwargs


@pytest.mark.parametrize("model", ["glm-5.3", "minimax-m2", "claude-sonnet-4-20250514"])
@pytest.mark.parametrize("max_tokens", [4096, 64000])
@pytest.mark.parametrize(
    ("effort", "expected_budget"),
    [("low", 4000), ("medium", 8000), ("high", 16000),
     ("xhigh", 32000), ("max", 32000), ("ultra", 32000)],
)
def test_manual_effort_preserves_budget_and_output_headroom(model, max_tokens, effort, expected_budget):
    kwargs = build_anthropic_kwargs(
        model=model,
        messages=[{"role": "user", "content": "Think carefully."}],
        tools=None,
        max_tokens=max_tokens,
        reasoning_config={"enabled": True, "effort": effort},
        base_url="https://api.z.ai/api/anthropic" if model.startswith("glm-") else None,
        preserve_dots=model.startswith("glm-"),
    )

    assert kwargs["thinking"] == {"type": "enabled", "budget_tokens": expected_budget}
    assert kwargs["max_tokens"] == max(max_tokens, expected_budget + 4096)
    assert kwargs["temperature"] == 1
    assert "output_config" not in kwargs


@pytest.mark.parametrize("effort", ["max", "ultra"])
def test_disabled_manual_thinking_stays_disabled_at_max_effort(effort):
    kwargs = build_anthropic_kwargs(
        model="glm-5.3",
        messages=[{"role": "user", "content": "Reply briefly."}],
        tools=None,
        max_tokens=4096,
        reasoning_config={"enabled": False, "effort": effort},
        base_url="https://api.z.ai/api/anthropic",
        preserve_dots=True,
    )

    assert "thinking" not in kwargs
    assert kwargs["max_tokens"] == 4096
    assert "temperature" not in kwargs
