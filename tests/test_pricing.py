from dataclasses import replace

from app.clients import parse_llm_usage
from app.config import settings
from app.service import calculate_deepseek_cost_cny


class FakeUsage:
    prompt_tokens = 1_000_000
    completion_tokens = 100_000
    model_extra = {
        "prompt_cache_hit_tokens": 600_000,
        "prompt_cache_miss_tokens": 400_000,
    }

    def model_dump(self):
        return {
            "prompt_tokens": self.prompt_tokens,
            "completion_tokens": self.completion_tokens,
        }


class GenericUsage:
    prompt_tokens = 1_000
    completion_tokens = 100
    model_extra = {}

    def model_dump(self):
        return {"prompt_tokens": self.prompt_tokens, "completion_tokens": self.completion_tokens}


def test_parse_deepseek_cache_usage_fields():
    result = parse_llm_usage(FakeUsage())
    assert result["cache_hit_input_tokens"] == 600_000
    assert result["cache_miss_input_tokens"] == 400_000
    assert result["cache_usage_reported"] is True


def test_missing_cache_fields_are_conservatively_counted_as_miss():
    result = parse_llm_usage(GenericUsage())
    assert result["cache_hit_input_tokens"] == 0
    assert result["cache_miss_input_tokens"] == 1_000
    assert result["cache_usage_reported"] is False


def test_deepseek_v4_flash_official_cny_formula():
    config = replace(
        settings,
        llm_cache_hit_cost_per_million_cny=0.02,
        llm_cache_miss_cost_per_million_cny=1.0,
        llm_output_cost_per_million_cny=2.0,
    )
    cost = calculate_deepseek_cost_cny(
        config,
        cache_hit_input_tokens=1_000_000,
        cache_miss_input_tokens=1_000_000,
        output_tokens=1_000_000,
    )
    assert cost == 3.02

