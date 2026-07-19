import re

_RESOURCEPACK_DEFAULT_MODEL_POLICY = {
    "endpoint": "chat.completions",
    "allow_sampling_params": True,
    "max_output_tokens": 4096,
}

_RESOURCEPACK_MODEL_CAPABILITY_RULES = [
    {
        "match": "contains",
        "value": "codex",
        "endpoint": "responses",
        "allow_sampling_params": False,
    },
    {
        "match": "regex",
        "value": r"^gpt-5.*-pro$",
        "endpoint": "responses",
        "allow_sampling_params": False,
    },
    {
        "match": "suffix",
        "value": "-chat-latest",
        "endpoint": "chat.completions",
        "allow_sampling_params": False,
    },
]


def get_resourcepack_model_policy(model):
    """Return endpoint/parameter policy for a model from centralized capability rules."""
    name = (model or "").strip().lower()
    policy = dict(_RESOURCEPACK_DEFAULT_MODEL_POLICY)
    for rule in _RESOURCEPACK_MODEL_CAPABILITY_RULES:
        if is_rule_match(name, rule):
            policy.update({k: v for k, v in rule.items() if k not in {"match", "value"}})
            break
    return policy


def should_use_responses_api(model):
    """Whether model should be routed to Responses API based on registry rules."""
    return get_resourcepack_model_policy(model)["endpoint"] == "responses"


def is_rule_match(model_name, rule):
    """Check whether a model string matches one capability rule."""
    matcher = rule.get("match")
    value = (rule.get("value") or "").lower()
    if matcher == "contains":
        return value in model_name
    if matcher == "suffix":
        return model_name.endswith(value)
    if matcher == "prefix":
        return model_name.startswith(value)
    if matcher == "regex":
        return bool(re.match(rule.get("value") or "", model_name))
    return False
