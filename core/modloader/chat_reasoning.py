"""The management snapshot's optional reasoning override; absence uses the service default."""


def validate_reasoning_effort(value):
    if value is None or isinstance(value, str) and value in ('none', 'low', 'medium', 'high', 'xhigh', 'max'):
        return
    if type(value) is int and 1 <= value <= 100:
        return
    raise ValueError('reasoningEffort 必须为受支持的档位或 1–100 的整数')
