"""Shared value validation for root and recursively referenced evidence."""
import math


def assertion_matches(assertion):
    """Check actual values; retain exact discrete decisions and fixed tolerances."""
    if not isinstance(assertion, dict): return False

    def match(expected, observed):
        if isinstance(expected, dict):
            return isinstance(observed, dict) and expected.keys() == observed.keys() and all(
                match(value, observed[key]) for key, value in expected.items())
        if isinstance(expected, list):
            return isinstance(observed, list) and len(expected) == len(observed) and all(
                match(a, b) for a, b in zip(expected, observed))
        if assertion.get('kind') == 'computation' and isinstance(expected, float):
            return (isinstance(observed, (int, float)) and not isinstance(observed, bool)
                    and math.isfinite(expected) and math.isfinite(observed)
                    and abs(expected-observed) <= 1e-5 + 1e-4*abs(expected))
        return type(expected) is type(observed) and expected == observed

    return ({'expected', 'observed'} <= assertion.keys()
            and assertion.get('matched') is True and match(assertion['expected'], assertion['observed']))
