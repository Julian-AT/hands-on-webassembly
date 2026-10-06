"""Keep synchronous previews from consuming the training random streams.

The caller must finish loading data before entering the scope. No await belongs
inside it: another task must never observe the temporary inspection state.
"""
from contextlib import contextmanager
from functools import wraps
import random
import numpy as np


@contextmanager
def preserve(torch):
    python_state, numpy_state = random.getstate(), np.random.get_state()
    native = not getattr(torch, '_course_rng_installed', False)
    if native:
        cpu_state = torch.get_rng_state().clone()
    else:
        stream = torch.default_generator._rng
        cpu_state = (stream.seed, stream.state.get_state())
        import borch._ops as ops
        last_seed = ops._LAST_SEED[0]
    try:
        yield
    finally:
        random.setstate(python_state)
        np.random.set_state(numpy_state)
        if native:
            torch.set_rng_state(cpu_state)
        else:
            stream.seed, state = cpu_state
            stream.state.set_state(state)
            ops._LAST_SEED[0] = last_seed


def protected(torch, prepare=None):
    def decorate(function):
        @wraps(function)
        def inspect(*args, **kwargs):
            # Data generation retains its legitimate RNG consumption. Only
            # inspection of the already prepared data runs in the scope.
            if prepare is not None:
                prepare()
            with preserve(torch):
                return function(*args, **kwargs)
        return inspect
    return decorate
