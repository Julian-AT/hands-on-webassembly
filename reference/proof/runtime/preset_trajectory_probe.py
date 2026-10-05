"""Synthetic first-operation diagnostics against retained native trajectories."""
import copy
from dataclasses import dataclass
import json
import math
import sys
import types
from typing import Any, Dict, List, Optional, Tuple, Union
import numpy as np


def rng_bytes(torch):
    # This diagnostic encodes only the existing uniform MT19937 stream. Its
    # representation is checked against the locked native engine independently.
    stream = torch.default_generator._rng
    _, words, position, gaussian, cached = stream.state.get_state()
    if gaussian or cached:
        raise ValueError('Normal-cache state is outside this uniform-only diagnostic')
    initial = np.random.RandomState(stream.seed % 2**32).get_state()[1]
    next_word = 0 if position == 624 and np.array_equal(words, initial) else position
    state = np.zeros(632, dtype='<u8')
    state[0] = stream.seed
    state[1] = np.uint64(2**32 + 625 - position)
    state[2] = next_word
    state[3:627] = words
    return state.view(np.uint8).copy()


def trajectory(torch, definitions, record):
    parent = record['parent_state']
    unit = parent['unit']
    module = types.ModuleType(f'_preset_diagnostic_unit{unit}')
    sys.modules[module.__name__] = module
    module.__dict__.update(globals(), torch=torch, nn=torch.nn)
    exec(definitions[str(unit)], module.__dict__)
    shape, classes, kind = record['state']
    torch.manual_seed(parent['seed'])
    architecture = module.PRESETS[parent['preset']]
    model = module.build_model(module.parse_architecture(json.dumps(architecture)), tuple(shape))
    arrays = {f'initial/{key}': value.detach().numpy().copy()
              for key, value in model.state_dict().items()}
    arrays['initial/rng'] = rng_bytes(torch)
    # Use the native operation's exact immutable input snapshot. Generating a
    # fresh linspace in each backend can introduce input rounding differences
    # before any model operation has been compared.
    with np.load(record['control'], allow_pickle=False) as control:
        x = torch.tensor(control['inputs'].copy(), dtype=torch.float32)
        y = torch.tensor(control['targets'].copy())
    output = model(x)
    arrays.update(inputs=x.numpy(), targets=y.numpy(), logits=output.detach().numpy())
    error = None
    try:
        loss = (torch.nn.MSELoss() if kind == 'regression' else torch.nn.CrossEntropyLoss())(output, y)
        arrays['loss'] = loss.detach().numpy()
        if not bool(torch.isfinite(loss)):
            raise ValueError('Nonfinite loss')
        optimizer = torch.optim.SGD(model.parameters(), lr=.01, momentum=.9)
        loss.backward()
        for key, value in model.named_parameters():
            arrays['gradient/'+key] = value.grad.detach().numpy().copy()
        optimizer.step()
        for key, value in model.state_dict().items():
            arrays['updated/'+key] = value.detach().numpy().copy()
        for key, value in model.named_parameters():
            arrays['momentum/'+key] = np.asarray(optimizer._state(value)['momentum_buffer']).copy()
    except Exception as exception:
        error = type(exception).__name__ + ': ' + str(exception)
    arrays['final/rng'] = rng_bytes(torch)
    observed = dict(loss_executable=error is None,
                    task_output_shape_matches=tuple(output.shape) == (2, classes),
                    output_shape=list(output.shape), error=error)
    return observed, arrays
