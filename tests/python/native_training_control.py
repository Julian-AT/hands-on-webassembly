"""Independent serial native control from unchanged supplied assignment code."""

import ast
import copy
import dataclasses
import json
import math
import random
import textwrap
import typing

import numpy as np
import pandas as pd
import torch
from common import ROOT, extract


def run(params):
    namespace = dict(vars(typing))
    namespace.update(
        torch=torch,
        nn=torch.nn,
        np=np,
        pd=pd,
        json=json,
        math=math,
        random=random,
        copy=copy,
        dataclass=dataclasses.dataclass,
        DataLoader=torch.utils.data.DataLoader,
        _q_put=lambda message: None,
    )
    names = [
        "parse_architecture",
        "build_model",
        "ACTIVATIONS",
        "PRESETS",
        "_yamlish_to_json",
        "LayerSpec",
        "_tuple_or_int",
        "get_first_conv_weights",
    ]
    exec(extract(7, names), namespace)
    source = (ROOT / "assignments/7/app.py").read_text()
    tree = ast.parse(source)
    server = next(n for n in tree.body if isinstance(n, ast.FunctionDef) and n.name == "server")
    body = next(
        n for n in server.body if isinstance(n, ast.FunctionDef) and n.name == "run_training_sync"
    )
    lines = source.splitlines(True)
    exec(textwrap.dedent("".join(lines[body.lineno - 1 : body.end_lineno])), namespace)
    return namespace["run_training_sync"](params)
