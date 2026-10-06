"""Wrap the exact Unit 6 training body with worker-local state containers."""

import ast
import textwrap
from common import ROOT, UNITS
from reference_corrections import replace


def training_source():
    # Avoid depending on the task wrapper when this is later shared by native
    # reference generation. Only the confirmed architecture-list correction
    # changes a helper; the supplied training calculation is copied verbatim.
    source = (ROOT / UNITS[6] / "app.py").read_text()
    source = replace(
        source,
        'layers = data.get("layers", data if isinstance(data, list) else [])',
        'layers = data if isinstance(data, list) else data.get("layers", []) if isinstance(data, dict) else None',
    )
    tree = ast.parse(source)
    lines = source.splitlines(keepends=True)
    names = {"LayerSpec", "parse_architecture", "build_model", "ACTIVATIONS", "PRESETS"}
    pieces = []
    for node in tree.body:
        found = {node.name} if isinstance(node, (ast.FunctionDef, ast.ClassDef)) else set()
        if isinstance(node, ast.Assign):
            found = {target.id for target in node.targets if isinstance(target, ast.Name)}
        if found & names:
            start = min([node.lineno] + [d.lineno for d in getattr(node, "decorator_list", [])]) - 1
            pieces.append("".join(lines[start : node.end_lineno]))
    server = next(
        node for node in tree.body if isinstance(node, ast.FunctionDef) and node.name == "server"
    )
    function = next(
        node
        for node in server.body
        if isinstance(node, ast.AsyncFunctionDef) and node.name == "_train"
    )
    pieces.append(textwrap.dedent("".join(lines[function.lineno - 1 : function.end_lineno])))
    pieces.append("""
class _CourseValue:
    def __init__(self, value=None, name=None):
        self.value, self.name = value, name
    def get(self):
        return self.value
    def __call__(self):
        return self.value
    def set(self, value):
        self.value = value
        if self.name is not None:
            _q_put(dict(type='value', name=self.name, value=value))

async def _course_flush():
    await _course_checkpoint()

async def run_training_async(params):
    from types import SimpleNamespace
    global input, data_bundle, arch_state, selected_device, reactive
    input = SimpleNamespace(**{key: _CourseValue(value) for key, value in params['inputs'].items()})
    data_bundle = lambda: params['bundle']
    arch_state = lambda: params['arch_text']
    selected_device = _CourseValue(params['device'])
    reactive = SimpleNamespace(flush=_course_flush)
    for name in ('train_progress_pct', 'train_progress_msg', 'best_info', 'history_df', 'trained_model'):
        globals()[name] = _CourseValue(name=name if name != 'trained_model' else None)
    await _train()
    return dict(model=trained_model.get(), history_df=history_df.get(), best_info=best_info.get())
""")
    return "\n\n".join(pieces)
