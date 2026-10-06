"""Source-checked asynchronous image loading candidates, preserving constructors.

Candidate staging is explicit. Production staging must opt in only after the
installed-browser loading and original calculation parity checks have passed.
"""

from repository import ASSETS, REQUIREMENTS
import ast


def prepared_only(source):
    prepared = "    if sys.platform == 'emscripten':\n        raise RuntimeError('Dataset is not ready. Load the complete dataset and retry.')"
    load_guard = "        if sys.platform == 'emscripten':\n            raise RuntimeError('Dataset is not ready. Load the complete dataset and retry.')"
    if (
        source.count(prepared) == 1
        and source.count(load_guard) == 1
        and "XMLHttpRequest" not in source
    ):
        return source
    marker = "    from js import location, XMLHttpRequest, Uint8Array"
    if source.count(marker) != 1:
        raise ValueError("Synchronous image transport source contract changed")
    start = source.index(marker)
    end = source.index("\n\n\ndef load(", start)
    return (
        source[:start]
        + "    raise RuntimeError('Dataset is not ready. Load the complete dataset and retry.')\n"
        + source[end:]
    )


def write_loading_config(stage, source):
    import hashlib
    import json
    from common import PROOF, sha

    source_id = hashlib.sha256(source.encode()).hexdigest()
    manifest = ASSETS / "v1/images/manifest.json"
    identity = dict(
        source_id=source_id,
        manifest_sha256=sha(manifest),
        dependencies={name: sha(stage / name) for name in ("image_preload.py", "image_data.py")},
        environment_sha256=sha(REQUIREMENTS / "native.lock"),
    )
    build_id = hashlib.sha256(
        json.dumps(identity, sort_keys=True, separators=(",", ":")).encode()
    ).hexdigest()
    (stage / "course-loading.json").write_text(
        json.dumps(
            dict(build_id=build_id, source_id=source_id, manifest_sha256=sha(manifest)),
            sort_keys=True,
        )
    )


def unit6_loading(source, replace_once):
    # Shinylive removes the temporary app directory from sys.path after module
    # initialization. Application-local helpers must be imported at that time.
    source = replace_once(
        source,
        "from browser_torch import torch\n",
        "from browser_torch import torch\nimport image_preload\n",
    )
    tree = ast.parse(source)
    server = next(n for n in tree.body if isinstance(n, ast.FunctionDef) and n.name == "server")
    definition = next(
        n for n in server.body if isinstance(n, ast.FunctionDef) and n.name == "data_bundle"
    )
    names = sorted(
        {
            node.func.attr
            for node in ast.walk(definition)
            if isinstance(node, ast.Call)
            and isinstance(node.func, ast.Attribute)
            and isinstance(node.func.value, ast.Name)
            and node.func.value.id == "input"
        }
    )
    lines = source.splitlines(keepends=True)
    start = min([definition.lineno] + [d.lineno for d in definition.decorator_list]) - 1
    original = "".join(lines[start : definition.end_lineno])
    body = "".join(lines[definition.lineno : definition.end_lineno])
    # The calculation body stays byte-for-byte, including seeds and splits.
    factory = (
        "    def _course_build_data(params):\n        from types import SimpleNamespace\n        input = SimpleNamespace(**{key: (lambda value=value: value) for key, value in params.items()})\n"
        + body
    )
    asynchronous = f"""
    from shiny import req as _course_req
    import json as _course_json
    from pathlib import Path as _CoursePath
    _course_loading_config = _course_json.loads(_CoursePath(__file__).with_name('course-loading.json').read_text())
    _course_data_owner = image_preload.Owner(_course_loading_config['build_id'], _course_loading_config['source_id'])
    _course_data_ticket = reactive.Value(0)

    @reactive.extended_task
    async def _course_load_data(ticket, params):
        if params['dataset'] in ('MNIST', 'FashionMNIST'):
            await _course_data_owner.preload(params['dataset'], manifest_sha256=_course_loading_config['manifest_sha256'])
        return dict(ticket=ticket, params=params)

    @reactive.effect
    @reactive.event(input.load)
    def _course_start_data_load():
        _course_data_owner.cancel()
        _course_load_data.cancel()
        ticket = _course_data_ticket.get() + 1
        _course_data_ticket.set(ticket)
        params = {{name: getattr(input, name)() for name in {tuple(names)!r}}}
        _course_load_data.invoke(ticket, params)

    @reactive.calc
    def data_bundle():
        receipt = _course_load_data.result()
        _course_req(receipt['ticket'] == _course_data_ticket.get())
        return _course_build_data(receipt['params'])

    @reactive.effect
    @reactive.event(input.dataset, input.reset)
    def _course_cancel_data_load():
        if _course_load_data.status() == 'running':
            _course_data_ticket.set(_course_data_ticket.get() + 1)
            _course_data_owner.cancel()
            _course_load_data.cancel()

    def _course_end_data_load():
        _course_data_owner.cancel()
        _course_load_data.cancel()

    session.on_ended(_course_end_data_load)
"""
    source = replace_once(source, original, factory + asynchronous)
    source = replace_once(
        source,
        "    @reactive.event(arch_state)\n",
        "    @reactive.event(arch_state, _course_load_data.status)\n",
    )
    return source


def unit7_loading(source, replace_once):
    """Keep the original loader calculation; suspend transport before calling it.

    A training click owns both snapshots. A reactive consumer starts training
    only when the matching data receipt is complete, so pending preparation
    cannot silently consume the click or launch with a newer dataset.
    """
    source = replace_once(
        source,
        "from browser_torch import torch\n",
        "from browser_torch import torch\nimport image_preload\n",
    )
    tree = ast.parse(source)
    server = next(n for n in tree.body if isinstance(n, ast.FunctionDef) and n.name == "server")
    definition = next(
        n for n in server.body if isinstance(n, ast.FunctionDef) and n.name == "loaders"
    )
    names = sorted(
        {
            node.func.attr
            for node in ast.walk(definition)
            if isinstance(node, ast.Call)
            and isinstance(node.func, ast.Attribute)
            and isinstance(node.func.value, ast.Name)
            and node.func.value.id == "input"
        }
    )
    lines = source.splitlines(keepends=True)
    start = min([definition.lineno] + [d.lineno for d in definition.decorator_list]) - 1
    original = "".join(lines[start : definition.end_lineno])
    body = "".join(lines[definition.lineno : definition.end_lineno])
    factory = (
        "    def _course_build_loaders(params):\n        from types import SimpleNamespace\n        input = SimpleNamespace(**{key: (lambda value=value: value) for key, value in params.items()})\n"
        + body
    )
    asynchronous = f"""
    from shiny import req as _course_req
    import json as _course_json
    from pathlib import Path as _CoursePath
    _course_loading_config = _course_json.loads(_CoursePath(__file__).with_name('course-loading.json').read_text())
    _course_data_owner = image_preload.Owner(_course_loading_config['build_id'], _course_loading_config['source_id'])
    _course_data_ticket = reactive.Value(0)
    _course_pending_train = reactive.Value(None)

    def _course_data_params():
        params = {{name: getattr(input, name)() for name in {tuple(names)!r}}}
        if not params['use_official_test']:
            params['test'] = input.test()
        return params

    @reactive.extended_task
    async def _course_load_data(ticket, params):
        await _course_data_owner.preload(params['variant'], manifest_sha256=_course_loading_config['manifest_sha256'])
        return dict(ticket=ticket, params=params)

    def _course_begin_data_load(params):
        _course_data_owner.cancel()
        _course_load_data.cancel()
        ticket = _course_data_ticket.get() + 1
        _course_data_ticket.set(ticket)
        _course_load_data.invoke(ticket, params)
        return ticket

    @reactive.effect
    @reactive.event(input.load_data)
    def _course_start_data_load():
        _course_pending_train.set(None)
        _training_context.cancel()
        train_task.cancel()
        while not _train_q.empty():
            _train_q.get_nowait()
        _course_begin_data_load(_course_data_params())

    @reactive.calc
    def loaders():
        receipt = _course_load_data.result()
        _course_req(receipt['ticket'] == _course_data_ticket.get())
        return _course_build_loaders(receipt['params'])

    @reactive.effect
    @reactive.event(input.variant, input.reset_model, ignore_none=False)
    def _course_cancel_data_load():
        _course_pending_train.set(None)
        _training_context.cancel()
        train_task.cancel()
        while not _train_q.empty():
            _train_q.get_nowait()
        if _course_load_data.status() == 'running':
            _course_data_ticket.set(_course_data_ticket.get() + 1)
            _course_data_owner.cancel()
            _course_load_data.cancel()

    def _course_end_data_load():
        _course_data_owner.cancel()
        _course_load_data.cancel()

    session.on_ended(_course_end_data_load)
"""
    source = replace_once(source, original, factory + asynchronous)
    # The event-bound output must be invalidated when async preparation ends.
    source = replace_once(
        source,
        "    @reactive.event(input.load_data)\n    def data_info():",
        "    @reactive.event(input.load_data, _course_load_data.status)\n    def data_info():\n"
        "        from types import SimpleNamespace\n"
        "        receipt = _course_load_data.result()\n"
        '        _course_req(receipt["ticket"] == _course_data_ticket.get())\n'
        '        input = SimpleNamespace(**{key: (lambda value=value: value) for key, value in receipt["params"].items()})',
    )
    source = replace_once(
        source,
        '                "loaders": loaders(),   # IMPORTANT: build loaders on main thread\n',
        "",
    )
    source = replace_once(
        source,
        "        # kick off background job\n        train_task.invoke(params)",
        """        ticket = _course_begin_data_load(_course_data_params())
        _course_pending_train.set(dict(ticket=ticket, params=params))

    @reactive.effect
    def _course_dispatch_pending_train():
        pending = _course_pending_train.get()
        _course_req(pending is not None)
        receipt = _course_load_data.result()
        _course_req(receipt['ticket'] == pending['ticket'] == _course_data_ticket.get())
        params = dict(pending['params'], loaders=loaders())
        _course_pending_train.set(None)
        train_task.invoke(params)""",
    )
    return source
