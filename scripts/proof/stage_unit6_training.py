"""Ownership-checked Unit 6 task invocation and reactive result consumption."""

import ast


def unit6_training(source, replace_once, *, native=False):
    tree = ast.parse(source)
    definition = next(
        n for n in ast.walk(tree) if isinstance(n, ast.AsyncFunctionDef) and n.name == "_train"
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
    marker = "import inspection_rng\n" if native else "from browser_torch import torch\n"
    source = replace_once(source, marker, marker + "import isolated_training\n")
    replacement = f"""
    import queue as _course_queue
    _course_train_q = _course_queue.SimpleQueue()
    _course_training_context = isolated_training.Executor(6)
    _course_train_ticket = reactive.Value(0)
    _course_data_epoch = reactive.Value(0)
    _course_pending_train = reactive.Value(None)

    class _CourseTrainingError(RuntimeError):
        def __init__(self, ticket, error):
            super().__init__(str(error))
            self.ticket = ticket

    @reactive.extended_task
    async def _course_train_task(ticket, params):
        try:
            if params['bundle'].get('dataset_name') != params['dataset_name']:
                raise ValueError("Dataset selection has changed. Load/Reload data before training.")
            result = await _course_training_context.run(params,
                lambda message: _course_train_q.put(dict(ticket=ticket, message=message)))
            with inspection_rng.preserve(torch):
                result['model'] = isolated_training.restore_model(result.pop('model_record'),
                    build_model, parse_architecture(params['arch_text']))
            return dict(ticket=ticket, **result)
        except Exception as error:
            raise _CourseTrainingError(ticket, error) from error

    def _course_cancel_train():
        _course_train_ticket.set(_course_train_ticket.get() + 1)
        _course_pending_train.set(None)
        _course_training_context.cancel()
        _course_train_task.cancel()
        while not _course_train_q.empty():
            _course_train_q.get_nowait()

    @reactive.effect
    @reactive.event(input.load, input.dataset, arch_state, ignore_none=False)
    def _course_cancel_training_generation():
        _course_data_epoch.set(_course_data_epoch.get() + 1)
        _course_cancel_train()
        trained_model.set(None)
        history_df.set(pd.DataFrame())
        best_info.set("")
        train_progress_pct.set(0)
        train_progress_msg.set("")

    @reactive.effect
    @reactive.event(input.train)
    def _course_start_train():
        _course_cancel_train()
        ticket = _course_train_ticket.get()
        params = dict(inputs={{name: getattr(input, name)() for name in {tuple(names)!r}}},
            arch_text=arch_state(), device=selected_device.get(), dataset_name=input.dataset())
        train_progress_pct.set(0)
        train_progress_msg.set("Initializing…")
        best_info.set("")
        history_df.set(pd.DataFrame())
        _course_pending_train.set(dict(ticket=ticket, data_epoch=_course_data_epoch.get(), params=params))

    @reactive.effect
    def _course_dispatch_train():
        pending = _course_pending_train.get()
        from shiny import req
        req(pending is not None)
        req(pending['data_epoch'] == _course_data_epoch.get())
        bundle = data_bundle()
        params = dict(pending['params'], bundle=bundle,
            loaders=tuple(bundle.get(name) for name in ('train','val','test')) if bundle['kind'] == 'img' else ())
        _course_pending_train.set(None)
        _course_train_task.invoke(pending['ticket'], params)

    @reactive.effect
    def _course_pump_train():
        reactive.invalidate_later(.05)
        values = dict(train_progress_pct=train_progress_pct, train_progress_msg=train_progress_msg,
            best_info=best_info, history_df=history_df)
        while not _course_train_q.empty():
            item = _course_train_q.get_nowait()
            if item['ticket'] != _course_train_ticket.get():
                continue
            message = item['message']
            if message.get('name') == 'train_progress_msg' and message.get('value') == 'Training complete':
                continue
            if message.get('type') == 'value' and message.get('name') in values:
                values[message['name']].set(message['value'])

    @reactive.effect
    def _course_finalize_train():
        if _course_train_task.status() == 'error':
            error = _course_train_task.error()
            if getattr(error, 'ticket', None) != _course_train_ticket.get():
                return
            message = str(error)
            best_info.set('Error: ' + message)
            train_progress_msg.set('Error: ' + message)
            train_progress_pct.set(0)
            trained_model.set(None)
            return
        if _course_train_task.status() != 'success':
            return
        receipt = _course_train_task.result()
        from shiny import req
        req(receipt['ticket'] == _course_train_ticket.get())
        while not _course_train_q.empty():
            _course_train_q.get_nowait()
        trained_model.set(receipt['model'])
        history_df.set(receipt['history_df'])
        best_info.set(receipt['best_info'])
        train_progress_pct.set(100)
        train_progress_msg.set('Training complete')

    def _course_end_training():
        _course_training_context.cancel()
        _course_train_task.cancel()

    session.on_ended(_course_end_training)
"""
    source = replace_once(source, original, replacement)
    source = replace_once(
        source,
        "    def _reset_training():\n",
        "    def _reset_training():\n        _course_cancel_train()\n",
    )
    return source
