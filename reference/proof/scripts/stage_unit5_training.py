"""Replace Unit 5's blocking effect with a session-owned coefficient task.

The original arithmetic executes verbatim in unit5_training_source.py. Only
invocation, progress delivery and reactive result consumption change here.
"""
import ast


def unit5_training(source, replace_once):
    tree = ast.parse(source)
    definition = next(n for n in ast.walk(tree)
        if isinstance(n, ast.AsyncFunctionDef) and n.name == '_mn_train')
    names = sorted({n.func.attr for n in ast.walk(definition)
        if isinstance(n, ast.Call) and isinstance(n.func, ast.Attribute)
        and isinstance(n.func.value, ast.Name) and n.func.value.id == 'input'})
    # Loader construction also consumes these inputs; snapshot them at Train.
    names = sorted(set(names) | {'mn_batch', 'mn_ds', 'mn_hflip', 'mn_invert'})
    lines = source.splitlines(keepends=True)
    start = min([definition.lineno] + [d.lineno for d in definition.decorator_list]) - 1
    original = ''.join(lines[start:definition.end_lineno])
    source = replace_once(source, 'import inspection_rng\n',
        'import inspection_rng\nimport isolated_training\n')
    source = replace_once(source, '    def _get_loaders():', '    def _get_loaders(input=input):')
    replacement = f'''
    import queue as _course_queue
    _course_train_q = _course_queue.SimpleQueue()
    _course_training_context = isolated_training.Executor(5)
    _course_train_ticket = reactive.Value(0)
    _course_data_generation = reactive.Value(0)

    class _CourseTrainingError(RuntimeError):
        def __init__(self, ticket, error):
            super().__init__(str(error))
            self.ticket = ticket

    @reactive.extended_task
    async def _mn_train_task(ticket, params):
        try:
            from types import SimpleNamespace
            snapshot = SimpleNamespace(**{{key: (lambda value=value: value) for key, value in params['inputs'].items()}})
            # Construction reseeds global generators. Preserve the UI streams;
            # the original constructors' reseeding is repeated in the worker.
            with inspection_rng.preserve(torch):
                loaders = _get_loaders(snapshot)
            result = await _course_training_context.run(dict(params, loaders=loaders),
                lambda message: _course_train_q.put(dict(ticket=ticket, message=message)))
            return dict(ticket=ticket, **result)
        except Exception as error:
            raise _CourseTrainingError(ticket, error) from error

    def _course_cancel_train():
        _course_train_ticket.set(_course_train_ticket.get() + 1)
        _course_training_context.cancel()
        _mn_train_task.cancel()
        while not _course_train_q.empty():
            _course_train_q.get_nowait()

    @reactive.effect
    @reactive.event(input.mn_train)
    def _mn_train():
        _course_cancel_train()
        ticket = _course_train_ticket.get()
        params = dict(inputs={{name: getattr(input, name)() for name in {tuple(names)!r}}},
            dataset_generation=_course_data_generation.get(), model_generation=ticket)
        mn_coeffs.set(None)
        mn_accuracy.set(None)
        mn_samples.set(None)
        mn_cm_mat.set(None)
        mn_mis.set(None)
        mn_progress_pct.set(0)
        mn_progress_msg.set("Initializing…")
        _mn_train_task.invoke(ticket, params)

    @reactive.effect
    def _course_pump_train():
        reactive.invalidate_later(.05)
        values = dict(mn_progress_pct=mn_progress_pct, mn_progress_msg=mn_progress_msg)
        while not _course_train_q.empty():
            item = _course_train_q.get_nowait()
            if item['ticket'] != _course_train_ticket.get():
                continue
            message = item['message']
            # Completion is committed atomically with coefficients by the
            # reactive consumer, never by a progress message.
            if message.get('name') == 'mn_progress_msg' and message.get('value') == 'Training complete':
                continue
            if message.get('type') == 'value' and message.get('name') in values:
                values[message['name']].set(message['value'])

    @reactive.effect
    def _course_finalize_train():
        if _mn_train_task.status() == 'error':
            error = _mn_train_task.error()
            if getattr(error, 'ticket', None) != _course_train_ticket.get():
                return
            mn_coeffs.set(None)
            mn_progress_pct.set(0)
            mn_progress_msg.set('Error: ' + str(error))
            return
        if _mn_train_task.status() != 'success':
            return
        receipt = _mn_train_task.result()
        if receipt['ticket'] != _course_train_ticket.get():
            return
        while not _course_train_q.empty():
            _course_train_q.get_nowait()
        mn_coeffs.set(receipt['coefficients'])
        mn_progress_pct.set(receipt['progress_pct'])
        mn_progress_msg.set(receipt['progress_msg'])

    def _course_end_training():
        _course_training_context.cancel()
        _mn_train_task.cancel()

    session.on_ended(_course_end_training)
'''
    source = replace_once(source, original, replacement)
    source = replace_once(source, '    def _mn_reset_on_dataset_change():\n',
        '    def _mn_reset_on_dataset_change():\n'
        '        _course_data_generation.set(_course_data_generation.get() + 1)\n'
        '        _course_cancel_train()\n')
    return source
