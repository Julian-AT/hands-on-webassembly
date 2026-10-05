"""Unit 5 abortable preparation; original synchronous preview bodies retained.

Training calculation isolation is a separate gate. This transformation removes
transport from synchronous constructors and keeps preparation in extended tasks.
"""
def unit5_loading(source,replace_once):
    source=replace_once(source,'from browser_torch import torch\n',
        'from browser_torch import torch\nimport image_preload\n')
    declarations='''    import json as _course_json
    from pathlib import Path as _CoursePath
    _course_loading_config = _course_json.loads(_CoursePath(__file__).with_name('course-loading.json').read_text())
    _course_train_data_owner = image_preload.Owner(_course_loading_config['build_id'], _course_loading_config['source_id'])
    _course_preview_owners = {name: image_preload.Owner(_course_loading_config['build_id'], _course_loading_config['source_id'])
        for name in ('MNIST', 'FashionMNIST')}

    @reactive.extended_task
    async def _course_mnist_preview():
        return await _course_preview_owners['MNIST'].preload('MNIST', manifest_sha256=_course_loading_config['manifest_sha256'])

    @reactive.extended_task
    async def _course_fashion_preview():
        return await _course_preview_owners['FashionMNIST'].preload('FashionMNIST', manifest_sha256=_course_loading_config['manifest_sha256'])

    def _course_end_image_loading():
        _course_train_data_owner.cancel()
        for owner in _course_preview_owners.values():
            owner.cancel()
        _course_mnist_preview.cancel()
        _course_fashion_preview.cancel()

    session.on_ended(_course_end_image_loading)

'''
    source=replace_once(source,'    def _get_loaders(input=input):',declarations+'    def _get_loaders(input=input):')
    for name,task in [('mnist_samples','_course_mnist_preview'),('fashionmnist_samples','_course_fashion_preview')]:
        source=replace_once(source,f'    def {name}():',f'''    def {name}():
        # Demand preparation only when Shiny evaluates this visible preview.
        if {task}.status() == 'initial':
            {task}.invoke()
        {task}.result()''')
    source=replace_once(source,'        try:\n            from types import SimpleNamespace',
        '''        try:
            await _course_train_data_owner.preload('MNIST' if params['inputs']['mn_ds'] == 'mnist' else 'FashionMNIST',
                manifest_sha256=_course_loading_config['manifest_sha256'])
            from types import SimpleNamespace''')
    source=replace_once(source,'    def _mn_train():\n',
        '    def _mn_train():\n        _course_train_data_owner.cancel()\n')
    source=replace_once(source,'    def _mn_reset_on_dataset_change():\n',
        '    def _mn_reset_on_dataset_change():\n        _course_train_data_owner.cancel()\n')
    # The training task's reactive consumer publishes owned errors to progress.
    return source
