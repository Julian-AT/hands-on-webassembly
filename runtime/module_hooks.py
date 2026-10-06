"""Forward observation hooks used by the original CNN layer catalog.

Matches Module.register_forward_hook's default callback, ordering, output
replacement and removable-handle behavior. Non-default flags are rejected.
"""
import itertools


def install(torch):
    module = torch.nn.Module
    if hasattr(module, 'register_forward_hook'):
        return
    ids = itertools.count()
    original_call = module.__call__

    class Handle:
        def __init__(self, hooks, key):
            self.hooks, self.id = hooks, key
        def remove(self):
            self.hooks.pop(self.id, None)
        def __enter__(self):
            return self
        def __exit__(self, *args):
            self.remove()

    def register(self, hook, *, prepend=False, with_kwargs=False, always_call=False):
        if prepend or with_kwargs or always_call:
            raise NotImplementedError('Only default forward hooks are used by the supplied apps')
        hooks = self.__dict__.setdefault('_course_forward_hooks', {})
        key = next(ids)
        hooks[key] = hook
        return Handle(hooks, key)

    def call(self, *args, **kwargs):
        output = original_call(self, *args, **kwargs)
        for hook in tuple(self.__dict__.get('_course_forward_hooks', {}).values()):
            replacement = hook(self, args, output)
            if replacement is not None:
                output = replacement
        return output

    module.register_forward_hook = register
    module.__call__ = call
