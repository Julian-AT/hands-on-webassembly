"""Experimental process-only adaptation; does not claim numerical/RNG parity.

PyTorch never calls worker_init_fn when num_workers=0. Removing that callback
is safe in this one configuration. Refuse requests for background workers.
"""


def single_process_loader(torch):
    class DataLoader(torch.utils.data.DataLoader):
        def __init__(self, *args, num_workers=0, worker_init_fn=None, **kwargs):
            if num_workers != 0:
                raise ValueError("The browser adapter supports num_workers=0 only")
            # Keep positional arguments away from the worker settings so nothing
            # is silently overwritten by the adapter.
            if len(args) > 5:
                raise TypeError("Pass worker settings by keyword")
            super().__init__(*args, num_workers=0, worker_init_fn=None, **kwargs)
    return DataLoader


def set_single_thread(count):
    if count != 1:
        raise ValueError("The browser adapter supports exactly one compute thread")
