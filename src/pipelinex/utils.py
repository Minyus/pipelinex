from functools import wraps
from typing import Callable
import logging

log = logging.getLogger(__name__)


def dict_of_list_to_list_of_dict(dict_of_list):
    return [
        dict(zip(dict_of_list.keys(), vals)) for vals in zip(*dict_of_list.values())
    ]


def list_of_dict_to_dict_of_list(list_of_dict):
    return {k: [d[k] for d in list_of_dict] for k in list_of_dict[0]}


def dict_io(func: Callable) -> Callable:
    @wraps(func)
    def _dict_io(*args):
        keys = args[0].keys()

        out_dict = {}
        for key in keys:
            a = [e.get(key) for e in args]
            out = func(*a)
            out_dict[key] = out
            log.debug("{}: {}".format(key, out))

        if isinstance(out_dict[key], tuple):
            return tuple(dict_of_list_to_list_of_dict(out_dict))

        else:
            return out_dict

    return _dict_io


class DictToDict:
    module = None
    fn = None

    def __init__(self, **kwargs):
        self.kwargs = kwargs
        assert isinstance(self.fn, str)

    def __call__(self, *args):
        def _apply_func_to_dicts(*args):

            module = self.module
            fn = self.fn
            kwargs = self.kwargs

            if module is None:
                fn = eval(fn)
            else:
                if isinstance(module, str):
                    module = eval(module)
                fn = getattr(module, fn)
            out = fn(*args, **kwargs)
            return out

        @dict_io
        def apply_func_to_dicts(*args):
            return _apply_func_to_dicts(*args)

        return apply_func_to_dicts(*args)


class ItemGetter:
    def __init__(self, item):
        self.item = item

    def __call__(self, d):
        return d[self.item]


class TransformCompose:
    def __init__(self, transforms):
        self.transforms = transforms

    def __call__(self, d):
        for t in self.transforms:
            d = t(d)
        return d
