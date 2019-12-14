def dict_of_list_to_list_of_dict(dict_of_list):
    return [
        dict(zip(dict_of_list.keys(), vals)) for vals in zip(*dict_of_list.values())
    ]


def list_of_dict_to_dict_of_list(list_of_dict):
    return {k: [d[k] for d in list_of_dict] for k in list_of_dict[0]}


class DictToDict:
    module = None
    fn = None

    def __init__(self, **kwargs):
        self.kwargs = kwargs
        assert isinstance(self.fn, str)

    def __call__(self, d):
        kwargs = self.kwargs
        if self.module is None:
            fn = eval(self.fn)
        else:
            fn = getattr(self.module, self.fn)
        assert callable(fn)
        assert isinstance(d, dict)
        out = {k: fn(e, **kwargs) for k, e in d.items()}
        return out
