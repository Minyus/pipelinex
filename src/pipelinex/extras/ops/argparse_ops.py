from argparse import Namespace


def namespace(d):
    assert isinstance(d, dict)
    return Namespace(**d)


class FeedArgsDict:
    def __init__(self, func, args={}, force_return=None):
        assert callable(func)
        args = namespace(args)
        self.func = func
        self.args = args
        self.force_return = force_return

    def __call__(self, *args_ignore, **kwargs_ignore):
        func = self.func
        args = self.args
        force_return = self.force_return

        output = func(args)
        if force_return is not None:
            return force_return
        return output
