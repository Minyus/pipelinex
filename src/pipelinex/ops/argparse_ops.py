from argparse import Namespace


def namespace(d):
    assert isinstance(d, dict)
    return Namespace(**d)


def feed_args_dict(func, args={}, force_return=None):
    assert callable(func)
    args = namespace(args)

    def _feed_args(*argsignore, **kwargsignore):
        output = func(args)
        if force_return is not None:
            return force_return
        return output

    return _feed_args
