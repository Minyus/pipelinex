from kedro.pipeline import node
from kedro.pipeline import Pipeline
from kedro.utils import load_obj
from typing import Any, Callable, Dict, List, Union  # NOQA


class SubPipeline(Pipeline):
    def __init__(
        self,
        inputs: Union[None, str, List[str], Dict[str, str]],
        outputs: Union[None, str, List[str], Dict[str, str]],
        *,
        func: Union[Callable, List[Callable]] = None,
        main_input_index: int = 0,
        module: str = "",
        decorator: Union[Callable, List[Callable]] = [],
        name: str = None
    ):
        funcs = _load_callables(func, module)

        if isinstance(inputs, str):
            inputs = [inputs]
        if inputs:
            main_input = inputs[main_input_index]
        else:
            inputs = []
            main_input = None
        nodes = []
        for i, f in enumerate(funcs):
            output = (
                "{}__{:03d}".format(main_input, i + 1)
                if (i + 1 < len(funcs))
                else outputs
            )
            nodes.append(node(func=f, inputs=inputs[:], outputs=output))
            if i + 1 < len(funcs):
                inputs[main_input_index] = output

        if decorator:
            decorators = _load_callables(decorator, module)
            nodes = [n.decorate(*decorators) for n in nodes]

        super().__init__(nodes=nodes, name=name)


def _load_callables(func, default_module):
    func = func or (
        lambda *args, **kwargs: (
            args[0] if args else list(kwargs.values())[0] if kwargs else None
        )
    )
    funcs = func if isinstance(func, list) else [func]

    for f in funcs:
        if isinstance(f, str):
            f_list = f.rsplit(".", 1)
            obj = f_list[-1]
            module = f_list[0] if len(f_list) == 2 else None
            assert module or default_module, (
                "The module to which '{}' belongs is unknown. ".format(obj)
                + "Specify the module (e.g. foo.bar) using the name format"
                " (e.g. 'foo.bar.{}') ".format(obj) + "or default_module argument."
            )
        else:
            assert callable(f), "{} should be callable or str.".format(f)

    funcs = [
        f
        if callable(f)
        else load_obj(f, default_obj_path=default_module)
        if isinstance(f, str)
        else None
        for f in funcs
    ]

    return funcs
