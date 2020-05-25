from kedro.pipeline import node
from kedro.pipeline import Pipeline
from kedro.utils import load_obj
from typing import Any, Callable, Dict, List, Union  # NOQA


class SubPipeline(Pipeline):
    def __init__(
        self,
        inputs: Union[str, List[str], Dict[str, str]] = None,
        outputs: Union[str, List[str], Dict[str, str]] = None,
        func: Union[Callable, List[Callable]] = None,
        module: str = "",
        decorator: Union[Callable, List[Callable]] = None,
        intermediate_node_name_fmt: str = "{}__{:03d}",
        **kwargs
    ):
        funcs = _load_callables(func, module)

        inputs = inputs or []
        intermediate_base = (
            outputs[0] if (outputs and isinstance(outputs, list)) else outputs
        )
        nodes = []
        for i, f in enumerate(funcs):
            intermediate_flag = i + 1 < len(funcs)
            intermediate = (
                intermediate_node_name_fmt.format(intermediate_base, i + 1)
                if intermediate_flag
                else outputs
            )
            nodes.append(node(func=f, inputs=inputs, outputs=intermediate, **kwargs))
            if intermediate_flag:
                inputs = intermediate

        if decorator:
            decorators = _load_callables(decorator, module)
            nodes = [n.decorate(*decorators) for n in nodes]

        super().__init__(nodes)


def _pass_through(*args, **kwargs):
    return args[0] if args else list(kwargs.values())[0] if kwargs else None


def _load_callables(func, default_module):
    func = func or _pass_through
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
