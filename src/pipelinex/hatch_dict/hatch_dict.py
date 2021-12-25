import importlib
from logging import getLogger
from typing import Any, Iterable, List, Union  # NOQA

log = getLogger(__name__)


class HatchDict:
    def __init__(
        self,
        egg,  # type: Union[dict, List]
        lookup={},  # type: dict
        support_nested_keys=True,  # type: bool
        self_lookup_key="$",  # type: str
        support_import=True,  # type: bool
        additional_import_modules=["pipelinex"],  # type: Union[List, str]
        obj_key="=",  # type: str
        eval_parentheses=True,  # type: bool
    ):
        # type: (...) -> None

        assert egg.__class__.__name__ in {"dict", "list"}
        assert lookup.__class__.__name__ in {"dict"}
        assert support_nested_keys.__class__.__name__ in {"bool"}
        assert self_lookup_key.__class__.__name__ in {"str"}
        assert additional_import_modules.__class__.__name__ in {"list", "str"}
        assert obj_key.__class__.__name__ in {"str"}

        aug_egg = {}
        if isinstance(egg, dict):
            if support_nested_keys:
                aug_egg = dot_flatten(egg)
            aug_egg.update(egg)
        self.aug_egg = aug_egg

        self.egg = egg

        self.lookup = {}

        self.lookup.update(_builtin_funcs())
        self.lookup.update(lookup)

        self.self_lookup_key = self_lookup_key
        self.support_import = support_import
        self.additional_import_modules = (
            [additional_import_modules]
            if isinstance(additional_import_modules, str)
            else additional_import_modules or [__name__]
        )
        self.obj_key = obj_key
        self.eval_parentheses = eval_parentheses

        self.warmed_egg = None
        self.snapshot = None

    def get(
        self,
        key=None,  # type: Union[str, int]
        default=None,  # type: Any
        lookup={},  # type: dict
    ):
        # type: (...) -> Any

        assert (key is None) or (
            key.__class__.__name__
            in {
                "str",
                "int",
            }
        ), "Received key: {}".format(key)
        assert lookup.__class__.__name__ in {"dict"}, "Received lookup: s{}".format(
            lookup
        )

        if key is None:
            d = self.egg
        else:
            if isinstance(self.egg, dict):
                d = self.aug_egg.get(key, default)
            if isinstance(self.egg, list):
                assert isinstance(key, int)
                d = self.egg[key] if (0 <= key < len(self.egg)) else default

        if self.self_lookup_key:
            s = dict()
            while d != s:
                d, s = _dfs_apply(
                    d_input=d,
                    hatch_args=dict(lookup=self.aug_egg, obj_key=self.self_lookup_key),
                )
            self.warmed_egg = d

        if self.eval_parentheses:
            d, s = _dfs_apply(
                d_input=d, hatch_args=dict(eval_parentheses=self.eval_parentheses)
            )
            self.warmed_egg = d

        lookup_input = {}
        lookup_input.update(self.lookup)
        lookup_input.update(lookup)

        if isinstance(self.egg, dict):
            forcing_module = self.egg.get("FORCING_MODULE", "")
            module_aliases = self.egg.get("MODULE_ALIASES", {})

        for m in self.additional_import_modules:
            d, s = _dfs_apply(
                d_input=d,
                hatch_args=dict(
                    lookup=lookup_input,
                    support_import=self.support_import,
                    default_module=m,
                    forcing_module=forcing_module,
                    module_aliases=module_aliases,
                    obj_key=self.obj_key,
                ),
            )
        self.snapshot = s
        return d

    def get_params(self):
        return self.snapshot

    def keys(self):
        return self.egg.keys()

    def items(self):
        assert isinstance(self.egg, dict)
        return [(k, self.get(k)) for k in self.egg.keys()]


def _dfs_apply(
    d_input,  # type: Any
    hatch_args,  # type: dict
):
    # type: (...) -> Any

    eval_parentheses = hatch_args.get("eval_parentheses", False)  # type: bool
    lookup = hatch_args.get("lookup", dict())  # type: dict
    support_import = hatch_args.get("support_import", False)  # type: bool
    default_module = hatch_args.get("default_module", "")  # type: str
    forcing_module = hatch_args.get("forcing_module", "")  # type: str
    module_aliases = hatch_args.get("module_aliases", {})  # type: dict
    obj_key = hatch_args.get("obj_key", "=")  # type: str

    d = d_input
    s = d_input

    if isinstance(d_input, dict):

        obj_str = d_input.get(obj_key)

        d, s = {}, {}
        for k, v in d_input.items():
            d[k], s[k] = _dfs_apply(v, hatch_args)

        if obj_str:
            if obj_str in lookup:
                a = lookup.get(obj_str)
                d = _hatch(d, a, obj_key=obj_key)
            elif support_import:
                if forcing_module:
                    obj_path_list = obj_str.rsplit(".", 1)
                    obj_str = "{}.{}".format(forcing_module, obj_path_list[-1])
                if module_aliases:
                    obj_path_list = obj_str.rsplit(".", 1)
                    if len(obj_path_list) == 2 and obj_path_list[0] in module_aliases:
                        module_alias = module_aliases.get(obj_path_list[0])
                        if module_alias is None:
                            obj_path_list.pop(0)
                        else:
                            obj_path_list[0] = module_alias
                        obj_str = ".".join(obj_path_list)
                a = load_obj(obj_str, default_obj_path=default_module)
                d = _hatch(d, a, obj_key=obj_key)

    if isinstance(d_input, list):

        d, s = [], []
        for v in d_input:
            _d, _s = _dfs_apply(v, hatch_args)
            d.append(_d)
            s.append(_s)

    if isinstance(d_input, str):
        if (
            eval_parentheses
            and len(d_input) >= 2
            and d_input[0] == "("
            and d_input[-1] == ")"
        ):
            d = eval(d)

    return d, s


def _hatch(
    d,  # type: dict
    a,  # type: Any
    obj_key="=",  # type: str
    pos_arg_key="_",  # type: str
    attr_key=".",  # type: str
):
    d.pop(obj_key)
    if d:
        assert callable(a), "{} is not callable.".format(a)

        pos_args = d.pop(pos_arg_key, None)
        if pos_args is None:
            pos_args = []
        if not isinstance(pos_args, list):
            pos_args = [pos_args]

        attribute_name = d.pop(attr_key, None)
        for k in d:
            assert isinstance(
                k, str
            ), "Non-string key '{}' in '{}' is not valid for callable: '{}'.".format(
                k, d, a.__name__
            )
        d = a(*pos_args, **d)
        if attribute_name:
            d = getattr(d, attribute_name)
            # if isinstance(d, MethodType):
            #     d = lambda *args: d(args[0])
    else:
        d = a
    return d


def dot_flatten(d):
    try:
        from flatten_dict import flatten

        d = flatten(d, reducer="dot", enumerate_types=(list,))
    except Exception:
        log.warning(
            "{} failed to be flattened. To install dependency, you can run: pip install flatten-dict>=0.3.0".format(
                d
            ),
            exc_info=True,
        )
    return d


def pass_(*argsignore, **kwargsignore):
    return None


def pass_through(*args, **kwargs):
    return args[0] if args else list(kwargs.values())[0] if kwargs else None


class ToPipeline:
    def __init__(self, *args):
        if len(args) == 1:
            args = args[0]
        self.args = args

    def __call__(self):
        return self.args


class Construct:
    def __init__(self, obj):
        self.obj = obj

    def __call__(self, *args, **kwargs):
        return self.obj(*args, **kwargs)


class Method:
    method = None

    def __init__(self, *args, **kwargs):
        if self.method is None:
            self.method = kwargs.pop("method")
        self.args = args
        self.kwargs = kwargs

    def __call__(self, d):
        if isinstance(d, dict):
            d = HatchDict(d)
        attr = getattr(d, self.method, None)
        if callable(attr):
            return attr(*self.args, **self.kwargs)
        else:
            return d


class Get(Method):
    method = "get"


def feed(func, args):
    assert callable(func)
    if isinstance(args, dict):
        posargs = args.pop("_", [])
        kwargs = args
    elif isinstance(args, (list, tuple)):
        posargs = args
        kwargs = dict()
    else:
        posargs = [args]
        kwargs = dict()

    def _feed(*argsignore, **kwargsignore):
        return func(*posargs, **kwargs)

    return _feed


def _builtin_funcs():
    return dict(
        pass_=pass_,
        pass_through=pass_through,
        ToPipeline=ToPipeline,
        Construct=Construct,
        Method=Method,
        Get=Get,
    )


"""
Copyright 2018-2019 QuantumBlack Visual Analytics Limited
regarding `load_obj` function copied from
https://github.com/quantumblacklabs/kedro/blob/0.15.4/kedro/utils.py
"""


def load_obj(obj_path: str, default_obj_path: str = "") -> Any:
    """Extract an object from a given path.
    Args:
        obj_path: Path to an object to be extracted, including the object name.
        default_obj_path: Default object path.
    Returns:
        Extracted object.
    Raises:
        AttributeError: When the object does not have the given named attribute.
    """
    obj_path_list = obj_path.rsplit(".", 1)
    obj_path = obj_path_list.pop(0) if len(obj_path_list) > 1 else default_obj_path
    obj_name = obj_path_list[0]
    module_obj = importlib.import_module(obj_path)
    if not hasattr(module_obj, obj_name):
        raise AttributeError(
            "Object `{}` cannot be loaded from `{}`.".format(obj_name, obj_path)
        )
    return getattr(module_obj, obj_name)
