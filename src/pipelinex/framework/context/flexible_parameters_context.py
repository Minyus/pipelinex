from typing import Any, Dict  # NOQA
from importlib import import_module
from logging import getLogger

from kedro.pipeline import Pipeline  # NOQA

from pipelinex import HatchDict
from .hooks_in_parameters_context import HooksInParametersContext


log = getLogger(__name__)


class FlexibleParametersContext(HooksInParametersContext):
    _params = None

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        params = self.params
        import_modules(params.pop("IMPORT", []))

        params = HatchDict(params).get()
        self._register_kedro_hooks(params.pop("HOOKS", None) or [])
        self._kedro_pipelines = params.pop("PIPELINES", None)
        self._kedro_run_config = params.pop("RUN_CONFIG", None) or {}
        self._params = params

    @property
    def params(self) -> Dict[str, Any]:
        if self._params is None:
            return super().params
        return self._params

    def _get_pipelines(self) -> Dict[str, Pipeline]:
        assert (
            self._kedro_pipelines
        ), "'PIPELINES' key or value is missing in parameters."
        return self._kedro_pipelines

    def run(self, *args, **kwargs):
        run_dict = self._kedro_run_config
        run_dict.update(kwargs)
        log.info("Run pipeline ({})".format(run_dict))
        return super().run(*args, **run_dict)


def import_modules(modules=None):
    if modules:
        if not isinstance(modules, list):
            modules = [modules]
        for module in modules:
            assert isinstance(module, str), "'{}' is not string.".format(module)
            import_module(module)
