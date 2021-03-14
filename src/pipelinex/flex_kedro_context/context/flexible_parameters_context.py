from typing import Any, Dict  # NOQA
from importlib import import_module
from logging import getLogger

import kedro
from kedro.pipeline import Pipeline  # NOQA

from pipelinex import HatchDict
from .context import KedroContext, get_hook_manager


log = getLogger(__name__)


class FlexibleParametersContext(KedroContext):
    _params = None

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        params = self.params
        import_modules(params.pop("IMPORT", []))

        params = HatchDict(params).get()
        hooks = params.pop("HOOKS", None)
        if hooks:
            if kedro.__version__.startswith("0.16."):
                self._hooks_in_params_read = False
                self._hook_manager = getattr(self, "_hook_manager", get_hook_manager())
                self._register_kedro_hooks(hooks)
            else:
                log.warning(
                    "HOOKS defined in config file is ignored as the installed kedro version is not 0.16.x"
                )
        self._kedro_pipelines = params.pop("PIPELINES", None)
        self._kedro_run_config = params.pop("RUN_CONFIG", None) or {}
        self._params = params
        log.info("RUN_CONFIG: \n{}".format(self._kedro_run_config))
        log.debug("PIPELINES: \n{}".format(self._kedro_pipelines))
        log.debug("params: \n{}".format(self._params))

    def _register_kedro_hooks(self, hooks):

        if not self._hooks_in_params_read:
            self._hooks_in_params_read = True

            if not isinstance(hooks, (list, tuple)):
                hooks = [hooks]

            for hook in hooks:
                if not self._hook_manager.is_registered(hook):
                    self._hook_manager.register(hook)
                    log.info(
                        "Registered {} with args: {}".format(
                            getattr(getattr(hook, "__class__", None), "__name__", None),
                            getattr(hook, "__dict__", None),
                        )
                    )

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
