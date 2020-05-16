from typing import Dict  # NOQA
from kedro.pipeline import Pipeline  # NOQA
from importlib import import_module
from .hatch_parameters_context import HatchParametersContext


class PipelinesInParametersContext(HatchParametersContext):
    def _get_pipelines(self) -> Dict[str, Pipeline]:
        parameters = self.catalog._data_sets["parameters"].load()
        import_modules(parameters.get("IMPORT"))
        pipelines = parameters.get("PIPELINES")
        assert pipelines
        return pipelines

    def run(self, *args, **kwargs):
        parameters = self.catalog._data_sets["parameters"].load()
        run_dict = parameters.get("RUN_CONFIG", dict())
        run_dict.update(kwargs)
        return super().run(*args, **run_dict)


def import_modules(modules=None):
    if modules:
        if not isinstance(modules, list):
            modules = [modules]
        for module in modules:
            assert isinstance(module, str), "'{}' is not string.".format(module)
            import_module(module)
