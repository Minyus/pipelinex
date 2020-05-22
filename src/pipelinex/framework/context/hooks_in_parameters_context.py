from typing import Any, Dict  # NOQA
from logging import getLogger

from .context import KedroContext
from ...hatch_dict import HatchDict

log = getLogger(__name__)


class HooksInParametersContext(KedroContext):
    _hooks_in_params_read = False

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        if not hasattr(self, "_hook_manager"):

            class Hook:
                def after_catalog_created(self, *args, **kwargs):
                    pass

                def before_node_run(self, *args, **kwargs):
                    pass

                def after_node_run(self, *args, **kwargs):
                    pass

                def on_node_error(self, *args, **kwargs):
                    pass

                def before_pipeline_run(self, *args, **kwargs):
                    pass

                def after_pipeline_run(self, *args, **kwargs):
                    pass

                def on_pipeline_error(self, *args, **kwargs):
                    pass

            hook = Hook()

            class HookManager:
                def __init__(self):
                    self.hook = hook

                def is_registered(self, *args, **kwargs):
                    return False

                def register(self, *args, **kwargs):
                    pass

            self._hook_manager = HookManager()

    @property
    def params(self) -> Dict[str, Any]:
        params = super().params

        if not self._hooks_in_params_read:
            self._hooks_in_params_read = True

            hooks = HatchDict(params).get("HOOKS", [])
            if not isinstance(hooks, (list, tuple)):
                hooks = [hooks]

            for hook in hooks:
                if not self._hook_manager.is_registered(hook):
                    self._hook_manager.register(hook)

        return params
