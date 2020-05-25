from typing import Any, Dict  # NOQA
from logging import getLogger

from .context import KedroContext

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

    def _register_kedro_hooks(self, hooks):

        if not self._hooks_in_params_read:
            self._hooks_in_params_read = True

            if not isinstance(hooks, (list, tuple)):
                hooks = [hooks]

            for hook in hooks:
                if not self._hook_manager.is_registered(hook):
                    self._hook_manager.register(hook)
