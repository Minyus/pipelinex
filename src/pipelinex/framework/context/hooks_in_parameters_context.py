from typing import Any, Dict  # NOQA
from logging import getLogger

from .context import KedroContext, get_hook_manager

log = getLogger(__name__)


class HooksInParametersContext(KedroContext):
    _hooks_in_params_read = False

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self._hook_manager = getattr(self, "_hook_manager", get_hook_manager())

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
