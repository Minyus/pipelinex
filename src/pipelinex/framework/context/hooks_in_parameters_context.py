from typing import Any, Dict  # NOQA
from logging import getLogger

import kedro

from .context import KedroContext
from ...hatch_dict import HatchDict

log = getLogger(__name__)


class HooksInParametersContext(KedroContext):
    @property
    def params(self) -> Dict[str, Any]:
        params = super().params

        if hasattr(self, "_hook_manager"):

            hooks = HatchDict(params).get("HOOKS", [])
            if not isinstance(hooks, (list, tuple)):
                hooks = [hooks]

            for hook in hooks:
                if not self._hook_manager.is_registered(hook):
                    self._hook_manager.register(hook)
        else:
            log.warning(
                "Hooks are not supported by kedro version: {}".format(kedro.__version__)
            )

        return params
