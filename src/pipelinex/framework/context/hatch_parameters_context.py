from typing import Any, Dict  # NOQA
from .context import KedroContext
from pipelinex import HatchDict


class HatchParametersContext(KedroContext):
    @property
    def params(self) -> Dict[str, Any]:
        params = super().params
        params = HatchDict(params)
        return params
