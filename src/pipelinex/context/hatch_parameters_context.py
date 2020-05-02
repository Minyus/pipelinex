from typing import Any, Dict  # NOQA
from kedro.context import KedroContext
from ..hatch_dict.hatch_dict import HatchDict


class HatchParametersContext(KedroContext):
    @property
    def params(self) -> Dict[str, Any]:
        params = super().params
        params = HatchDict(params)
        return params
