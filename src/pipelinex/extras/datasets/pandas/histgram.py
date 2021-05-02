import logging
from typing import Any, Dict

import matplotlib.pyplot as plt
import pandas as pd

from ..core import AbstractDataSet

log = logging.getLogger(__name__)


class HistgramDataSet(AbstractDataSet):
    def __init__(
        self,
        filepath: str,
        save_args: Dict[str, Any] = None,
        hist_args: Dict[str, Any] = None,
    ) -> None:
        self._filepath = filepath
        self._save_args = save_args

    def _describe(self) -> Dict[str, Any]:
        return dict(
            filepath=self._filepath,
            save_args=self._save_args,
        )

    def _load(self) -> Any:
        """loading is not supported."""
        return None

    def _save(self, df: pd.DataFrame) -> None:
        save_args = self._save_args
        savefig_args = save_args.pop("savefig_args", {})
        df.hist(**save_args)
        plt.savefig(fname=self._filepath, **savefig_args)
