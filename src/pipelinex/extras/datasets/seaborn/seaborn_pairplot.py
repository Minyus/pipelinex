import copy
import logging
from pathlib import Path
from typing import Any, Dict, List

import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns

from ..core import AbstractVersionedDataSet, DataSetError, Version

log = logging.getLogger(__name__)


class SeabornPairPlotDataSet(AbstractVersionedDataSet):

    DEFAULT_SAVE_ARGS = dict()  # type: Dict[str, Any]

    def __init__(
        self,
        filepath: str,
        save_args: Dict[str, Any] = None,
        sample_args: Dict[str, Any] = None,
        version: Version = None,
    ) -> None:

        super().__init__(
            filepath=Path(filepath), version=version, exists_function=self._exists
        )
        self._load_args = {}
        self._save_args = save_args
        self._sample_args = sample_args

    def _describe(self) -> Dict[str, Any]:
        return dict(
            filepath=self._filepath,
            save_args=self._save_args,
            sampling_args=self._sample_args,
            version=self._version,
        )

    def _load(self) -> Any:
        """loading is not supported."""
        return None

    def _save(self, data: pd.DataFrame) -> None:
        save_path = Path(self._get_save_path())
        save_path.parent.mkdir(parents=True, exist_ok=True)

        if self._sample_args is not None:
            data = data.sample(**self._sample_args)

        save_args = copy.deepcopy(self._save_args)

        cols_all = data.columns.to_list()

        chunk_size = save_args.pop("chunk_size", None)
        if chunk_size:
            x_vars = save_args.pop("x_vars", None)
            y_vars = save_args.pop("y_vars", None)

            p = Path(save_path)
            p.mkdir(parents=True, exist_ok=True)
            i = 0
            for x in _reshape(x_vars or cols_all, chunk_size):
                for y in _reshape(y_vars or cols_all, chunk_size):
                    log.info(
                        "Generating pairplot: x_vars = {}, y_vars = {}".format(x, y)
                    )
                    try:
                        plt.figure()
                        g = sns.pairplot(data, x_vars=x, y_vars=y, **save_args)
                        plt.suptitle("{} vs {}".format(y, x), va="bottom")
                        s = p / "{}_{:d}{}".format(p.stem, i, p.suffix)
                        i += 1
                        plt.savefig(s)
                        plt.close("all")
                    except Exception:
                        log.error(
                            "Failed to generate pairplot: x_vars = {}, y_vars = {}".format(
                                x, y
                            ),
                            exc_info=True,
                        )
        else:
            plt.figure()
            save_args.setdefault("x_vars", cols_all)
            save_args.setdefault("y_vars", cols_all)
            sns.pairplot(data, **self._save_args)
            plt.savefig(save_path)
            plt.close("all")

    def _exists(self) -> bool:
        try:
            path = self._get_load_path()
        except DataSetError:
            return False
        return Path(path).is_file()


def _reshape(ls: List[Any], size: int) -> List[List[Any]]:
    return [ls[i : i + size] for i in range(0, len(ls), size)]


# def _col_names_from_dtypes(
#     df: pd.DataFrame,
#     include: List[str] =["object"],
# ) -> List[str]:
#     dtypes_dict = df.dtypes.to_dict()
#     return [col for col, dtype in dtypes_dict.items() if dtype.name in include]
