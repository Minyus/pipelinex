import copy
from pathlib import Path
from typing import Any, Dict, List, Union
import torchvision
from ...hatch_dict import HatchDict
from kedro.contrib.io import DefaultArgumentsMixIn
from kedro.io.core import AbstractVersionedDataSet, DataSetError, Version
import logging

log = logging.getLogger(__name__)


class IterableImagesDataSet(DefaultArgumentsMixIn, AbstractVersionedDataSet):
    def __init__(
        self,
        filepath: str,
        load_args: Dict[str, Any] = None,
        save_args: Dict[str, Any] = None,
        version: Version = None,
    ) -> None:

        super().__init__(
            filepath=Path(filepath),
            load_args=load_args,
            save_args=save_args,
            version=version,
        )

    def _load(self) -> Any:

        load_path = Path(self._get_load_path())

        load_args = copy.deepcopy(self._load_args)
        load_args = load_args or dict()

        load_args = HatchDict(load_args).get()

        load_args.setdefault("root", load_path)
        load_args.setdefault("transform", torchvision.transforms.ToTensor())

        vision_dataset = torchvision.datasets.ImageFolder(**load_args)

        return vision_dataset

    def _save(self, vision_dataset) -> None:
        """ Not Implemented """
        return None

    def _describe(self) -> Dict[str, Any]:
        return dict(
            filepath=self._filepath,
            load_args=self._save_args,
            save_args=self._save_args,
            version=self._version,
        )

    def _exists(self) -> bool:
        try:
            path = self._get_load_path()
        except DataSetError:
            return False
        return Path(path).exists()
