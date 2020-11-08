import copy
from pathlib import Path
from typing import Any, Dict
import logging

import torchvision

from pipelinex.hatch_dict.hatch_dict import HatchDict
from ..core import AbstractVersionedDataSet, DataSetError, Version

log = logging.getLogger(__name__)


class IterableImagesDataSet(AbstractVersionedDataSet):
    """Loads a folder containing images as an iterable.
    Wrapper of:
    https://pytorch.org/docs/stable/torchvision/datasets.html#imagefolder
    """

    def __init__(
        self,
        filepath: str,
        load_args: Dict[str, Any] = None,
        save_args: Dict[str, Any] = None,
        version: Version = None,
    ) -> None:
        """
        Args:
            filepath: `root` fed to:
                https://pytorch.org/docs/stable/torchvision/datasets.html#imagefolder
            load_args: Args fed to:
                https://pytorch.org/docs/stable/torchvision/datasets.html#imagefolder
            save_args: Ignored as saving is not supported.
            version: If specified, should be an instance of
                ``kedro.io.core.Version``. If its ``load`` attribute is
                None, the latest version will be loaded.
        """

        super().__init__(
            filepath=Path(filepath), version=version, exists_function=self._exists
        )
        self._load_args = load_args
        self._save_args = save_args

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
