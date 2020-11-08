import copy
from pathlib import Path
from typing import Any, Dict
import logging

import cv2
from ..core import AbstractVersionedDataSet, DataSetError, Version

log = logging.getLogger(__name__)


class OpenCVImagesLocalDataSet(AbstractVersionedDataSet):
    def __init__(
        self,
        filepath: str,
        load_args: Dict[str, Any] = None,
        save_args: Dict[str, Any] = None,
        channel_first=False,
        version: Version = None,
    ) -> None:

        super().__init__(
            filepath=Path(filepath),
            version=version,
            exists_function=self._exists,
        )
        self._load_args = load_args
        self._save_args = save_args
        self.channel_first = channel_first

    def _load(self) -> Any:

        load_path = Path(self._get_load_path())
        load_args = copy.deepcopy(self._load_args) or dict()
        img = cv2.imread(str(load_path), **load_args)

        return img

    def _save(self, img) -> None:
        save_path = Path(self._get_save_path())
        save_path.parent.mkdir(parents=True, exist_ok=True)
        cv2.imwrite(str(save_path), img)

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
