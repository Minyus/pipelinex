from importlib.util import find_spec

if find_spec("kedro"):
    from kedro.io.core import (
        AbstractDataSet,
        AbstractVersionedDataSet,
        DataSetError,
        Version,
    )
else:

    class AbstractDataSet:
        pass

    class AbstractVersionedDataSet:
        def __init__(
            self, filepath=None, version=None, exists_function=None, glob_function=None
        ):
            self._filepath = filepath
            self._version = version
            self._exists_function = exists_function
            self._glob_function = glob_function

    class DataSetError(Exception):
        pass

    class Version:
        pass
