import sys
import os
from pathlib import Path
from typing import Union

from .context.context import KedroContext
from .context.flexible_context import FlexibleContext


def configure_source(project_path: Union[str, Path], source_dir="src"):

    if isinstance(project_path, str):
        project_path = Path(project_path)

    source_path = (project_path / source_dir).resolve()
    source_path_str = str(source_path)

    if source_path_str not in sys.path:
        sys.path.insert(0, source_path_str)

    if "PYTHONPATH" not in os.environ:
        os.environ["PYTHONPATH"] = source_path_str

    return source_path


def load_context(project_path, **kwargs) -> KedroContext:
    return FlexibleContext(project_path, **kwargs)
