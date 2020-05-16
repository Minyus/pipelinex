import sys
import os
from pathlib import Path


def configure_paths(main_mod_path, src_dir_name="src"):
    project_path = Path(main_mod_path).resolve().parent

    src_path = project_path / "src"
    src_path_str = str(src_path)

    if src_path_str not in sys.path:
        sys.path.insert(0, src_path_str)

    if "PYTHONPATH" not in os.environ:
        os.environ["PYTHONPATH"] = src_path_str

    return project_path, src_path
