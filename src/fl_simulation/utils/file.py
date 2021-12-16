"""Utilities for working with the files."""

import zipfile
from typing import Union, Optional, IO
import os


def extract_zip(
        zip_path: Union[str, "os.PathLike[str]", IO[bytes]],
        target_dir: Optional[Union[str, "os.PathLike[str]"]] = None,
):
    """Unpack the specified zip file.

    Args:
        zip_path (Union[str, os.PathLike[str], IO[bytes]]): path to the zip archive.
        target_dir (Optional[Union[str, os.PathLike[str]]], optional): directory to extract files. Defaults to None.
    """
    with zipfile.ZipFile(zip_path, "r") as zip_ref:
        zip_ref.extractall(path=target_dir)