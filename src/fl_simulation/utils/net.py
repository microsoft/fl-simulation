"""Utilities for working with the net and web."""

import os
from typing import Callable, Optional, Union
from urllib import request

from fl_simulation.logging.logger import get_logger

logger = get_logger()


def download_url(
        url: str,
        destination_filename: Union[str, bytes, "os.PathLike[str]", "os.PathLike[bytes]"],
        progress_updater: Optional[Callable[[int, int, int], None]] = None,
        force_download: bool = False,
        verbose: bool = False,
) -> int:
    """Download the url to the specified location.

    Args:
        url (str): address to download a file from.
        destination_filename (Union[str, bytes, os.PathLike[str], os.PathLike[bytes]]): storage path to store the downloaded file.
        progress_updater (Optional[Callable[[int, int, int], None]], optional): download progress report callback. Defaults to None.
        force_download (bool, optional): whether to replace the destination file if it already exists. Defaults to False.
        verbose (bool, optional): whether to print reporting messages to the console. Defaults to False.

    Raises:
        ConnectionError: raised if the download failed and no file was created at the destination.

    Returns:
        int: size of the downloaded data in bytes.
    """
    if not verbose:
        progress_updater = None

    if not force_download and (os.path.isfile(destination_filename)):
        if verbose:
            logger.debug("Bypassing download of already-downloaded file for \"%s\".", os.path.basename(url))
            print("Bypassing download of already-downloaded file", os.path.basename(url))
        return os.path.getsize(destination_filename)

    if verbose:
        logger.debug("Downloading file \"%s\" to \"%s\".", os.path.basename(url), destination_filename)
        print(
                "Downloading file",
                os.path.basename(url),
                "to",
                destination_filename,
                end="",
        )

    request.urlretrieve(url, destination_filename, progress_updater)

    if not os.path.isfile(destination_filename):
        raise ConnectionError("file was could not be downloaded")

    nBytes = os.path.getsize(destination_filename)

    if verbose:
        print("...done,", nBytes, "bytes.")

    return nBytes
