from typing import Any, Optional, Union

import os
import pickle
from datetime import datetime
from pathlib import Path

import joblib
import pandas as pd
from loguru import logger


def open_file(filepath: Union[str, os.PathLike], sep: Optional[str] = ";") -> Any:
    """Open csv, pkl or joblib files from filepath

    Args:
        filepath (Union[str, os.PathLike]): filepath to the file to open
        sep (Optional[str], optional): sep in csv if need be. Defaults to ";".

    Raises:
        FileNotFoundError: If filepath does not exists
        NotImplementedError: if filepath suffix not in .csv, .pkl, .joblib

    Returns:
        Any: object opened.
    """
    filepath = Path(filepath)
    suffix = filepath.suffix

    if not filepath.exists():
        raise FileNotFoundError(filepath)

    if suffix == ".csv":
        data = pd.read_csv(filepath, sep=sep)
    elif suffix == ".pkl":
        with open(filepath, "rb") as f:
            data = pickle.load(f)
    elif suffix == ".joblib":
        data = joblib.load(filepath)
    else:
        raise NotImplementedError(f"File type not handled: {suffix}")

    return data


def save_to_file(
    data: Any,
    filepath: Union[str, os.PathLike],
    version: bool = False,
    version_dt_format: str = "%Y%m%d_%H%M",
    overwrite: bool = False,
    makedirs: bool = False,
    sep: Optional[str] = ";",
) -> None:
    """Save to csv, pkl or joblib files.

    Args:
        data (Any): data to be saved.
        filepath (Union[str, os.PathLike]): filepath for the data.
        version (bool, optional): Whether to version the filepath by date using version_dt_format.
            Defaults to False.
        version_dt_format (str, optional): date format to use for date versioning of files.
            Defaults to "%Y%m%d_%H%M".
        overwrite (bool, optional): If the files already exists, whether to replace it or to leave
            it as is. If the file exists and overwrite exists, an FileExistsError is raised.
            Defaults to False.
        makedirs (bool, optional): Whether to create non-existing intermediate directories.
            Defaults to False.
        sep (Optional[str], optional): for csv files, columns separator. Defaults to ";".

    Raises:
        NotADirectoryError: The filepath parent directory does not exist and makedirs is False
        FileExistsError: The filepath already exists and overwrite is False
        NotImplementedError: The suffix of the file is not one of '.csv', '.pkl', '.joblib'.
    """
    filepath = Path(filepath)
    dirpath = filepath.parent
    suffix = filepath.suffix

    if not dirpath.exists():
        if makedirs:
            dirpath.mkdir(parents=True)
        else:
            raise NotADirectoryError(dirpath)

    if version:
        timestamp = datetime.now().strftime(version_dt_format)
        filename = filepath.stem + f"_{timestamp}" + suffix
        filepath = dirpath / filename

    if filepath.exists():
        if overwrite:
            os.remove(filepath)
        else:
            raise FileExistsError(filepath)

    if suffix == ".csv":
        data.to_csv(filepath, index=False, sep=sep, encoding="utf-8")
    elif suffix == ".pkl":
        with open(filepath, "wb") as f:
            pickle.dump(data, f)
    elif suffix == ".joblib":
        joblib.dump(data, filepath, compress=1)
    else:
        raise NotImplementedError(f"File type not handled: {suffix}")

    logger.debug(f"Data saved to {filepath}")

    return filepath
