from typing import Union

import os
from pathlib import Path

import joblib
import pandas as pd
from loguru import logger


def open_file(filepath: Union[str, Path], sep: str = ";"):
    """Function to open files from filepath, either cs or joblib or pkl"""
    filepath = Path(filepath)
    extension = filepath.suffix
    print(extension, filepath)
    if not filepath.exists():
        raise FileNotFoundError(filepath)
    if extension == ".csv":
        f = pd.read_csv(filepath, sep=sep)
    else:
        f = joblib.load(filepath)
    return f


def save_file(file, dirpath: Union[str, Path], file_name: str, replace=False):
    """
    Save file with or without replacing previous versions, in cv or pkl
    input: file: python model or df to save
            path: path to save to
            file_name: name to give to the file, including extension
            replace: False if you do not want to delete and replace previous file with same name
    """
    dirpath = Path(dirpath)
    if not dirpath.exists():
        dirpath.mkdir(parents=True)
    file_name, extension = file_name.split(".")
    if replace:
        try:
            os.remove(dirpath / file_name)
        except OSError:
            pass
    else:
        i = 0
        while True:
            path = dirpath / ".".join((file_name + "_{:d}".format(i), extension))
            if not path.exists():
                break
            i += 1
        file_name += "_{:d}".format(i)
    file_name = ".".join((file_name, extension))
    if extension == "csv":
        file.to_csv(dirpath + file_name, index=False, sep=";", encoding="utf-8")
    elif extension == "joblib":
        joblib.dump(file, dirpath + file_name, compress=1)
    else:
        raise NotImplementedError(f"File type not handled: {extension}")
    logger.info(f"Saved file {file_name} in dir {dirpath}")
