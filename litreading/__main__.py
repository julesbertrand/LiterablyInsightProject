from typing import Optional

import json
import sys
from pathlib import Path

import pandas as pd
import typer
from loguru import logger

from litreading.config import DEFAULT_MODEL_SCALER
from litreading.grader import Grader
from litreading.trainer import ModelTrainer
from litreading.utils.files import save_to_file

app = typer.Typer()
state = {"verbose": True}

logger.configure(handlers=[{"sink": sys.stderr, "level": "INFO"}])


@app.command()
def hello_world(name: str):
    typer.echo(f"Hello {name}")


@app.command()
def grade(
    model_filepath: Path = typer.Argument(
        ...,
        exists=True,
        file_okay=True,
        dir_okay=False,
        writable=False,
        readable=True,
        resolve_path=True,
    ),
    data_filepath: Path = typer.Argument(
        ...,
        exists=True,
        file_okay=True,
        dir_okay=False,
        writable=False,
        readable=True,
        resolve_path=True,
    ),
    output_filepath: Optional[Path] = typer.Option(
        None,
        "-s",
        file_okay=True,
        dir_okay=False,
        writable=True,
        readable=False,
        resolve_path=True,
    ),
):
    df = pd.read_csv(data_filepath)
    grades = Grader(model_filepath=model_filepath).grade(df)
    grades = pd.Series(grades)
    save_to_file(grades, output_filepath, version=False, overwrite=False, makedirs=True)


@app.command()
def train(
    estimator: str = typer.Argument(None),
    dataset_filepath: Path = typer.Argument(
        ...,
        exists=True,
        file_okay=True,
        dir_okay=False,
        writable=False,
        readable=True,
        resolve_path=True,
    ),
    parameters: Optional[typer.FileText] = typer.Option(None),
    baseline_mode: bool = typer.Option(False, "--baseline/", "-b/"),
    output_dirpath: Optional[Path] = typer.Option(
        None,
        "-s",
        file_okay=False,
        dir_okay=True,
        writable=True,
        readable=False,
        resolve_path=True,
    ),
):
    if parameters is not None:
        parameters = json.load(parameters)

    if estimator is None and not baseline_mode:
        raise ValueError("You must either use baseline mode or specify an estimator")

    m = ModelTrainer(
        estimator=estimator,
        scaler=DEFAULT_MODEL_SCALER(),
        baseline_mode=baseline_mode,
        verbose=True,
    )

    df = pd.read_csv(dataset_filepath)
    m.prepare_train_test_set(df)
    m = m.fit()
    metrics = m.evaluate()
    logger.success(f"\n{metrics}")

    if output_dirpath is not None:
        m.save_model(output_dirpath / f"model_{estimator}.pkl")


@app.command()
def set_default_model():
    pass


@app.command()
def get_default_model():
    pass


@app.callback()
def verbose(verbose: bool = typer.Option(True, "--verbose / --no-verbose", "-v / -nv")):
    print(f"app verbose is {verbose}")
    state["verbose"] = verbose


if __name__ == "__main__":
    app()
