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
    parameters: Optional[typer.FileText] = typer.Option(None, "--params, -p"),
    baseline_mode: bool = typer.Option(False, "--baseline/", "-b/"),
    output_dirpath: Optional[Path] = typer.Option(
        None,
        "--output_dirpath, -s",
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
def gridsearch(
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
    parameters_grid: typer.FileText = typer.Argument(None),
    cv: int = typer.Option(5),
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
    if parameters_grid is not None:
        parameters_grid = json.load(parameters_grid)

    if estimator is None:
        raise ValueError("You must either use baseline mode or specify an estimator")

    m = ModelTrainer(
        estimator=estimator,
        scaler=DEFAULT_MODEL_SCALER(),
        baseline_mode=False,
        verbose=True,
    )

    df = pd.read_csv(dataset_filepath)
    m.prepare_train_test_set(df)
    m = m.grid_search(param_grid_estimator=parameters_grid, cv=cv, set_best_model=True, verbose=1)
    metrics = m.evaluate()
    logger.success(f"\n{metrics}")

    if output_dirpath is not None:
        m.save_model(output_dirpath / f"model_gs_{estimator}.pkl")


@app.callback()
def verbose(verbose: bool = typer.Option(True, "--verbose / --no-verbose", "-v / -nv")):
    print(f"app verbose is {verbose}")
    state["verbose"] = verbose


if __name__ == "__main__":
    app()
