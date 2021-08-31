from typing import Any, Dict, Optional, Type

from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import LinearRegression
from sklearn.neighbors import KNeighborsRegressor
from sklearn.preprocessing import StandardScaler
from xgboost import XGBRegressor

from litreading.config import DEFAULT_MODEL_ESTIMATOR, DEFAULT_MODEL_SCALER

AvailableEstimatorsCLI = dict(
    default=DEFAULT_MODEL_ESTIMATOR,
    linearregression=LinearRegression,
    randomforest=RandomForestRegressor,
    xgboost=XGBRegressor,
    knn=KNeighborsRegressor,
)

AvailableScalersCLI = dict(default=DEFAULT_MODEL_SCALER, standard=StandardScaler)


class Instanciator:
    def __init__(self, available_objects: Dict[str, Any], object_type_name: str = None) -> None:
        self.available_objects = available_objects
        self.name = "object" if object_type_name is None else object_type_name

    def instanciate(self, object_name: str, parameters: Optional[Dict[str, Any]] = None):
        input_cls = self._parse_input_cls(input_cls_name=object_name)
        parameters = {} if parameters is None else parameters
        return input_cls(**parameters)

    def _parse_input_cls(self, input_cls_name: Optional[str]) -> Type:
        if input_cls_name is None:
            return None

        if not isinstance(input_cls_name, str):
            raise TypeError(f"Please give a string as input. Current input: {input_cls_name}")

        input_cls = self.available_objects.get(input_cls_name)
        if input_cls is None:
            msg = f"This {self.name} is not available: {input_cls_name}."
            msg += f"Please choose among keys in \n{self.available_objects}"
            raise ValueError(msg)

        return input_cls


class EstimatorInit(Instanciator):
    def __init__(self) -> None:
        super().__init__(available_objects=AvailableEstimatorsCLI, object_type_name="estimator")


class ScalerInit(Instanciator):
    def __init__(self) -> None:
        super().__init__(available_objects=AvailableScalersCLI, object_type_name="scaler")
