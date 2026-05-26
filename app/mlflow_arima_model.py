from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import joblib
import mlflow.pyfunc
import pandas as pd
from pmdarima.arima import ARIMA


class ArimaLiveForecastModel(mlflow.pyfunc.PythonModel):
    """
    MLflow wrapper for a pmdarima ARIMA model.

    Request contract:
    - live_data: list[float] or a JSON string representing that list
    - horizon: optional int, defaults to 5

    Behavior:
    - The model starts with the saved train+validation history.
    - Each request appends new live observations to that history.
    - Forecasts are generated from the updated state.

    Note:
    - State is in-memory inside the serving process. If the container restarts
      or multiple replicas are used, history will not be globally shared.
    """

    def load_context(self, context: mlflow.pyfunc.PythonModelContext) -> None:
        model_path = Path(context.artifacts["model"])
        history_path = Path(context.artifacts["history"])

        self.serving_model: ARIMA = joblib.load(model_path)
        history_df = pd.read_parquet(history_path)

        if history_df.shape[1] != 1:
            raise ValueError("history.parquet must contain exactly one target column.")

        self.target_col = history_df.columns[0]
        self.base_history = history_df[self.target_col].astype(float).reset_index(drop=True)
        self.live_history = pd.Series(dtype=float, name=self.target_col)

    def _coerce_live_data(self, value: Any) -> list[float]:
        if value is None:
            return []
        if isinstance(value, str):
            parsed = json.loads(value)
            if not isinstance(parsed, list):
                raise ValueError("live_data JSON string must decode to a list.")
            return [float(item) for item in parsed]
        if isinstance(value, (list, tuple, pd.Series)):
            return [float(item) for item in value]
        raise ValueError("live_data must be a list-like value or JSON string.")

    def _coerce_bool(self, value: Any, default: bool) -> bool:
        if value is None:
            return default
        if isinstance(value, bool):
            return value
        if isinstance(value, str):
            return value.strip().lower() in {"1", "true", "yes", "y"}
        return bool(value)

    def predict(self, context: mlflow.pyfunc.PythonModelContext, model_input: Any) -> pd.DataFrame:
        if not isinstance(model_input, pd.DataFrame):
            raise ValueError("MLflow pyfunc input must be a pandas DataFrame.")

        outputs: list[dict[str, Any]] = []

        for _, row in model_input.iterrows():
            live_data = self._coerce_live_data(row.get("live_data"))
            horizon = int(row.get("horizon", 5))

            if horizon <= 0:
                raise ValueError("horizon must be greater than 0.")

            if live_data:
                self.serving_model.update(live_data)

            forecast = [float(value) for value in self.serving_model.predict(n_periods=horizon)]

            if live_data:
                self.live_history = pd.concat(
                    [self.live_history, pd.Series(live_data, name=self.target_col)],
                    ignore_index=True
                )

            outputs.append({
                "forecast": forecast,
                "history_size": int(len(self.base_history) + len(self.live_history)),
                "applied_live_points": len(live_data),
                "horizon": horizon,
            })

        return pd.DataFrame(outputs)
