"""Локальная проверка качества для двух больших задач урока 9.

Официальные task1_test.csv и task2_test.csv не содержат ответов. Поэтому эти
функции выделяют фиксированную отложенную часть из taskN_train.csv. Обучайте
модель только на первом DataFrame из get_taskN_data(), предсказывайте для
второго и передавайте предсказания в check_taskN().

Пример:
    from capstone_quality import get_task1_data, check_task1
    fit, local_test = get_task1_data()
    # обучите модель на fit и получите предсказания для local_test
    check_task1(predictions)

Вместо массива в порядке local_test можно передать DataFrame с ID и прогнозом.
"""

from functools import lru_cache
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import (accuracy_score, f1_score, log_loss,
                             mean_absolute_error, mean_squared_error,
                             precision_score, r2_score, recall_score)
from sklearn.model_selection import train_test_split

__all__ = ["get_task1_data", "check_task1", "get_task2_data", "check_task2"]

DATA_DIR = Path(__file__).resolve().parent


@lru_cache(maxsize=2)
def _split(task):
    if task == 1:
        data = pd.read_csv(DATA_DIR / "task1_train.csv")
        fit, local_test = train_test_split(data, test_size=0.25, random_state=42)
        return fit, local_test, "flight_id", "fuel_used_liters"
    if task == 2:
        data = pd.read_csv(DATA_DIR / "task2_train.csv")
        fit, local_test = train_test_split(
            data, test_size=0.25, random_state=42, stratify=data["is_authentic"])
        return fit, local_test, "artifact_id", "is_authentic"
    raise ValueError("task должен быть 1 или 2")


def get_task1_data():
    """Вернуть (train с ответом, локальный test без ответа) для регрессии."""
    fit, local_test, _, target = _split(1)
    return fit.copy(), local_test.drop(columns=target).copy()


def get_task2_data():
    """Вернуть (train с ответом, локальный test без ответа) для классификации."""
    fit, local_test, _, target = _split(2)
    return fit.copy(), local_test.drop(columns=target).copy()


def _values(predictions, *, task, value_column):
    _, local_test, id_column, _ = _split(task)
    if isinstance(predictions, pd.DataFrame):
        required = {id_column, value_column}
        if not required.issubset(predictions.columns):
            raise ValueError(f"В DataFrame нужны столбцы {id_column} и {value_column}")
        if predictions[id_column].isna().any() or predictions[id_column].duplicated().any():
            raise ValueError(f"В {id_column} есть пропуски или повторы")
        expected_ids = pd.Index(local_test[id_column])
        given_ids = pd.Index(predictions[id_column])
        if len(given_ids) != len(expected_ids) or not given_ids.isin(expected_ids).all():
            raise ValueError("ID не совпадают с локальным test из get_taskN_data()")
        ordered = predictions.set_index(id_column).loc[expected_ids, value_column]
        raw = ordered.to_numpy()
    else:
        raw = np.asarray(predictions)
        if raw.ndim != 1:
            raise ValueError("Передайте одномерный массив либо DataFrame с ID и прогнозом")
        if len(raw) != len(local_test):
            raise ValueError(f"Нужно {len(local_test)} предсказаний для локального test, получено {len(raw)}")
    try:
        values = np.asarray(raw, dtype=float)
    except (TypeError, ValueError) as exc:
        raise ValueError("Предсказания должны быть числами") from exc
    if not np.isfinite(values).all():
        raise ValueError("В предсказаниях есть NaN или бесконечность")
    return values


def check_task1(predictions):
    """Проверить расход топлива: массив или DataFrame flight_id/fuel_used_liters."""
    fit, local_test, _, target = _split(1)
    predicted = _values(predictions, task=1, value_column=target)
    truth = local_test[target].to_numpy()
    result = {
        "mae": mean_absolute_error(truth, predicted),
        "rmse": np.sqrt(mean_squared_error(truth, predicted)),
        "r2": r2_score(truth, predicted),
        "baseline_mae": mean_absolute_error(truth, np.full(len(truth), fit[target].mean())),
    }
    print(f"Локальный test: {len(truth)} рейсов")
    print(f"MAE: {result['mae']:.2f} л | RMSE: {result['rmse']:.2f} л | R²: {result['r2']:.3f}")
    print(f"MAE постоянного прогноза: {result['baseline_mae']:.2f} л")
    return result


def check_task2(probabilities, threshold=0.5):
    """Проверить подлинность: вероятности класса 1 в массиве или DataFrame.

    DataFrame должен содержать artifact_id и is_authentic_probability.
    """
    if not 0 < threshold < 1:
        raise ValueError("threshold должен быть строго между 0 и 1")
    _, local_test, _, target = _split(2)
    predicted_probability = _values(
        probabilities, task=2, value_column="is_authentic_probability")
    if ((predicted_probability < 0) | (predicted_probability > 1)).any():
        raise ValueError("Вероятности должны лежать между 0 и 1")
    truth = local_test[target].to_numpy()
    predicted_class = (predicted_probability >= threshold).astype(int)
    result = {
        "log_loss": log_loss(truth, predicted_probability),
        "accuracy": accuracy_score(truth, predicted_class),
        "precision": precision_score(truth, predicted_class, zero_division=0),
        "recall": recall_score(truth, predicted_class, zero_division=0),
        "f1": f1_score(truth, predicted_class, zero_division=0),
    }
    print(f"Локальный test: {len(truth)} артефактов | порог: {threshold:g}")
    print(f"Log loss: {result['log_loss']:.3f} | Accuracy: {result['accuracy']:.3f}")
    print(f"Precision: {result['precision']:.3f} | Recall: {result['recall']:.3f} | F1: {result['f1']:.3f}")
    return result
