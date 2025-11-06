# TSA: Interactive time series assistant with loading, quality metrics, cleaning, preparation, forecasting, and visualization
# Saves as tsa.py with a runnable CLI that starts an interactive session.

import os
import sys
import json
from typing import List, Optional, Dict, Any, Tuple
import pandas as pd
import numpy as np

try:
    from graphviz import Digraph
    _GRAPHVIZ_AVAILABLE = True
except Exception:
    _GRAPHVIZ_AVAILABLE = False


def _prompt_yes_no(msg: str, default_yes: bool = True) -> bool:
    """
    Simple yes/no prompt.
    Enter -> yes (default), Space -> no
    """
    while True:
        try:
            ans = input(msg + (" [Enter=Да, Пробел=Нет] ")).strip()
        except EOFError:
            return default_yes
        if ans == "":
            return default_yes
        if ans == " ":
            return not default_yes
        print("Нажмите Enter для Да или пробел для Нет.")


def _prompt_text(msg: str, default_val: str = "") -> str:
    try:
        ans = input(msg + (" "))
    except EOFError:
        ans = default_val
    ans = ans.strip()
    if ans == "":
        return default_val
    return ans


def _safe_to_numeric(s: pd.Series) -> pd.Series:
    return pd.to_numeric(s, errors="coerce")


class TSA:
    """
    Time Series Assistant: интерактивный помощник анализа, подготовки и базового прогнозирования временного ряда.

    Этапы:
    - Загрузка (с промптом)
    - Выбор столбцов времени и целевого
    - Индексация, сортировка, вывод частоты
    - Метрики качества до
    - Очистка/сегментация/препроцесс по желанию
    - Метрики качества после
    - Разбиение по времени
    - Прогнозы-бенчмарки и метрики качества прогноза
    - Визуализация пайплайна (Graphviz) и ASCII-график ряда с прогнозом
    - Экспорт артефактов
    """

    def __init__(self) -> None:
        self.df_raw: Optional[pd.DataFrame] = None
        self.df: Optional[pd.DataFrame] = None
        self.datetime_col: Optional[str] = None
        self.target_col: Optional[str] = None
        self.freq: Optional[str] = None
        self.tz: Optional[str] = None

        self.config: Dict[str, Any] = {}
        self.history: List[str] = []

        # Pipeline graph
        self.pipeline_nodes: List[Tuple[str, str]] = []
        self.pipeline_edges: List[Tuple[str, str]] = []
        self._node_counter: int = 0

        # Metrics snapshots
        self.metrics_before: Optional[Dict[str, Any]] = None
        self.metrics_after: Optional[Dict[str, Any]] = None

        # Split and forecast
        self.train_index: Optional[pd.DatetimeIndex] = None
        self.test_index: Optional[pd.DatetimeIndex] = None
        self.y_true_test: Optional[pd.Series] = None
        self.y_pred_dict: Dict[str, pd.Series] = {}
        self.forecast_metrics: Optional[pd.DataFrame] = None

    # ---------------- Pipeline helpers ----------------
    def _add_node(self, label: str) -> str:
        self._node_counter += 1
        node_id = "n" + str(self._node_counter)
        self.pipeline_nodes.append((node_id, label))
        if self._node_counter > 1:
            prev_id = "n" + str(self._node_counter - 1)
            self.pipeline_edges.append((prev_id, node_id))
        return node_id

    def _add_edge_custom(self, src_id: str, dst_id: str) -> None:
        self.pipeline_edges.append((src_id, dst_id))

    # ---------------- 1) Loading ----------------
    def load_data_interactive(self) -> None:
        print("Загрузка данных")
        self._add_node("Старт")
        from_file = _prompt_yes_no("Загрузить данные из файла? Иначе ожидать готовый DataFrame в коде.", True)
        if from_file:
            path = _prompt_text("Укажите путь к файлу CSV/JSON/Parquet:")
            if not os.path.exists(path):
                print("Файл не найден. Попробуйте снова, позже можно перезапустить.")
                raise FileNotFoundError("Не найден файл: " + path)
            ext = os.path.splitext(path)[1].lower()
            if ext in [".csv", ".txt"]:
                self.df_raw = pd.read_csv(path)
            elif ext in [".json"]:
                self.df_raw = pd.read_json(path)
            elif ext in [".parquet"]:
                self.df_raw = pd.read_parquet(path)
            else:
                self.df_raw = pd.read_csv(path)
            print("Данные загружены. Формат: " + ext + ", размер: " + str(self.df_raw.shape))
            self._add_node("Загрузка из файла")
        else:
            print("Ожидается, что вы присвоите self.df_raw DataFrame в коде перед запуском следующих шагов.")
            self._add_node("Загрузка из памяти")
        if self.df_raw is None or self.df_raw.shape[0] == 0:
            raise ValueError("Пустые данные после загрузки.")

    # ---------------- 2) Column selection ----------------
    def select_columns_interactive(self) -> None:
        print("Выбор столбцов времени и цели")
        df = self.df_raw.copy()
        candidate_time = []
        for c in df.columns:
            lc = str(c).lower()
            if "date" in lc or "time" in lc or "dt" in lc or "timestamp" in lc:
                candidate_time.append(c)
        if len(candidate_time) == 0:
            candidate_time = [df.columns[0]]
        print("Найдены кандидаты на столбец времени: " + ", ".join([str(x) for x in candidate_time]))
        self.datetime_col = _prompt_text("Введите имя столбца времени (по умолчанию " + str(candidate_time[0]) + "):", str(candidate_time[0]))

        numeric_cols = []
        for c in df.columns:
            if c == self.datetime_col:
                continue
            if pd.api.types.is_numeric_dtype(df[c]):
                numeric_cols.append(c)
        if len(numeric_cols) == 0:
            for c in df.columns:
                if c != self.datetime_col:
                    try:
                        test = pd.to_numeric(df[c], errors="coerce")
                        if test.notna().sum() > 0:
                            numeric_cols.append(c)
                    except Exception:
                        pass
        if len(numeric_cols) == 0:
            raise ValueError("Не найден ни один числовой столбец для цели.")
        print("Числовые кандидаты на цель: " + ", ".join([str(x) for x in numeric_cols]))
        self.target_col = _prompt_text("Введите имя целевого столбца (по умолчанию " + str(numeric_cols[0]) + "):", str(numeric_cols[0]))

        self.df = df[[self.datetime_col, self.target_col]].copy()
        print("Выбраны столбцы: время=" + str(self.datetime_col) + ", цель=" + str(self.target_col))
        self._add_node("Выбор столбцов")

    # ---------------- 3) Index, sort, freq ----------------
    def set_index_sort_infer_freq(self) -> None:
        assert self.df is not None
        print("Подготовка временного индекса и вывод частоты")
        self.df[self.datetime_col] = pd.to_datetime(self.df[self.datetime_col], errors="coerce", utc=False)
        self.df = self.df.dropna(subset=[self.datetime_col])
        self.df = self.df.sort_values(self.datetime_col)
        self.df = self.df.set_index(self.datetime_col)
        self.df.index = pd.DatetimeIndex(self.df.index).tz_localize(None)
        self.df[self.target_col] = _safe_to_numeric(self.df[self.target_col])
        self.df = self.df.dropna(subset=[self.target_col])

        # Deduplicate index
        if self.df.index.has_duplicates:
            how = _prompt_text("Обнаружены дубликаты по времени. Как агрегировать (mean/sum/first)? по умолчанию mean:", "mean").lower()
            if how not in ["mean", "sum", "first"]:
                how = "mean"
            if how == "mean":
                self.df = self.df.groupby(level=0).mean(numeric_only=True)
            elif how == "sum":
                self.df = self.df.groupby(level=0).sum(numeric_only=True)
            else:
                self.df = self.df[~self.df.index.duplicated(keep="first")]

        # Infer frequency
        try:
            self.freq = pd.infer_freq(self.df.index)
        except Exception:
            self.freq = None
        if self.freq is None:
            print("Не удалось вывести частоту автоматически.")
            if _prompt_yes_no("Установить частоту вручную и при необходимости выполнить ресемплинг?", True):
                self.freq = _prompt_text("Введите частоту, например D, H, 15T:", "D").strip()
                agg = _prompt_text("Агрегат при ресемплинге (mean/sum/min/max/median), по умолчанию mean:", "mean").lower()
                before = int(self.df.shape[0])
                if agg == "sum":
                    self.df = self.df.resample(self.freq).sum()
                elif agg == "min":
                    self.df = self.df.resample(self.freq).min()
                elif agg == "max":
                    self.df = self.df.resample(self.freq).max()
                elif agg == "median":
                    self.df = self.df.resample(self.freq).median()
                else:
                    self.df = self.df.resample(self.freq).mean()
                after = int(self.df.shape[0])
                print("Ресемплировано до частоты " + str(self.freq) + ". Было: " + str(before) + ", стало: " + str(after))
                self._add_node("Ресемплирование " + str(self.freq) + " " + str(agg))
        else:
            print("Выведенная частота: " + str(self.freq))
        self._add_node("Индекс и частота")

    # ---------------- 4) Quality metrics ----------------
    def _quality_metrics(self) -> Dict[str, Any]:
        assert self.df is not None
        n = int(self.df.shape[0])
        coverage = None
        step_stats = None
        if n >= 2:
            diffs = np.diff(self.df.index.view(np.int64))
            diffs_sec = diffs / 1e9
            coverage = {
                "start": str(self.df.index.min()),
                "end": str(self.df.index.max()),
                "n_obs": n
            }
            step_stats = {
                "median_step_sec": float(np.nanmedian(diffs_sec)),
                "mean_step_sec": float(np.nanmean(diffs_sec))
            }
        s = _safe_to_numeric(self.df[self.target_col])
        desc = s.describe(percentiles=[0.01, 0.05, 0.25, 0.5, 0.75, 0.95, 0.99]).to_dict()
        nan_ratio = float(s.isna().mean())
        zero_ratio = float((s == 0).mean()) if n > 0 else 0.0
        acf_vals = self._acf(s.values, max_lag=min(40, max(1, n - 1))) if n > 1 else []
        return {
            "coverage": coverage,
            "step_stats": step_stats,
            "describe": desc,
            "nan_ratio": nan_ratio,
            "zero_ratio": zero_ratio,
            "acf": acf_vals
        }
        

    def present_quality_metrics(self, title: str, snap: Dict[str, Any]) -> None:
        print("==== " + str(title) + " ====")
        if snap.get("coverage") is not None:
            cov = snap["coverage"]
            print("Покрытие: c " + str(cov.get("start")) + " по " + str(cov.get("end")) + ", наблюдений: " + str(cov.get("n_obs")))
        if snap.get("step_stats") is not None:
            st = snap["step_stats"]
            print("Шаг медианный, сек: " + str(round(st.get("median_step_sec", 0.0), 4)) + ", средний, сек: " + str(round(st.get("mean_step_sec", 0.0), 4)))
        print("NaN доля: " + str(round(snap.get("nan_ratio", 0.0), 6)) + ", нулей доля: " + str(round(snap.get("zero_ratio", 0.0), 6)))
        d = snap.get("describe", {})
        keys = ["count", "mean", "std", "min", "1%", "5%", "25%", "50%", "75%", "95%", "99%", "max"]
        for k in keys:
            if k in d:
                print(str(k) + ": " + str(d[k]))
        acf_vals = snap.get("acf", [])
        if isinstance(acf_vals, list) and len(acf_vals) > 0:
            acf_str = ", ".join([str(round(v, 3)) for v in acf_vals[:10]])
            print("ACF первые лага: " + acf_str)
        print("==== Конец метрик ====")

    def _acf(self, x: np.ndarray, max_lag: int = 40) -> List[float]:
        x = np.asarray(x, dtype=float)
        mask = np.isfinite(x)
        x = x[mask]
        if x.size < 2:
            return []
        x = x - np.mean(x)
        var = np.var(x)
        if var == 0:
            return [0.0] * max_lag
        res = []
        for lag in range(1, max_lag + 1):
            if lag >= x.size:
                res.append(0.0)
                continue
            c = np.dot(x[:-lag], x[lag:]) / (x.size - lag)
            res.append(float(c / var))
        return res

    # ---------------- 5) Cleaning and preparation ----------------
    def prepare_interactive(self) -> None:
        assert self.df is not None
        print("Подготовка данных: сегментация, пропуски, выбросы, сглаживание, дифференцирование, лаги")
        # Segmentation by date range
        if _prompt_yes_no("Ограничить период данных по дате?", False):
            start = _prompt_text("Дата начала (YYYY-MM-DD) или пусто:", "")
            end = _prompt_text("Дата конца (YYYY-MM-DD) или пусто:", "")
            before = int(self.df.shape[0])
            if start != "":
                try:
                    start_dt = pd.to_datetime(start)
                    self.df = self.df[self.df.index >= start_dt]
                except Exception:
                    pass
            if end != "":
                try:
                    end_dt = pd.to_datetime(end)
                    self.df = self.df[self.df.index <= end_dt]
                except Exception:
                    pass
            after = int(self.df.shape[0])
            print("Сегментация выполнена. Было: " + str(before) + ", стало: " + str(after))
            self._add_node("Сегментация")

        # Optional resample change even if freq known
        if _prompt_yes_no("Изменить частоту ряда с ресемплированием?", False):
            new_freq = _prompt_text("Введите частоту (например D, H, 15T):", "D").strip()
            agg = _prompt_text("Агрегат (mean/sum/min/max/median), по умолчанию mean:", "mean").lower()
            before = int(self.df.shape[0])
            if agg == "sum":
                self.df = self.df.resample(new_freq).sum()
            elif agg == "min":
                self.df = self.df.resample(new_freq).min()
            elif agg == "max":
                self.df = self.df.resample(new_freq).max()
            elif agg == "median":
                self.df = self.df.resample(new_freq).median()
            else:
                self.df = self.df.resample(new_freq).mean()
            self.freq = new_freq
            after = int(self.df.shape[0])
            print("Ресемплирование применено. Было: " + str(before) + ", стало: " + str(after) + ". Частота: " + str(self.freq))
            self._add_node("Ресемплирование " + str(self.freq) + " " + str(agg))

        # Missing values
        s = self.df[self.target_col]
        nans = int(s.isna().sum())
        if nans > 0 and _prompt_yes_no("Обнаружено пропусков: " + str(nans) + ". Заполнить?", True):
            method = _prompt_text("Метод (ffill/bfill/interpolate/mean/median/const/drop):", "interpolate").lower()
            before = int(self.df.shape[0])
            if method == "ffill":
                self.df[self.target_col] = self.df[self.target_col].ffill()
            elif method == "bfill":
                self.df[self.target_col] = self.df[self.target_col].bfill()
            elif method == "interpolate":
                self.df[self.target_col] = self.df[self.target_col].interpolate()
            elif method == "mean":
                self.df[self.target_col] = self.df[self.target_col].fillna(self.df[self.target_col].mean())
            elif method == "median":
                self.df[self.target_col] = self.df[self.target_col].fillna(self.df[self.target_col].median())
            elif method == "const":
                cval_s = _prompt_text("Значение:", "0")
                try:
                    cval = float(cval_s)
                except Exception:
                    cval = 0.0
                self.df[self.target_col] = self.df[self.target_col].fillna(cval)
            elif method == "drop":
                self.df = self.df.dropna(subset=[self.target_col])
            else:
                self.df[self.target_col] = self.df[self.target_col].interpolate()
                method = "interpolate"
            after = int(self.df.shape[0])
            print("Пропуски обработаны методом " + str(method) + ". Осталось NaN: " + str(int(self.df[self.target_col].isna().sum())) + ". Наблюдений: " + str(after) + " (было " + str(before) + ")")
            self._add_node("Пропуски " + str(method))

        # Outliers clip by IQR
        if _prompt_yes_no("Ограничить выбросы (IQR-клиппинг)?", False):
            s = _safe_to_numeric(self.df[self.target_col])
            q1 = s.quantile(0.25)
            q3 = s.quantile(0.75)
            iqr = q3 - q1
            low = q1 - 1.5 * iqr
            high = q3 + 1.5 * iqr
            before = int(((s < low) | (s > high)).sum())
            s2 = s.clip(lower=low, upper=high)
            self.df[self.target_col] = s2
            after = int(((s2 < low) | (s2 > high)).sum())
            print("IQR-клиппинг применён. Потенциальных выбросов до: " + str(before) + ", после: " + str(after))
            self._add_node("IQR-клиппинг")

        # Smoothing
        if _prompt_yes_no("Применить сглаживание (скользящее среднее)?", False):
            w_s = _prompt_text("Окно (целое, например 3):", "3")
            try:
                w = int(w_s)
            except Exception:
                w = 3
            w = max(1, w)
            new_col = self.target_col + "_smooth"
            self.df[new_col] = self.df[self.target_col].rolling(window=w, min_periods=1).mean()
            self.target_col = new_col
            print("Сглаживание применено. Новая целевая: " + str(self.target_col))
            self._add_node("Сглаживание окно " + str(w))

        # Differencing
        if _prompt_yes_no("Выполнить дифференцирование?", False):
            p_s = _prompt_text("Порядок diff (по умолчанию 1):", "1")
            s_s = _prompt_text("Сезонный период (пусто если нет):", "")
            try:
                p = int(p_s) if len(p_s) > 0 else 1
            except Exception:
                p = 1
            seasonal = None
            if len(s_s) > 0:
                try:
                    seasonal = int(s_s)
                except Exception:
                    seasonal = None
            s = _safe_to_numeric(self.df[self.target_col])
            if seasonal is not None and seasonal > 0 and seasonal < s.shape[0]:
                s = s.diff(seasonal)
            if p > 0:
                for _ in range(p):
                    s = s.diff()
            new_col = self.target_col + "_diff"
            self.df[new_col] = s
            self.df = self.df.dropna(subset=[new_col])
            self.target_col = new_col
            print("Дифференцирование применено. Новая целевая: " + str(self.target_col) + ". Осталось наблюдений: " + str(int(self.df.shape[0])))
            self._add_node("Дифференцирование p=" + str(p) + (", s=" + str(seasonal) if seasonal else ""))

        # Lags
        if _prompt_yes_no("Создать лаги?", False):
            lag_s = _prompt_text("Лаги через запятую (например 1,2,7):", "1")
            lags: List[int] = []
            if len(lag_s.strip()) > 0:
                for t in lag_s.split(","):
                    try:
                        lv = int(t.strip())
                        if lv > 0:
                            lags.append(lv)
                    except Exception:
                        pass
            lags = sorted(list(set(lags)))
            before = int(self.df.shape[0])
            for l in lags:
                self.df[self.target_col + "_lag" + str(l)] = _safe_to_numeric(self.df[self.target_col]).shift(l)
            if len(lags) > 0:
                self.df = self.df.dropna()
            after = int(self.df.shape[0])
            print("Создано лагов: " + str(len(lags)) + ". После dropna: " + str(after) + " (было " + str(before) + ")")
            self._add_node("Лаги " + ",".join([str(x) for x in lags]))

        # Снимок метрик после подготовки
        self.metrics_after = self._quality_metrics()
        print("Снимок метрик качества после подготовки сохранен.")
        # ---------------- 6) Презентация и сравнение метрик ----------------
    def present_and_compare_quality(self) -> None:
        if self.metrics_before is not None:
            self.present_quality_metrics("Метрики ДО подготовки", self.metrics_before)
        if self.metrics_after is not None:
            self.present_quality_metrics("Метрики ПОСЛЕ подготовки", self.metrics_after)

        if self.metrics_before is None or self.metrics_after is None:
            print("Для сравнения метрик необходимы оба снимка (до и после).")
            return

        print("==== Сравнение ключевых метрик ====")
        kb = self.metrics_before.get("describe", {})
        ka = self.metrics_after.get("describe", {})
        compare_keys = ["count", "mean", "std", "min", "25%", "50%", "75%", "max"]
        for k in compare_keys:
            if k in kb and k in ka:
                try:
                    vb = float(kb[k])
                    va = float(ka[k])
                    delta = va - vb
                    print(str(k) + ": было " + str(vb) + ", стало " + str(va) + ", дельта " + str(delta))
                except Exception:
                    pass
        nb = float(self.metrics_before.get("nan_ratio", 0.0))
        na = float(self.metrics_after.get("nan_ratio", 0.0))
        print(" доля: было " + str(nb) + ", стало " + str(na) + ", дельта " + str(na - nb))
        zb = float(self.metrics_before.get("zero_ratio", 0.0))
        za = float(self.metrics_after.get("zero_ratio", 0.0))
        print("Нулей доля: было " + str(zb) + ", стало " + str(za) + ", дельта " + str(za - zb))
        print("==== Конец сравнения ====")

    # ---------------- 7) Разбиение train/test ----------------
    def train_test_split_interactive(self) -> None:
        assert self.df is not None and self.target_col is not None
        n = int(self.df.shape[0])
        if n < 5:
            print("Мало наблюдений для разбиения. Весь ряд будет train, тест пуст.")
            self.train_index = self.df.index
            self.test_index = self.df.index[-0:]
            self.y_true_test = pd.Series([], dtype=float)
            return

        print("Разбиение на train/test")
        mode = _prompt_text("Способ разбиения (ratio/date) по умолчанию ratio:", "ratio").lower()
        if mode == "date":
            date_s = _prompt_text("Дата начала теста (YYYY-MM-DD):", "").strip()
            try:
                dt = pd.to_datetime(date_s)
                self.train_index = self.df.index[self.df.index < dt]
                self.test_index = self.df.index[self.df.index >= dt]
            except Exception:
                print("Некорректная дата. Использую ratio.")
                mode = "ratio"

        if mode != "date":
            ratio_s = _prompt_text("Доля теста (например 0.2):", "0.2")
            try:
                ratio = float(ratio_s)
            except Exception:
                ratio = 0.2
            ratio = min(max(ratio, 0.05), 0.9)
            split = int(round(n * (1.0 - ratio)))
            split = max(1, min(split, n - 1))
            self.train_index = self.df.index[:split]
            self.test_index = self.df.index[split:]

        print("Train наблюдений: " + str(len(self.train_index)) + ", Test наблюдений: " + str(len(self.test_index)))
        self.y_true_test = self.df.loc[self.test_index, self.target_col].astype(float)
        self._add_node("Train/Test split")

    # ---------------- 8) Прогнозирование ----------------
    def _metrics_for_forecast(self, y_true: pd.Series, y_pred: pd.Series) -> Dict[str, float]:
        y_true = pd.to_numeric(y_true, errors="coerce")
        y_pred = pd.to_numeric(y_pred, errors="coerce")
        mask = y_true.notna() & y_pred.notna()
        yt = y_true[mask].values.astype(float)
        yp = y_pred[mask].values.astype(float)
        if yt.size == 0:
            return {"MAE": np., "RMSE": np., "MAPE": np.}
        mae = float(np.mean(np.abs(yt - yp)))
        rmse = float(np.sqrt(np.mean((yt - yp) ** 2)))
        denom = np.where(np.abs(yt) < 1e-12, 1.0, np.abs(yt))
        mape = float(np.mean(np.abs((yt - yp) / denom))) * 100.0
        return {"MAE": mae, "RMSE": rmse, "MAPE": mape}

    def forecast_benchmarks(self) -> None:
        assert self.df is not None and self.target_col is not None
        assert self.train_index is not None and self.test_index is not None

        y = self.df[self.target_col].astype(float)
        train = y.loc[self.train_index]
        test = y.loc[self.test_index]

        print("Построение бенчмарк-прогнозов: naive, seasonal_naive, moving_average")

        # Naive (предыдущее значение)
        naive_pred = y.shift(1).loc[self.test_index]
        self.y_pred_dict["naive"] = naive_pred

        # Seasonal naive
        default_season = 7
        if self.freq is not None:
            f = str(self.freq)
            # простая эвристика сезонов
            if f.upper().startswith("D"):
                default_season = 7
            elif f.upper().startswith("H"):
                default_season = 24
            elif f.upper().startswith("T") or f.endswith("T"):
                default_season = 60
            elif f.upper().startswith("W"):
                default_season = 52
            elif f.upper().startswith("MS") or f.upper().startswith("M"):
                default_season = 12
            elif f.upper().startswith("Q"):
                default_season = 4
        seas_s = _prompt_text("Сезон для seasonal naive (по умолчанию " + str(default_season) + "):", str(default_season))
        try:
            season = int(seas_s)
        except Exception:
            season = default_season
        if season < 1:
            season = default_season
        seasonal_naive_pred = y.shift(season).loc[self.test_index]
        self.y_pred_dict["seasonal_naive"] = seasonal_naive_pred

        # Moving average
        ma_win_s = _prompt_text("Окно для moving average (по умолчанию 7):", "7")
        try:
            ma_win = int(ma_win_s)
        except Exception:
            ma_win = 7
        ma_win = max(1, ma_win)
        ma_pred = y.rolling(window=ma_win, min_periods=1).mean().shift(1).loc[self.test_index]
        self.y_pred_dict["moving_average"] = ma_pred

        # Метрики
        rows = []
        for name, pred in self.y_pred_dict.items():
            m = self._metrics_for_forecast(self.y_true_test, pred)
            rows.append({"model": name, "MAE": m["MAE"], "RMSE": m["RMSE"], "MAPE": m["MAPE"]})
            print("Модель " + str(name) + " рассчитана. MAE=" + str(m["MAE"]) + ", RMSE=" + str(m["RMSE"]) + ", MAPE=" + str(m["MAPE"]))
        self.forecast_metrics = pd.DataFrame(rows).set_index("model")
        self._add_node("Benchmark forecasts")

    # ---------------- 9) Визуализация ----------------
    def ascii_plot_with_forecast(self, last_n: int = 200) -> None:
        assert self.df is not None and self.target_col is not None
        series = self.df[self.target_col].astype(float)
        tail_index = series.index[-min(last_n, series.shape[0]):]
        series_tail = series.loc[tail_index]
        preds = {}
        for name, p in self.y_pred_dict.items():
            preds[name] = p.reindex(tail_index)

        all_vals = [series_tail.values]
        for p in preds.values():
            all_vals.append(pd.to_numeric(p, errors="coerce").values)
        all_vals = np.concatenate(all_vals)
        if all_vals.size == 0:
            print("Нет данных для визуализации.")
            return
        vmin = float(np.nanmin(all_vals))
        vmax = float(np.nanmax(all_vals))
        span = vmax - vmin if vmax > vmin else 1.0

        def norm(v):  # 0..50
            return int(round((v - vmin) / span * 50.0))
        
        for t in tail_index:
            val = series_tail.loc[t]
            pos = norm(val)
            line = [\" \"] * 52
            if pos >= 0 and pos < 51:
                line[pos] = \".\"
            for name, char in [(\"naive\", \"n\"), (\"seasonal_naive\", \"s\"), (\"moving_average\", \"m\")]:
                if name in preds and pd.notna(preds[name].loc[t]) if name in preds and t in preds[name].index else False:
                    pv = preds[name].loc[t]
                    if pd.notna(pv):
                        pp = norm(float(pv))
                        if pp >= 0 and pp < 51:
                            line[pp] = char
            print(str(t) + \" |\" + \"\".join(line) + \"|\")\n
    def visualize_pipeline_graph(self, filename_base: str = \"tsa_pipeline\") -> None:
        if not _GRAPHVIZ_AVAILABLE:
            print(\"Graphviz недоступен (модуль не установлен). Устновите для продолжения работы.\")\n            return\n        dot = Digraph(comment=\"TSA Pipeline\", format=\"png\")\n        for nid, label in self.pipeline_nodes:\n            dot.node(nid, label)\n        for a, b in self.pipeline_edges:\n            dot.edge(a, b)\n        dot.render(filename_base, cleanup=True)\n        print(\"Граф пайплайна сохранен: \" + filename_base + \".png\")\n
    
    # ---------------- 10) Экспорт артефактов ----------------
    def export_artifacts(self, base_name: str = "tsa_output") -> None:
    if self.df is None or self.target_col is None:
        print("Нечего экспортировать: df или target_col отсутствуют.")
        return

    # Подготовка каталога экспорта
    ts = pd.Timestamp.utcnow().strftime("%Y%m%d_%H%M%S")
    export_dir = os.path.join(os.getcwd(), base_name + "_export_" + ts)
    try:
        os.makedirs(export_dir, exist_ok=True)
    except Exception as e:
        print("Не удалось создать каталог экспорта: " + str(e))
        export_dir = os.getcwd()
        print("Буду сохранять в текущий каталог: " + str(export_dir))

    # 1) Обработанные данные
    try:
        df_csv_path = os.path.join(export_dir, base_name + "_processed.csv")
        self.df.to_csv(df_csv_path)
        print("Экспортирован обработанный датасет (CSV): " + df_csv_path)
    except Exception as e:
        print("Ошибка при сохранении обработанного датасета (CSV): " + str(e))

    # Дублируем в Parquet при возможности
    try:
        df_parquet_path = os.path.join(export_dir, base_name + "_processed.parquet")
        self.df.to_parquet(df_parquet_path, index=True)
        print("Экспортирован обработанный датасет (Parquet): " + df_parquet_path)
    except Exception as e:
        print("Parquet недоступен, пропускаю: " + str(e))

    # 2) Метрики качества (до/после)
    try:
        qual_path = os.path.join(export_dir, base_name + "_quality.json")
        quality = {"before": self.metrics_before, "after": self.metrics_after}
        with open(qual_path, "w", encoding="utf-8") as f:
            json.dump(quality, f, ensure_ascii=False, indent=2)
        print("Экспортированы метрики качества (JSON): " + qual_path)
    except Exception as e:
        print("Ошибка при сохранении метрик качества: " + str(e))

    # 3) Метрики прогноза
    try:
        if self.forecast_metrics is not None:
            fm_path = os.path.join(export_dir, base_name + "_forecast_metrics.csv")
            self.forecast_metrics.to_csv(fm_path, index=False)
            print("Экспортированы метрики прогноза (CSV): " + fm_path)
        else:
            print("Метрики прогноза отсутствуют, пропускаю.")
    except Exception as e:
        print("Ошибка при сохранении метрик прогноза: " + str(e))

    # 4) Прогнозы по моделям
    try:
        if isinstance(self.y_pred_dict, dict) and len(self.y_pred_dict) > 0:
            for name, ser in self.y_pred_dict.items():
                try:
                    p_path = os.path.join(export_dir, base_name + "_pred_" + str(name) + ".csv")
                    # Приведем к Series/DataFrame с именем столбца
                    if hasattr(ser, "to_frame"):
                        ser_to_save = ser.to_frame(name=str(name))
                        ser_to_save.to_csv(p_path)
                    else:
                        # На всякий случай — конверсия из массива
                        pd.Series(ser, name=str(name)).to_csv(p_path)
                    print("Экспортирован прогноз " + str(name) + ": " + p_path)
                except Exception as pe:
                    print("Ошибка при сохранении прогноза " + str(name) + ": " + str(pe))
        else:
            print("Прогнозы отсутствуют, пропускаю.")
    except Exception as e:
        print("Ошибка при обработке словаря прогнозов: " + str(e))

    # 5) Конфигурация и история
    try:
        cfg_path = os.path.join(export_dir, base_name + "_config.json")
        cfg = {
            "datetime_col": self.datetime_col,
            "target_col": self.target_col,
            "freq": self.freq,
            "tz": self.tz,
            "config": self.config,
            "history": self.history
        }
        with open(cfg_path, "w", encoding="utf-8") as f:
            json.dump(cfg, f, ensure_ascii=False, indent=2)
        print("Экспортированы конфигурация и история: " + cfg_path)
    except Exception as e:
        print("Ошибка при сохранении конфигурации/истории: " + str(e))

    # 6) Граф пайплайна (если доступен graphviz)
    try:
        if '_GRAPHVIZ_AVAILABLE' in globals() and _GRAPHVIZ_AVAILABLE and len(self.pipeline_nodes) > 0:
            try:
                self.visualize_pipeline_graph(os.path.join(export_dir, base_name + "_pipeline"))
            except Exception as ge:
                print("Не удалось сохранить граф пайплайна: " + str(ge))
        else:
            print("Graphviz недоступен или нет узлов пайплайна — пропускаю отрисовку.")
    except Exception as e:
        print("Ошибка при попытке сохранить граф пайплайна: " + str(e))

    print("Экспорт завершен. Папка: " + str(export_dir)))