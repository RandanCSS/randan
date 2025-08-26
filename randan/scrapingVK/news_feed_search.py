import os
import re
import shutil
import time
from datetime import datetime
from typing import Dict, List, Optional, Tuple

import pandas as pd
import requests


class VKAPIConfig:
    API_VERSION = "5.199"
    BASE_URL = "https://api.vk.ru/method/newsfeed.search"
    DEFAULT_COUNT = 200
    DEFAULT_PAUSE = 0.25
    CREDENTIALS_FILE = "credentialsVK.txt"

    def __init__(self):
        self.file_format = ".xlsx"
        self.slash = "\\" if os.name == "nt" else "/"
        self.current_moment = datetime.now()


class VKAPIError(Exception):
    pass


class APIKeyManager:
    def __init__(self, access_token: Optional[str] = None):
        self.keys = self._load_keys(access_token)
        self.current_key_index = 0

    def _load_keys(self, access_token: Optional[str]) -> List[str]:
        if access_token:
            return [access_token]

        if os.path.exists(VKAPIConfig.CREDENTIALS_FILE):
            with open(VKAPIConfig.CREDENTIALS_FILE, "r") as file:
                keys_str = file.read()
        else:
            keys_str = self._get_keys_from_user()
            self._save_keys(keys_str)

        return self._parse_keys(keys_str)

    def _get_keys_from_user(self) -> str:
        while True:
            keys_input = input("Enter API key(s): ").strip()
            if keys_input:
                return keys_input

    def _save_keys(self, keys_str: str) -> None:
        try:
            from randan.tools.textPreprocessor import multispaceCleaner

            keys_str = multispaceCleaner(keys_str)
        except ImportError:
            keys_str = re.sub(r"\s+", " ", keys_str.strip())

        keys_str = keys_str.rstrip(",")

        with open(VKAPIConfig.CREDENTIALS_FILE, "w") as file:
            file.write(keys_str)

    def _parse_keys(self, keys_str: str) -> List[str]:
        keys_str = keys_str.replace(" ", "").replace(",", ", ")
        return keys_str.split(", ")

    def get_current_key(self) -> str:
        return self.keys[self.current_key_index]

    def rotate_key(self) -> str:
        self.current_key_index = (self.current_key_index + 1) % len(self.keys)
        return self.get_current_key()

    def get_key_count(self) -> int:
        return len(self.keys)


class VKAPIClient:
    def __init__(self, key_manager: APIKeyManager):
        self.key_manager = key_manager
        self.pause = VKAPIConfig.DEFAULT_PAUSE
        self.iteration = 0

    def _build_params(self, **kwargs) -> Dict:
        params = {
            "access_token": self.key_manager.get_current_key(),
            "v": VKAPIConfig.API_VERSION,
            "extended": 1,
        }

        optional_params = [
            "count",
            "end_time",
            "fields",
            "latitude",
            "longitude",
            "q",
            "start_from",
            "start_time",
        ]

        for param in optional_params:
            if param in kwargs and kwargs[param] is not None:
                params[param] = kwargs[param]

        return params

    def _handle_api_error(self, error_msg: str) -> bool:
        if "Too many requests per second" in error_msg:
            self.key_manager.rotate_key()
            self.pause += 0.25
            return True

        elif "Unknown application" in error_msg:
            self.key_manager.rotate_key()
            self.pause += 0.25
            return True

        elif "Internal server error" in error_msg:
            return False

        elif "User authorization failed" in error_msg:
            return False

        else:
            return False

    def search_request(self, **kwargs) -> Tuple[pd.DataFrame, bool, Dict]:
        params = self._build_params(**kwargs)

        while True:
            response = requests.get(VKAPIConfig.BASE_URL, params=params)
            response_data = response.json()

            if "response" in response_data:
                response_content = response_data["response"]
                df_add = (
                    pd.json_normalize(response_content["items"])
                    if response_content["items"]
                    else pd.DataFrame()
                )

                if not df_add.empty:
                    df_add = self._process_response_data(
                        df_add, response_content, kwargs.get("fields")
                    )

                self._print_iteration_info(len(response_content["items"]))
                return df_add, True, response_content

            elif "error" in response_data:
                error_msg = response_data["error"]["error_msg"]
                should_continue = self._handle_api_error(error_msg)

                if not should_continue:
                    empty_response = {"items": [], "total_count": 0}
                    return pd.DataFrame(), False, empty_response

                params["access_token"] = self.key_manager.get_current_key()
                time.sleep(self.pause)

            time.sleep(self.pause)

    def _process_response_data(
        self, df: pd.DataFrame, response: Dict, fields: Optional[str]
    ) -> pd.DataFrame:
        df["date"] = df["date"].apply(
            lambda timestamp: datetime.fromtimestamp(timestamp).strftime("%Y.%m.%d")
        )

        df["URL"] = df["from_id"].astype(str)

        user_mask = ~df["URL"].str.contains("-")
        df.loc[user_mask, "URL"] = "id" + df.loc[user_mask, "URL"]

        group_mask = df["URL"].str.contains("-")
        df.loc[group_mask, "URL"] = df.loc[group_mask, "URL"].str.replace("-", "public")

        df["URL"] = (
            "https://vk.com/"
            + df["URL"]
            + "?w="
            + df["inner_type"].str.split("_").str[0]
            + df["owner_id"].astype(str)
            + "_"
            + df["id"].astype(str)
        )

        if fields:
            for fields_column in ["groups", "profiles"]:
                if fields_column in response:
                    df = self._process_fields(df, fields_column, response)

        return df

    def _process_fields(self, df: pd.DataFrame, fields_column: str) -> pd.DataFrame:
        try:
            df[fields_column] = ""
        except Exception:
            df[fields_column] = ""

        return df

    def _print_iteration_info(self, items_count: int) -> None:
        self.iteration += 1


class DataProcessor:
    def __init__(self, config: VKAPIConfig):
        self.config = config

    def merge_dataframes(
        self, df_main: pd.DataFrame, df_add: pd.DataFrame
    ) -> pd.DataFrame:
        if df_add.empty:
            return df_main

        df_combined = pd.concat([df_main, df_add], ignore_index=True)

        id_columns = [col for col in df_combined.columns if "id" in col.lower()]

        if id_columns:
            df_combined = df_combined.drop_duplicates(subset=id_columns, keep="last")

        return df_combined.reset_index(drop=True)

    def save_temporal_data(
        self, df: pd.DataFrame, params: Dict, complicated_name: str
    ) -> None:
        temporal_dir = f"{self.config.current_moment.strftime('%Y%m%d')}{complicated_name}_Temporal"

        if not os.path.exists(temporal_dir):
            os.makedirs(temporal_dir)
            print(f'Created temporal directory "{temporal_dir}"')

        param_files = {
            "method.txt": "newsfeed.search",
            "q.txt": params.get("q", ""),
            "stageTarget.txt": "0",
            "targetCount.txt": str(params.get("target_count", 0)),
            "year.txt": str(params.get("year", datetime.now().year)),
            "yearsRange.txt": params.get("years_range", ""),
        }

        for filename, content in param_files.items():
            filepath = os.path.join(temporal_dir, filename)
            with open(filepath, "w", encoding="utf-8") as f:
                f.write(str(content))

        try:
            df.to_excel(os.path.join(temporal_dir, "data.xlsx"))
        except Exception:
            pass


class TemporalDataLoader:
    def __init__(self, config: VKAPIConfig):
        self.config = config

    def find_temporal_data(self) -> Tuple[Optional[pd.DataFrame], Optional[Dict]]:
        root_contents = os.listdir(".")

        for item in root_contents:
            if "Temporal" in item and os.path.isdir(item):
                if len(os.listdir(item)) == 7:
                    return self._load_temporal_data(item)

        return None, None

    def _load_temporal_data(self, temporal_dir: str) -> Tuple[pd.DataFrame, Dict]:
        try:
            params = {}
            param_files = [
                "targetCount.txt",
                "method.txt",
                "year.txt",
                "q.txt",
                "yearsRange.txt",
                "stageTarget.txt",
            ]

            for param_file in param_files:
                filepath = os.path.join(temporal_dir, param_file)
                if os.path.exists(filepath):
                    with open(filepath, "r", encoding="utf-8") as f:
                        content = f.read().strip()
                        key = param_file.replace(".txt", "")

                        if key in ["targetCount", "year", "stageTarget"]:
                            params[key] = int(content) if content else 0
                        else:
                            params[key] = content if content else None

            data_files = [f for f in os.listdir(temporal_dir) if f.endswith(".xlsx")]
            if data_files:
                df = pd.read_excel(
                    os.path.join(temporal_dir, data_files[0]), index_col=0
                )

                json_files = [
                    f for f in os.listdir(temporal_dir) if f.endswith(".json")
                ]
                if json_files:
                    df_json = pd.read_json(os.path.join(temporal_dir, json_files[0]))
                    df = df.merge(df_json, on="id", how="outer")

                return df, params

        except Exception:
            pass

        return pd.DataFrame(), {}


class UserInputHandler:
    @staticmethod
    def get_search_query() -> Optional[str]:
        query = input().strip()
        return query if query else None

    @staticmethod
    def get_time_range() -> Tuple[Optional[int], Optional[int], Optional[str]]:
        while True:
            time_input = input().strip()

            if not time_input:
                return None, None, None

            time_input = re.sub(r"\s*", "", time_input)

            if "-" not in time_input:
                continue

            years = time_input.split("-")

            if len(years) != 2:
                continue

            try:
                year_min, year_max = int(years[0]), int(years[1])

                if len(years[0]) != 4 or len(years[1]) != 4:
                    continue

                if year_min > year_max:
                    year_min, year_max = year_max, year_min

                return year_min, year_max, time_input

            except ValueError:
                continue

    @staticmethod
    def get_user_files() -> Tuple[Optional[pd.DataFrame], Optional[str]]:
        while True:
            file_input = input().strip()

            if not file_input:
                return None, None

            if os.path.exists(file_input) and file_input.endswith(".xlsx"):
                try:
                    df = pd.read_excel(file_input)
                    folder = os.path.dirname(file_input)
                    return df, folder
                except Exception:
                    pass


class VKNewsfeedSearcher:
    def __init__(self, access_token: Optional[str] = None):
        self.config = VKAPIConfig()
        self.key_manager = APIKeyManager(access_token)
        self.api_client = VKAPIClient(self.key_manager)
        self.data_processor = DataProcessor(self.config)
        self.temporal_loader = TemporalDataLoader(self.config)
        self.input_handler = UserInputHandler()

    def search(
        self,
        count: Optional[int] = None,
        end_time: Optional[int] = None,
        fields: Optional[str] = None,
        latitude: Optional[float] = None,
        longitude: Optional[float] = None,
        q: Optional[str] = None,
        start_time: Optional[int] = None,
        params: Optional[Dict] = None,
        return_dfs: bool = False,
        experienced_mode: bool = False,
    ) -> Optional[pd.DataFrame]:
        if params:
            self._parse_params(params, locals())

        count = count or VKAPIConfig.DEFAULT_COUNT

        if not experienced_mode:
            self._print_instructions()

        temporal_df, temporal_params = self.temporal_loader.find_temporal_data()

        if temporal_df is not None and not temporal_df.empty:
            if self._should_use_temporal_data(temporal_params):
                return self._continue_from_temporal(
                    temporal_df, temporal_params, return_dfs
                )

        if not experienced_mode:
            q = q or self.input_handler.get_search_query()

            if not start_time and not end_time:
                year_min, year_max, years_range = self.input_handler.get_time_range()
                if year_min:
                    start_time = int(datetime(year_min, 1, 1).timestamp())
                if year_max:
                    end_time = int(datetime(year_max, 12, 31).timestamp())

        return self._perform_search(
            count=count,
            end_time=end_time,
            fields=fields,
            latitude=latitude,
            longitude=longitude,
            q=q,
            start_time=start_time,
            return_dfs=return_dfs,
        )

    def _parse_params(self, params: Dict, local_vars: Dict) -> None:
        param_mapping = {
            "access_token": "access_token",
            "q": "q",
            "start_time": "start_time",
            "count": "count",
            "end_time": "end_time",
            "fields": "fields",
            "latitude": "latitude",
            "longitude": "longitude",
        }

        for param_key, local_key in param_mapping.items():
            if param_key in params and local_vars.get(local_key) is None:
                value = params[param_key]
                if param_key in [
                    "start_time",
                    "count",
                    "end_time",
                    "latitude",
                    "longitude",
                ]:
                    try:
                        value = int(value) if isinstance(value, str) else value
                    except ValueError:
                        continue
                local_vars[local_key] = value

    def _should_use_temporal_data(self, temporal_params: Dict) -> bool:
        decision = input().strip()

        if decision == "R":
            for item in os.listdir("."):
                if "Temporal" in item:
                    shutil.rmtree(item, ignore_errors=True)
            return False
        elif decision == " ":
            return False
        else:
            return True

    def _continue_from_temporal(
        self, df: pd.DataFrame, params: Dict, return_dfs: bool
    ) -> Optional[pd.DataFrame]:
        result_df = self._perform_search_continuation(df, params)

        if return_dfs:
            return result_df

        return None

    def _perform_search(self, **kwargs) -> Optional[pd.DataFrame]:
        df_result, success, response = self.api_client.search_request(**kwargs)

        if not success:
            return None

        total_count = response.get("total_count", 0)

        if total_count == 0:
            return pd.DataFrame()

        df_result = self._paginate_search(df_result, response, **kwargs)

        if len(df_result) < total_count:
            df_result = self._segment_by_time_periods(df_result, total_count, **kwargs)

        print(f"Final result: {len(df_result)} objects collected")

        self._save_results(df_result, kwargs.get("q"))

        if kwargs.get("return_dfs", False):
            return df_result

        return None

    def _paginate_search(
        self, df_initial: pd.DataFrame, response: Dict, **kwargs
    ) -> pd.DataFrame:
        df_result = df_initial

        while "next_from" in response:
            kwargs["start_from"] = response["next_from"]
            df_add, success, response = self.api_client.search_request(**kwargs)

            if not success:
                break

            df_result = self.data_processor.merge_dataframes(df_result, df_add)

        return df_result

    def _segment_by_time_periods(
        self, df_initial: pd.DataFrame, **kwargs
    ) -> pd.DataFrame:
        df_result = df_initial
        current_year = datetime.now().year

        for year in range(current_year, 2019, -1):
            year_start = int(datetime(year, 1, 1).timestamp())
            year_end = int(datetime(year, 12, 31).timestamp())

            year_kwargs = kwargs.copy()
            year_kwargs["start_time"] = year_start
            year_kwargs["end_time"] = year_end
            year_kwargs.pop("start_from", None)

            df_year, success, response = self.api_client.search_request(**year_kwargs)

            if not success:
                break

            if df_year.empty:
                continue

            df_year = self._paginate_search(df_year, response, **year_kwargs)
            df_result = self.data_processor.merge_dataframes(df_result, df_year)

            print(f"Year {year}: {len(df_year)} objects added, total: {len(df_result)}")

        return df_result

    def _perform_search_continuation(self, df: pd.DataFrame) -> pd.DataFrame:
        return df

    def _save_results(self, df: pd.DataFrame, query: Optional[str]) -> None:
        timestamp = self.config.current_moment.strftime("%Y%m%d_%H%M")
        filename = f"{timestamp}_VK"

        if query:
            safe_query = re.sub(r"[^\w\s-]", "", query)[:50]
            filename += f"_{safe_query}"

        filename += ".xlsx"

        try:
            df.to_excel(filename)
            print(f"Results saved to {filename}")
        except Exception as e:
            print(f"Error saving results: {e}")


def newsfeed_search(**kwargs) -> Optional[pd.DataFrame]:
    ModuleInstaller.install_missing_modules()

    has_params = any(
        kwargs.get(param) is not None
        for param in [
            "access_token",
            "count",
            "end_time",
            "fields",
            "latitude",
            "longitude",
            "q",
            "start_time",
            "params",
        ]
    ) or kwargs.get("return_dfs", False)

    searcher = VKNewsfeedSearcher(kwargs.get("access_token"))

    return searcher.search(experienced_mode=has_params, **kwargs)


def newsFeedSearch(**kwargs):
    return newsfeed_search(**kwargs)
