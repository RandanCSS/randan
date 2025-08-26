#transcribe_youtube.py
"""
Использование
-----
from randan.transcribe_youtube import transcribe_youtube
df = transcribe_youtube()  
"""
#Загрузка всего нужного

from pathlib import Path
from importlib import import_module
import typing as _t
import time
import pandas as _pd
from tqdm.auto import tqdm
from youtube_transcript_api import (
    YouTubeTranscriptApi,
    NoTranscriptFound,
    TranscriptsDisabled,
)
from pytube import YouTube
from googleapiclient.discovery import build

MIN_LEN = 100          # минимальная длина «содержательного» текста
THRESHOLD = 0.70       # доля качеств. транскриптов, после которой останавливаемся
SLEEP = 0.2            # пауза между запросами

__all__ = ["transcribe_youtube"]


# ────────────────────────────────────────────────────────────── хелперы
def _prompt(msg: str, default_yes: bool = True) -> bool:
    """
    Enter  -> True  (принять действие)
    Пробел  -> False (отклонить)
    Любое другое -> заново
    """
    while True:
        ans = input(msg + (" [Enter=Да, Пробел=Нет] ")).strip()
        if ans == "":
            return default_yes
        if ans == " ":
            return not default_yes
        print("Нажмите Enter или пробел.")


def _load_ids_from_file(path: str | Path) -> list[str]:
    """
    Читает файл со списком video_id (csv / xlsx / parquet / txt).
    Умеет обрабатывать:
      - ~ (домашний каталог)
      - одинарные обратные слэши из Windows-копипаста
      - относительные и абсолютные пути
    Возвращает уникальный список video_id (str).
    """
    # --- нормализация пути -----------------------------------------------
     # 1. превратим в строку, уберём внешние кавычки, заменим \ на /
    raw = str(path).strip()
    if (raw.startswith('"') and raw.endswith('"')) or \
       (raw.startswith("'") and raw.endswith("'")):
        raw = raw[1:-1]
    raw = raw.replace("\\", "/")

    # 2. создаём Path, раскрываем ~ и получаем абсолютный путь
    path = Path(raw).expanduser().resolve()

    if not path.exists():
        raise FileNotFoundError("Файл не найден: " + str(path))
    
    ext = path.suffix.lower()
    if ext in {".csv", ".txt"}:
        df = _pd.read_csv(path)
    elif ext in {".xlsx", ".xls"}:
        df = _pd.read_excel(path)
    elif ext == ".parquet":
        df = _pd.read_parquet(path)
    else:
        raise ValueError("Поддерживаются csv / xlsx / parquet / txt")
    if "video_id" not in df.columns:
        raise ValueError("Файл должен содержать колонку video_id")
    ids = (
        df["video_id"]
        .astype(str)
        .str.strip()
        .unique()
        .tolist()
    )
    if not ids:
        raise ValueError("Колонка video_id пуста")
    return ids


def _best_transcript(video_id: str, ytb_build, languages=None) -> str | None:
    """Порядок: youtube-transcript-api ➔ captions через Data API ➔ pytube."""
    languages = languages or ["ru", "ru-RU", "en", "en-US", "en-GB"]

    # 1) youtube-transcript-api
    try:
        for lang in languages:
            try:
                ts = YouTubeTranscriptApi.get_transcript(video_id, languages=[lang])
                return " ".join(chunk["text"] for chunk in ts)
            except NoTranscriptFound:
                pass
    except TranscriptsDisabled:
        pass

    # 2) Data API – official captions (требует ключ)
    try:
        caps = (
            ytb_build.captions()
            .list(part="id", videoId=video_id)
            .execute()
            .get("items", [])
        )
        for cap in caps:
            cap_id = cap["id"]
            body = (
                ytb_build.captions()
                .download(id=cap_id, tfmt="ttml")
                .execute()["body"]
            )
            if body:
                # очень грубо избавляемся от XML-тегов
                import re

                text = re.sub("<[^<]+?>", " ", body)
                return text
    except Exception:
        pass

    # 3) pytube auto-captions
    try:
        yt = YouTube("https://www.youtube.com/watch?v=" + video_id)
        caption = yt.captions.get_by_language_code("a.en") or yt.captions.get_by_language_code("a.ru")
        if caption:
            return caption.generate_srt_captions()
    except Exception:
        pass
    return None


def _good(txt: str | None) -> bool:
    return bool(txt) and len(txt.strip()) >= MIN_LEN


# ────────────────────────────────────────────────────────────── тело функции
def transcribe_youtube() -> _pd.DataFrame:
    """
    Полностью интерактивный мастер скачивания транскриптов YouTube.

    Возвращает pandas.DataFrame: channel_id, video_id, title, transcript
    """
    # 1) API-ключ
    api_key = input("Введите YouTube Data API v3 KEY (или Enter, если без него): ").strip()
    ytb_build = build("youtube", "v3", developerKey=api_key) if api_key else None

     # 2) откуда берём список видео
    if _prompt("\nЕсть ли файл со списком video_id?"):
        while True:
            path_str = input("Укажите путь к файлу (начиная с C:, без кавычек): ").strip()
            try:
                video_ids = _load_ids_from_file(Path(path_str))
                break
            except Exception as e:
                print("Ошибка: " + str(e))
    else:
         from randan.scrapingYouTube import searchByText
         searchByText()

    print("Всего найдено видео: ", len(video_ids))
    

    df_out = _pd.DataFrame(columns=["channel_id", "video_id", "title", "transcript"])
    iteration = 1

    while True:
        print("\n── Итерация", iteration, "──────────────────")
        pending = [v for v in video_ids if v not in df_out.video_id.values]
        if not pending:
            print("Больше нечего обрабатывать.")
            break

        rows = []
        for vid in tqdm(pending):
            txt = _best_transcript(vid, ytb_build)
            if _good(txt):
                try:
                    yt = YouTube("https://www.youtube.com/watch?v=" + vid)
                    rows.append(
                        dict(
                            channel_id=yt.channel_id,
                            video_id=vid,
                            title=yt.title,
                            transcript=txt,
                        )
                    )
                except Exception:
                    # если pytube упал, всё равно не теряем текст
                    rows.append(dict(channel_id="", video_id=vid, title="", transcript=txt))
            time.sleep(SLEEP)

        if rows:
            df_out = _pd.concat([df_out, _pd.DataFrame(rows)], ignore_index=True)

        pct = len(df_out) / len(video_ids)
        print(f"Получены содержательные транскрипты для {pct:.1%} видео.")

        out_path = Path(f"transcripts_iter{iteration}.parquet")
        df_out.to_parquet(out_path, index=False)
        print("Файл сохранён: ", out_path)

        if pct >= THRESHOLD:
            print("Достигнут порог " + str(int(THRESHOLD * 100)) + "% – останавливаюсь.")
            break
        if not _prompt("Допарсить ещё?"):
            break
        iteration += 1

    df_out.reset_index(drop=True, inplace=True)
    print("\nГотово! Возвращаю DataFrame.")
    return df_out