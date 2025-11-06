"""
Модуль тематической кластеризации
─────────────────────────────────
Назначение:
  - Интерактивно собрать несколько файлов с контентом
  - Нормализовать поля: `network`, `source_id`, `content`
  - Очистить пропуски
  - Подобрать лучшую модель (BERTopic / LDA / др.) по метрикам
  - Провести кластеризацию (с перекрёстным отнесением по желанию)
  - Итерироваться со стоп-словами
  - (Опц.) присвоить названия темам
  - Сохранить Excel + JSON с итогами

Зависимости (в requirements модуля):
  pandas, numpy, scikit-learn, bertopic, openpyxl, tqdm, rich (красивый вывод)
"""

from __future__ import annotations

# --- Compatibility bootstrap  ---------------------------------------
import importlib, sys
from types import ModuleType
from dataclasses import dataclass

# 1) гарантируем: list_repo_tree, OfflineModeIsEnabled   (huggingface_hub)
def _patch_huggingface_hub():
    try:
        # если пакет уже загружен, берём его, иначе импортируем «вручную»
        hf = sys.modules.get("huggingface_hub")
        if hf is None:
            hf = importlib.import_module("huggingface_hub")
        # utils submodule
        utils_mod = importlib.import_module("huggingface_hub.utils")

        # OfflineModeIsEnabled
        if not hasattr(utils_mod, "OfflineModeIsEnabled"):
            class OfflineModeIsEnabled(Exception):
                pass
            utils_mod.OfflineModeIsEnabled = OfflineModeIsEnabled

        # list_repo_tree (переместили в _utils.list_repo_files.list_repo_tree)
        if not hasattr(hf, "list_repo_tree"):
            list_mod = importlib.import_module(
                "huggingface_hub._utils.list_repo_files"
            )
            hf.list_repo_tree = list_mod.list_repo_tree  # type: ignore

    except ModuleNotFoundError:
        # huggingface_hub не установлен → позже pip подскажет
        pass

# 2) гарантируем: MODELS_TO_PIPELINE, Transform­­ersKwargs, can_return_tuple (transformers)
def _patch_transformers():
    try:
        doc_mod = importlib.import_module("transformers.utils.doc")
        for missing in (
            "MODELS_TO_PIPELINE",
            "PIPELINE_TASKS_TO_SAMPLE_DOCSTRINGS",
        ):
            if not hasattr(doc_mod, missing):
                setattr(doc_mod, missing, {})

        gen_mod = importlib.import_module("transformers.utils.generic")
        if not hasattr(gen_mod, "TransformersKwargs"):
            @dataclass
            class TransformersKwargs:      # минимальная заглушка
                pass
            gen_mod.TransformersKwargs = TransformersKwargs
        if not hasattr(gen_mod, "can_return_tuple"):
            def can_return_tuple(*_, **__):  # type: ignore
                return False
            gen_mod.can_return_tuple = can_return_tuple
    except ModuleNotFoundError:
        pass

_patch_huggingface_hub()
_patch_transformers()
# ---------------------------------------------------------------------
import json
import sys
from pathlib import Path
from typing import List, Dict, Tuple

import pandas as pd
from tqdm import tqdm
from rich import print as rprint

# -----------------------------------------------------------
# ВСПОМОГАТЕЛЬНЫЕ ФУНКЦИИ
# -----------------------------------------------------------


def _prompt(msg: str, default_yes: bool = True) -> bool:
    """
    Enter  → True   (принять)
    Любая непустая строка → False (отклонить)
    """
    ans = input(msg + " [Enter = Да, любой ввод = Нет] ").strip()
    if ans == "":                       # просто Enter
        return default_yes
    return not default_yes              # набрали хоть что-нибудь

def _ask_column(
    df: pd.DataFrame,
    role: str,
    guesses: Tuple[str, ...],
    allow_missing: bool = False,
) -> str | None:
    """
    Возвращает имя столбца для роли (с учётом allow_missing).

    allow_missing=True  → допускает отсутствие в датасете
                          (возвращает None, если пользователь так решил).
    """
    # 1) авто-угадывание
    for g in guesses:
        if g in df.columns and _prompt(f"Столбец «{g}» считать «{role}»?"):
            return g

    # 2) у пользователя
    if allow_missing and _prompt(
        f"В файле нет столбца «{role}». "
        f"Использовать одно общее значение для всех строк?"
        f" (если «нет», укажите имя существующего столбца)", default_yes=True
    ):
        return None

    while True:
        col = input(f"Как называется столбец с «{role}»? ").strip()
        if col in df.columns:
            return col
        print("Такого столбца нет. Повторите.")

# ── двуязычные стоп-слова (англ + рус) ───────────────────────────────
from sklearn.feature_extraction import text as sk_text

def combined_stopwords() -> set[str]:
    """
    Возвращает объединённый set стоп-слов:
    - английские (встроенные в scikit-learn)
    - русские (из NLTK, докачиваются при первом запуске)
    """
    eng_stop = sk_text.ENGLISH_STOP_WORDS

    # берём русский список
    try:
        from nltk.corpus import stopwords
        ru_stop = set(stopwords.words("russian"))
    except LookupError:
        import nltk
        nltk.download("stopwords", quiet=True)
        from nltk.corpus import stopwords
        ru_stop = set(stopwords.words("russian"))

    return eng_stop.union(ru_stop)


# -----------------------------------------------------------
# ОСНОВНОЙ КЛАСС
# -----------------------------------------------------------

class ThematicClustering:
    NETWORK_GUESSES = ("network", "platform", "site")
    SOURCE_GUESSES = ("videoid", "video_id", "post_id", "id", "source")
    CONTENT_GUESSES = ("text", "transcript", "content", "full_text")

    def __init__(self) -> None:
        self.frames: List[pd.DataFrame] = []
        self.stopwords: set[str] = set()
        self.allow_multi = False
        self.extra_stopwords = set()
        self.topic_labels = None
        self.topic_probs  = None
        self.topic_table  = None
        self.topic_names  = None
        self.vectorizer   = None

       # ---------- ШАГ 1-2 ----------
    def load_datasets(self) -> None:
        """Интерактивный сбор файлов с нормализацией колонок."""
        import os

        while True:
            raw = input("Путь к файлу (Enter = закончить ввод): ").strip()
            if raw == "":
                if self.frames:
                    break         # есть хоть один файл – завершаем цикл
                print("Нужно загрузить хотя бы один файл.")
                continue          # возвращаемся к while

            path = Path(raw).expanduser().resolve()

            # --- проверка пути ---
            if path.is_dir():
                print("Это каталог, а не файл:", path)
                print("Файлы:\n", *os.listdir(path), sep=" • ")
                continue          # обратно к while
            if not path.is_file():
                print("Файл не найден:", path)
                continue

            # --- чтение файла ---
            try:
                if path.suffix.lower() in (".xlsx", ".xls"):
                    df = pd.read_excel(path)
                else:
                    df = pd.read_csv(path, sep=None, engine="python")
            except Exception as e:
                print("Не удалось прочитать:", e)
                continue          # снова спросить путь

            print("✔ строк загружено:", len(df))

            # --- выбор колонок ---
            net_col = _ask_column(
                df, "сеть", self.NETWORK_GUESSES, allow_missing=True
            )
            src_col = _ask_column(
                df, "источник", self.SOURCE_GUESSES, allow_missing=False
            )
            text_col = _ask_column(
                df, "контент", self.CONTENT_GUESSES, allow_missing=False
            )

            # обязательные проверки
            if src_col is None or text_col is None:
                print("Файл пропущен — нет обязательных столбцов.")
                continue          # к следующему файлу

            # --- заполняем / переименовываем ---
            if net_col is None:
                val = input(
                    "Колонки «сеть» нет. "
                    "Какое значение поставить для всех строк? "
                ).strip() or "unknown"
                df["network"] = val
            else:
                df.rename(columns={net_col: "network"}, inplace=True)

            df.rename(columns={src_col: "source_id", text_col: "content"},
                      inplace=True)

            # оставляем только нужное
            self.frames.append(df[["network", "source_id", "content"]])

        # после выхода из while
        self.data = pd.concat(self.frames, ignore_index=True)
        print("\nИтого строк:", len(self.data))
        print(self.data.head())

    # ---------- ШАГ 3 ----------
    def clean(self) -> None:
        before = len(self.data)
        self.data.dropna(subset=["content"], inplace=True)
        after = len(self.data)
        rprint(
            f"[yellow]Удалено {before - after:,} строк без контента.[/yellow]")

    # ---------- ШАГ 4 ----------
    
    def pick_model(self) -> None:
        """
        Мини-бенчмарк LDA, NMF, BERTopic по одной метрике (C_v coherence).
        """
        import re, nltk, gensim
        from sklearn.feature_extraction.text import (
            CountVectorizer, TfidfVectorizer
        )
        from sklearn.decomposition import LatentDirichletAllocation, NMF
        from bertopic import BERTopic
        from gensim.models.coherencemodel import CoherenceModel
        from rich import print as rprint
        rprint(
            f"[yellow]Выбираю лучшую модель[/yellow]")
        texts_raw = self.data["content"].astype(str).tolist()

        # ---------- подготовка токенов ---------------------------------
        tokenizer = re.compile(r"\b\w\w+\b", flags=re.I)
        docs_tok = [[t.lower() for t in tokenizer.findall(t)] for t in texts_raw]
        id2word   = gensim.corpora.Dictionary(docs_tok)
        corpus    = [id2word.doc2bow(d) for d in docs_tok]

        stop = combined_stopwords()

        scores = {}
        topics_dict = {}

        # ---------- LDA ------------------------------------------------
        vec = CountVectorizer(max_df=0.9, stop_words=list(stop))
        X   = vec.fit_transform(texts_raw)

        lda = LatentDirichletAllocation(
            n_components=10, random_state=0, learning_method="batch"
        ).fit(X)

        # топ-10 слов для каждой темы
        lda_topics = [
            [vec.get_feature_names_out()[i] for i in comp.argsort()[-10:][::-1]]
            for comp in lda.components_
        ]
        cm_lda = CoherenceModel(
            topics=lda_topics, texts=docs_tok,
            dictionary=id2word, coherence="c_v"
        )
        scores["LDA"] = cm_lda.get_coherence()
        topics_dict["LDA"] = lda_topics

        # ---------- NMF ------------------------------------------------
        tfidf = TfidfVectorizer(max_df=0.9, stop_words=list(stop))
        X2    = tfidf.fit_transform(texts_raw)

        nmf = NMF(n_components=10, random_state=0).fit(X2)

        nmf_topics = [
            [tfidf.get_feature_names_out()[i] for i in comp.argsort()[-10:][::-1]]
            for comp in nmf.components_
        ]
        cm_nmf = CoherenceModel(
            topics=nmf_topics, texts=docs_tok,
            dictionary=id2word, coherence="c_v"
        )
        scores["NMF"] = cm_nmf.get_coherence()
        topics_dict["NMF"] = nmf_topics

        # ---------- BERTopic ------------------------------------------
        topic_model = BERTopic(
            language="multilingual",
            calculate_probabilities=True,
            verbose=False
        )
        topic_model.fit(texts_raw)

        # собираем топики, игнорируя -1
        bert_topics = [
            [w for w, _ in topic_model.get_topic(tid)[:10]]
            for tid in topic_model.get_topics().keys()      # ids
            if tid != -1 and len(topic_model.get_topic(tid)) > 0
        ]

        if len(bert_topics) >= 2:       # нужно ≥2 тем для осмысленной метрики
            cm_bert = CoherenceModel(
                topics=bert_topics,
                texts=docs_tok,
                dictionary=id2word,
                coherence="c_v"
            )
            scores["BERTopic"] = cm_bert.get_coherence()
        else:
            scores["BERTopic"] = float("-inf")   # или просто не добавлять

        # ---------- вывод результатов ---------------------------------
        rprint("[bold]C_v coherence (↑ лучше)[/bold]")
        for m, sc in scores.items():
            rprint(f"{m:8}: {sc:.4f}")

        self.model_name = max(scores, key=scores.get)
        rprint(f"[bold green]Выбрана модель: {self.model_name}[/bold green]")

   # ---------- ШАГ 5 -------------------------------------------------
    def build_topics(self) -> None:
        # ---------- ШАГ 5. Вопросы перед построением тем -----------------
        from rich import print as rprint
        # 1. Согласен ли пользователь с выбором модели?
        resp = _prompt(
            "Согласны ли вы с выбором модели?"
            "\n   Enter — да, оставить как есть"
            "\n   Любой ввод — выбрать вручную",
            default_yes=True
        )

        if resp == True:                                    # нажали только Enter
            print("✔ Оставляем автоматически выбранную модель.")
        else:                                       # хотят выбрать сами
            while True:
                choice = input(
                    "\nВыберите модель:"
                    "\n   1 — BERTopic"
                    "\n   2 — LDA"
                    "\n   3 — NMF"
                    "\nВаш выбор: "
                ).strip()

                if choice == "1":
                    self.model_name = "BERTopic"
                    print("✅ Вы выбрали BERTopic")
                    break
                elif choice == "2":
                    self.model_name = "LDA"
                    print("✅ Вы выбрали LDA")
                    break
                elif choice == "3":
                    self.model_name = "NMF"
                    print("✅ Вы выбрали NMF")
                    break
                else:
                    print("❌ Неверный ввод — пожалуйста, введите 1, 2 или 3.")

        # 2. soft / hard — только для LDA и BERTopic
        if self.model_name in {"LDA", "BERTopic"}:
            raw = _prompt("Могут ли документы относиться к нескольким темам?", default_yes=True)
            self.allow_multi = (raw is True)       # Enter = да, иначе — нет
        else:
            self.allow_multi = False

        # ── 2. общие заготовки ──────────────────────────────────────
        import numpy as np, re
        from sklearn.feature_extraction.text import (
            CountVectorizer, TfidfVectorizer
        )
        from sklearn.decomposition import LatentDirichletAllocation, NMF
        from sklearn.cluster import KMeans
        from sklearn.metrics import silhouette_score
        from bertopic import BERTopic
        from rich import print as rprint

        texts = self.data["content"].astype(str).tolist()
        stop = list(combined_stopwords() | self.extra_stopwords)
        k_range = range(3, 26)

        model = self.model_name
        soft  = self.allow_multi
        rprint(f"[bold]Строю темы:[/bold] {model} | "
            f"{'soft' if soft else 'hard'}")

        # ── 3. LDA ──────────────────────────────────────────────────
        if model == "LDA":
            self.vectorizer = CountVectorizer(max_df=0.9, stop_words=stop)
            X = self.vectorizer.fit_transform(texts)

            # ищем k с лучшей перплексией
            best_k, best_perp, best_mdl = None, 1e9, None
            for k in k_range:
                mdl = LatentDirichletAllocation(
                    n_components=k, random_state=0
                ).fit(X)
                p = mdl.perplexity(X)
                if p < best_perp:
                    best_k, best_perp, best_mdl = k, p, mdl

            self.model = best_mdl
            theta = best_mdl.transform(X)        # распределение doc–topic
            self.topic_probs = theta

            if soft:                             # ---------- SOFT
                self.topic_labels = [
                    list(np.where(row > 0.05)[0]) for row in theta
                ]
            else:                                # ---------- HARD
                self.topic_labels = theta.argmax(axis=1)

            rprint(f"LDA k={best_k}, perplexity={best_perp:.1f}")

        # ── 4. BERTopic ─────────────────────────────────────────────
        elif model == "BERTopic":
            tm = BERTopic(
                language="multilingual",
                calculate_probabilities=True,
                verbose=False
            )
            labels, probs = tm.fit_transform(texts)

            self.model = tm
            self.topic_probs = probs

            if soft:
                self.topic_labels = [
                    list(np.where(p > 0.05)[0]) for p in probs
                ]
            else:
                self.topic_labels = probs.argmax(axis=1)

            rprint(f"BERTopic итоговых тем: {len(set(labels))}")

        # ── 5.  NMF + KMeans  (всегда hard) ─────────────────────────
        else:                                       # "NMF"
            self.vectorizer = TfidfVectorizer(max_df=0.9, stop_words=stop)
            X = self.vectorizer.fit_transform(texts)

            best_k, best_s, W_best = None, -1, None
            for k in k_range:
                nmf = NMF(n_components=k, random_state=0)
                W = nmf.fit_transform(X)
                s = silhouette_score(W, W.argmax(axis=1))
                if s > best_s:
                    best_k, best_s, W_best = k, s, W
            # после определения best_k и обучения
            nmf_best = NMF(n_components=best_k, random_state=0)
            W_best   = nmf_best.fit_transform(X)

            km = KMeans(n_clusters=best_k, random_state=0, n_init=10)
            self.topic_labels = km.fit_predict(W_best)

            #  —––––– главное! сохраняем объекты
            self.nmf        = nmf_best          # понадобится для top-слов
            self.kmeans     = km                # понадобится только для меток
            self.model_name = "NMF"             # остаётся так же
            rprint(f"NMF+KMeans k={best_k}, silhouette={best_s:.3f}")

        # ── 6. пишем колонку topic в data ───────────────────────────
        if soft:
            self.data["topic"] = [
                ", ".join(map(str, lbls)) if lbls else ""
                for lbls in self.topic_labels
            ]
        else:
            self.data["topic"] = self.topic_labels

        rprint("[green]Кластеризация завершена[/green]")

   # ---------- ШАГ 6. Показ тем ------------------------------------
    def show_topics(self, topn: int = 20) -> None:
        """
        Выводит таблицу «topic | top_words» для результата шага 5.
        Поддерживает ровно три модели: BERTopic, LDA, NMF.
        (soft-/hard-метки на вывод не влияют.)
        """
        import numpy as np, pandas as pd
        from IPython.display import display

        rows = []

        # ── 1. BERTopic ───────────────────────────────────────────
        if self.model_name == "BERTopic":
            topic_ids = sorted(
                t for t in self.model.get_topics().keys() if t != -1
            )
            for tid in topic_ids:
                words = [w for w, _ in self.model.get_topic(tid)[:topn]]
                rows.append({"topic": tid, "top_words": " ".join(words)})

        # ── 2. LDA ────────────────────────────────────────────────
        elif self.model_name == "LDA":
            vocab  = self.vectorizer.get_feature_names_out()
            comps  = self.model.components_               # shape k × |V|
            for tid, row in enumerate(comps):
                idx = row.argsort()[-topn:][::-1]
                rows.append({"topic": tid,
                            "top_words": " ".join(vocab[idx])})

        # ── 3. NMF ────────────────────────────────────────────────
        elif self.model_name == "NMF":
            vocab = self.vectorizer.get_feature_names_out()

            if hasattr(self, "nmf"):
                comps = self.nmf.components_        # ← гарантированно NMF
            else:
                print("⚠️  self.nmf не найден – проверьте шаг 5")
                return

            for tid, row in enumerate(comps):
                idx = row.argsort()[-topn:][::-1]
                rows.append({"topic": tid,
                            "top_words": " ".join(vocab[idx])})

        # ── вывод таблицы ─────────────────────────────────────────
        self.topic_table = pd.DataFrame(rows)
        display(self.topic_table)

    # ---------- ШАГ 7  -------------------------------------------------
    def refine_stopwords(self) -> None:
        """
        1) Показывает таблицу из show_topics().
        2) Спрашивает, встретились ли в топ-словах новые стоп-слова.
        • Если «y» — запрашивает список через запятую
        • Добавляет их в общий список stop-слов
        • Перезапускает build_topics() и show_topics()
        """
        while True:
            ans = _prompt(
            'Есть ли среди топ-слов стоп-слова, которые нужно удалить?',
                default_yes=False
            )
            if ans == 'n':
                break
            extra = input(
                'Введите новые стоп-слова через запятую: '
            ).strip()
            if not extra:
                break

            self.extra_stopwords.update(w.strip() for w in extra.split(',') if w.strip())
            # переобучаем модель с расширенным списком
            self.build_topics()
            self.show_topics()

    # ---------- ШАГ 8  ------------------------------------------------
    def name_topics(self) -> None:
        """
        Интерактивно присваивает человеко-читаемые названия темам.
        После ввода сразу сохраняет их в self.topic_names.
        """

        # 0. Показываем таблицу, чтобы было на что опираться
        from IPython.display import display
        display(self.topic_table)

        # 1. Нужно ли переименовывать?
        if not _prompt('Хотите задать собственные названия темам?', default_yes=True):
            # оставляем «Тема <id>»
            self.topic_names = {t: 'Тема ' + str(t) for t in self.topic_table.topic}
            return

        # 2. Сбор имён
        self.topic_names = {}
        for t, words in zip(self.topic_table.topic, self.topic_table.top_words):
            print('\nTop-слова темы', t, '→', words)
            name = input('Введите название для темы ' + str(t) + ': ').strip()
            self.topic_names[t] = name if name else 'Тема ' + str(t)

        # 3. Итоговая сводка
        import pandas as pd
        renamed = pd.DataFrame({
            'topic': self.topic_table.topic,
            'name':  [self.topic_names[t] for t in self.topic_table.topic],
            'top_words': self.topic_table.top_words
        })
        display(renamed)

   # ---------- ШАГ 9  -------------------------------------------------
    def export(self, outfile: str = "thematic_clusters.xlsx") -> None:
        """
        Сохраняет:
        • Excel-файл
            – лист summary  (topic | top_words | count)
            – по листу на каждую тему  (topic | net? | source? | snippet)
        • JSON-файл со сводкой тем
        """
        import re, json
        from pathlib import Path
        import pandas as pd

        # ---------- 1. сводная таблица --------------------------------
        rows = []
        for tid, words in self.topic_table[["topic", "top_words"]].values:
            if self.allow_multi:
                mask = self.data["topic"].str.contains(rf"\b{tid}\b")
            else:
                mask = self.data["topic"] == tid
            rows.append(
                {
                    "topic": tid,
                    "top_words": words,
                    "count": int(mask.sum()),
                }
            )
        summary = pd.DataFrame(rows)

        writer = pd.ExcelWriter(outfile, engine="openpyxl")
        summary.to_excel(writer, sheet_name="summary", index=False)

        # ---------- 2. документы по темам -----------------------------
        base_cols = ["topic"]
        optional_cols = [c for c in ["network", "source_id"] if c in self.data.columns]
        use_cols = base_cols + optional_cols + ["content"]

        def make_snippet(text, n=20):
            tokens = re.findall(r"\w+", str(text))[:n]
            return " ".join(tokens)

        for _, row in summary.iterrows():
            tid = int(row["topic"])

            # имя листа: берём из self.topic_names, иначе 'T<id>'
            title = str(self.topic_names.get(tid, "")).strip()
            sheet_name = (title if title else "T" + str(tid))[:31]

            # выбираем документы этой темы
            if self.allow_multi:
                mask = self.data["topic"].str.contains(rf"\b{tid}\b")
            else:
                mask = self.data["topic"] == tid
            df_t = self.data.loc[mask, use_cols].copy()
            if df_t.empty:
                continue

            df_t.rename(columns={"content": "snippet"}, inplace=True)
            df_t["snippet"] = df_t["snippet"].apply(make_snippet)
            df_t.to_excel(writer, sheet_name=sheet_name, index=False)

        writer.close()

        # ---------- 3. JSON-файл --------------------------------------
        json_path = Path(outfile).with_suffix(".json")
        with open(json_path, "w", encoding="utf8") as fp:
            json.dump(summary.to_dict(orient="records"), fp,
                    ensure_ascii=False, indent=2)

        rprint(f"[bold green]Экспорт завершён:[/bold green]  "
            f"{outfile}  и  {json_path.name}")
    # ---------- PIPELINE ----------
    def run(self) -> None:
        self.load_datasets()
        self.clean()
        self.pick_model()
        self.build_topics()
        self.show_topics()
        self.refine_stopwords()
        self.name_topics()
        self.export()

# -----------------------------------------------------------
# CLI
# -----------------------------------------------------------


if __name__ == "__main__":
    wizard = ThematicClustering()
    wizard.run()
