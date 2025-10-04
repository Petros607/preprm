import argparse
import datetime
import logging
from pathlib import Path
from typing import Any

import config
from llm.llm_client import LlmClient
from llm.perp_client import PerplexityClient
from logger import setup_logging
from utils.db import DatabaseManager
from utils.md_exporter import MarkdownExporter

setup_logging(level=logging.INFO)
logger = logging.getLogger(__name__)


def clean_and_create_db() -> None:
    """
    Очистка исходной базы данных и создание новых таблиц:
    - cleaned_table_name
    - result_table_name
    """
    logger.info("Начинаем очистку базы данных и создание новых таблиц.")
    db = DatabaseManager()
    db.create_cleaned_table(
        source_table_name=config.source_table_name,
        new_table_name=config.cleaned_table_name
    )
    db.create_result_table(
        source_table_name=config.cleaned_table_name,
        result_table_name=config.result_table_name,
        drop_table=True
    )
    db.close()
    logger.info("База данных успешно подготовлена.")


def transform_record_to_chunk(data: list[dict[str, Any]]) -> dict[int, dict[str, str]]:
    """
    Преобразует список записей из базы данных в словарь для пакетной обработки LLM.
    :param data: список записей из базы
    :return: словарь {индекс: запись} с ключами person_id, first_name, last_name, about
    """
    result: dict[int, dict[str, str]] = {}
    for index, row in enumerate(data):
        result[index] = {
            "person_id": row.get('person_id'),
            "first_name": row.get('first_name') or "",
            "last_name": row.get('last_name') or "",
            "about": row.get('about') or ""
        }
    return result


def safe_execute(db: DatabaseManager, query: str, params: tuple[Any, ...]) -> bool:
    """
    Выполняет SQL-запрос и логирует ошибки.
    :param db: объект DatabaseManager
    :param query: SQL-запрос
    :param params: параметры для запроса
    :return: True, если была изменена хотя бы одна строка
    """
    try:
        result = db.execute_query(query, params)
        affected = result[0].get('affected_rows', 0) if result else 0
        return affected > 0
    except Exception as e:
        logger.error(f"Ошибка при выполнении запроса: {e}")
        return False


def process_llm_batch(db: DatabaseManager, chunk: dict[int, dict[str, str]],
                      update_query: str, llm: LlmClient
                      ) -> int:
    """
    Обрабатывает один батч записей через LLM и обновляет базу.
    :param db: объект DatabaseManager
    :param chunk: батч записей
    :param update_query: SQL для обновления
    :param llm: объект LlmClient
    :return: количество успешно обновлённых строк
    """
    parsed_chunk = llm.parse_names_and_about_chunk(chunk)
    count_of_affected_rows = 0

    if not parsed_chunk:
        logger.warning("Батч не обработался ИИ!")
        return 0

    for data in parsed_chunk.values():
        params = (
            data.get('meaningful_first_name', ''),
            data.get('meaningful_last_name', ''),
            data.get('meaningful_about', ''),
            bool(data.get('meaningful_first_name')
                 and data.get('meaningful_last_name')
                 and data.get('meaningful_about')
            ),
            data.get('person_id')
        )
        if safe_execute(db, update_query, params):
            count_of_affected_rows += 1
        else:
            logger.info(f"БД не изменила строку с person_id {data.get('person_id')}")

    return count_of_affected_rows


def test_llm(start_position: int, row_count: int) -> None:
    """
    Обрабатывает записи из result_table_name через LLM и обновляет базу данных.
    """
    logger.info("Начинаем обработку записей через LLM.")
    db = DatabaseManager()
    query: str = f"SELECT * FROM {config.result_table_name} ORDER BY person_id"
    if 0 < row_count: query += f" LIMIT {row_count}"
    if 0 < start_position: query += f" OFFSET {start_position}"

    records = db.execute_query(query)
    total = len(records)
    llm = LlmClient()
    CHUNK_SIZE: int = config.CHUNK_SIZE
    i = 0
    retry = 0

    update_query = f"UPDATE {config.result_table_name} \
        SET meaningful_first_name = %s, meaningful_last_name = %s, \
            meaningful_about = %s, valid = %s WHERE person_id = %s"

    while i < total:
        record_chunk = records[i:i + CHUNK_SIZE]
        chunk = transform_record_to_chunk(record_chunk)
        count_of_affected_rows = process_llm_batch(db, chunk, update_query, llm)

        if count_of_affected_rows < CHUNK_SIZE:
            logger.warning(f"Батч с {i} по {i + CHUNK_SIZE} не обработался ИИ! "
                           f"Попытка №{retry}"
            )
            retry += 1
            if retry > 3: retry = 0
        else: retry = 0

        if retry == 0:
            i += CHUNK_SIZE
            logger.info(f"Обработано {min(i, total)} / {total} записей, "
                        f"БД изменила {count_of_affected_rows} строк"
            )
    db.close()
    logger.info("Обработка LLM завершена.")


def process_person_md(db: DatabaseManager, person: dict[str, Any],
                      update_query: str, exporter: MarkdownExporter,
                      md_flag: bool, perp: PerplexityClient
    ) -> int | None:
    """
    Обрабатывает одного человека: поиск через Perplexity и экспорт в Markdown.
    """
    if not person.get("valid"):
        return

    ans, urls = perp.search_info(
        first_name=person.get("meaningful_first_name", ""),
        last_name=person.get("meaningful_last_name", ""),
        additional_info=person.get("meaningful_about", ""),
        personal_channel_name=person.get("personal_channel_username"),
        personal_channel_about=person.get("personal_channel_about"),
        temperature=0.1
    )

    safe_execute(db, update_query, (ans, urls, person.get('person_id')))

    if md_flag:
        export_to_md(person, exporter, ans, urls)

    return person.get('person_id')


def export_to_md(person: dict[str, Any], exporter: MarkdownExporter,
                 ans, urls
    ) -> str | None:
    """
    Экспорт в Markdown.
    """
    filepath = exporter.export_to_md(
            first_name=person.get("meaningful_first_name", "Unknown"),
            last_name=person.get("meaningful_last_name", "Unknown"),
            content=ans,
            urls=urls,
            personal_channel=person.get('personal_channel_username', '')
        )

    if filepath:
        logger.info(f"✅ Создан файл: {filepath}")
        return filepath
    else:
        logger.error("❌ Ошибка создания файла")
        return


def test_mdsearch(start_position: int, row_count: int) -> None:
    """
    Выполняет поиск информации через PerplexityClient
    и экспортирует результаты в Markdown.
    """
    logger.info("Начинаем поиск информации через PerplexityClient.")
    perp = PerplexityClient()
    db = DatabaseManager()

    query: str = f"SELECT * FROM {config.result_table_name} \
        WHERE valid ORDER BY person_id"
    if 0 < row_count: query += f" LIMIT {row_count}"
    if 0 < start_position: query += f" OFFSET {start_position}"

    persons = db.execute_query(query)
    update_query = f"UPDATE {config.result_table_name} \
        SET summary = %s, urls = %s WHERE person_id = %s"

    date_str: str = datetime.datetime.now().strftime("%Y-%m-%d-%H:%M")
    exporter = MarkdownExporter(f"data/{date_str}_person_reports")

    count_of_affected_rows = 0
    total = len(persons)
    for person in persons:
        id = process_person_md(db, person, update_query,
                               exporter, False, perp
        )
        count_of_affected_rows += 1
        if not (count_of_affected_rows % 5) or count_of_affected_rows == total:
            logger.info(f"Обработано {count_of_affected_rows} / {total} записей, "
                        f"БД изменила id - {id}"
            )

    db.close()
    logger.info("Экспорт в Markdown завершен.")


def export_to_html() -> None:
    logger.info("Начинаем экспорт людей в html таблицу.")
    db = DatabaseManager()
    query: str = f"SELECT * FROM {config.result_table_name} \
        WHERE valid ORDER BY person_id"
    persons = db.execute_query(query)

    try:
        html_template = Path('templates/template.html').read_text(encoding='utf-8')
        person_template = Path('templates/person_row.html').read_text(encoding='utf-8')
    except FileNotFoundError as e:
        logger.error(f"Файл шаблона не найден: {e}")
        return

    people_html = ""

    for index, person in enumerate(persons):
        first_name = person.get('meaningful_first_name', '') or ''
        last_name = person.get('meaningful_last_name', '') or ''
        about = person.get('meaningful_about', '') or ''
        username = person.get('username', '') or ''
        personal_channel_username = person.get('personal_channel_username', '') or ''
        personal_channel_about = person.get('personal_channel_about', '') or ''
        summary = person.get('summary', '') or ''
        person_id = person.get('person_id', '')
        urls = person.get('urls', []) or []

        full_name = f"{first_name} {last_name} ({person_id})".strip()

        if not personal_channel_username:
            channel_display = '<span class="empty-field">Отсутствует</span>'
        else:
            channel_display = f'@{personal_channel_username}'

        if not personal_channel_about:
            channel_about_display = '<span class="empty-field">Описание не указано</span>'
        else:
            channel_about_display = personal_channel_about

        urls_html = ""
        if urls:
            urls_html = '<div class="urls-list"><strong>Ссылки:</strong>'
            for url in urls:
                urls_html += f'<div class="url-item">• <a href="{url}" \
                class="url-link" target="_blank">{url}</a></div>'
            urls_html += '</div>'
        else:
            urls_html = '<div class="no-urls">Ссылки не найдены</div>'

        person_html = person_template.format(
            index=index,
            full_name=full_name,
            username=username,
            channel_display=channel_display,
            channel_about_display=channel_about_display,
            about=about,
            summary=summary,
            urls_html=urls_html
        )

        people_html += person_html

    final_html = html_template.replace('<!-- PEOPLE_DATA -->', people_html)

    result_filename = "people_analysis.html"
    Path(result_filename).write_text(final_html, encoding='utf-8')

    logger.info(f"HTML таблица сохранена в файл: {result_filename}")


def main() -> None:
    """
    Основная функция CLI. По умолчанию выводит справку.
    """
    parser = argparse.ArgumentParser(description="Инструменты работы с БД и LLM")
    parser.add_argument("--clean-db", action="store_true",
                        help="Очистка и подготовка базы данных"
    )
    parser.add_argument("--llm", action="store_true",
                        help="Обработка записей через LLM"
    )
    parser.add_argument("--mdsearch", action="store_true",
                        help="Поиск информации и экспорт в Markdown"
    )
    parser.add_argument("--start", type=int, default=0,
                        help="Начальная позиция записи"
    )
    parser.add_argument("--count", type=int, default=-1,
                        help="Количество записей"
    )
    parser.add_argument("--to_html", action="store_true",
                        help="Экспорт в html таблицу"
    )
    args = parser.parse_args()

    if args.clean_db:
        clean_and_create_db()
    elif args.llm:
        test_llm(start_position=args.start, row_count=args.count)
    elif args.mdsearch:
        test_mdsearch(start_position=args.start, row_count=args.count)
    elif args.to_html:
        export_to_html()
    else:
        parser.print_help()


if __name__ == "__main__":
    main()
