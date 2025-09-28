import argparse
import datetime
import logging
from typing import Any

import config
from db import DatabaseManager
from llm_client import LlmClient
from logger import setup_logging
from md_exporter import MarkdownExporter
from perp_client import PerplexityClient

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
        result_table_name=config.result_table_name
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
                 and data.get('meaningful_about')),
            data.get('person_id')
        )
        if safe_execute(db, update_query, params):
            count_of_affected_rows += 1
        else:
            logger.info(f"БД не изменила строку с person_id {data.get('person_id')}")

    return count_of_affected_rows


def test_llm() -> None:
    """
    Обрабатывает записи из result_table_name через LLM и обновляет базу данных.
    """
    logger.info("Начинаем обработку записей через LLM.")
    db = DatabaseManager()
    start_position: int = 0
    row_count: int = 100
    query: str = f"SELECT * FROM {config.result_table_name} ORDER BY person_id"
    if 0 < row_count:
        query += f" LIMIT {row_count}"
    if 0 < start_position <= row_count:
        query += f" OFFSET {start_position}"

    records = db.execute_query(query)
    total = len(records)
    llm = LlmClient()
    CHUNK_SIZE: int = 10
    i = 0
    retry = 0

    update_query = f"UPDATE {config.result_table_name} \
        SET meaningful_first_name = %s, meaningful_last_name = %s, \
            meaningful_about = %s, valid = %s WHERE person_id = %s"

    while i < total:
        record_chunk = records[i:i + CHUNK_SIZE]
        chunk = transform_record_to_chunk(record_chunk)
        count_of_affected_rows = process_llm_batch(db, chunk, update_query, llm)

        if count_of_affected_rows == 0:
            logger.warning(f"Батч с {i} по {i + CHUNK_SIZE} не обработался ИИ! "
                           "Попытка №{retry}")
            retry += 1
            if retry > 3:
                retry = 0
        else:
            retry = 0

        if retry == 0:
            i += CHUNK_SIZE
            logger.info(f"Обработано {min(i, total)} / {total} записей, "
                        "БД изменила {count_of_affected_rows} строк")

    db.close()
    logger.info("Обработка LLM завершена.")


def process_person_md(db: DatabaseManager, person: dict[str, Any],
                      update_query: str, exporter: MarkdownExporter,
                      perp: PerplexityClient
                    ) -> None:
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

    filepath = exporter.export_to_md(
        first_name=person.get("meaningful_first_name", "Unknown"),
        last_name=person.get("meaningful_last_name", "Unknown"),
        content=ans,
        urls=urls,
        personal_channel=person.get('personal_channel_username', '')
    )

    if filepath:
        logger.info(f"✅ Создан файл: {filepath}")
    else:
        logger.info("❌ Ошибка создания файла")


def test_mdsearch() -> None:
    """
    Выполняет поиск информации через PerplexityClient
    и экспортирует результаты в Markdown.
    """
    logger.info("Начинаем поиск информации через PerplexityClient.")
    perp = PerplexityClient()
    db = DatabaseManager()

    start_position: int = -1
    row_count: int = -1
    query: str = f"SELECT * FROM {config.result_table_name} \
        WHERE valid ORDER BY person_id"
    if 0 < row_count:
        query += f" LIMIT {row_count}"
    if 0 < start_position <= row_count:
        query += f" OFFSET {start_position}"

    persons = db.execute_query(query)
    update_query = f"UPDATE {config.result_table_name} \
        SET summary = %s, urls = %s WHERE person_id = %s"

    date_str: str = datetime.datetime.now().strftime("%Y-%m-%d-%H:%M")
    exporter = MarkdownExporter(f"{date_str}_person_reports")

    for person in persons:
        process_person_md(db, person, update_query, exporter, perp)

    db.close()
    logger.info("Экспорт в Markdown завершен.")


def main() -> None:
    """
    Основная функция CLI. По умолчанию выводит справку.
    """
    parser = argparse.ArgumentParser(description="Инструменты работы с БД и LLM")
    parser.add_argument("--clean-db", action="store_true",
                        help="Очистка и подготовка базы данных")
    parser.add_argument("--llm", action="store_true",
                        help="Обработка записей через LLM")
    parser.add_argument("--mdsearch", action="store_true",
                        help="Поиск информации и экспорт в Markdown")
    args = parser.parse_args()

    if args.clean_db:
        clean_and_create_db()
    elif args.llm:
        test_llm()
    elif args.mdsearch:
        test_mdsearch()
    else:
        parser.print_help()


if __name__ == "__main__":
    main()
