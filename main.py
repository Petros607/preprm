import logging
from logger import setup_logging
from db import DatabaseManager
from llm_client import LlmClient
import csv


setup_logging(level=logging.INFO)
logger = logging.getLogger(__name__)

def main():
    db = DatabaseManager()
    llm = LlmClient()
    
    records = db.get_temp_person_data(300)

    with open("processed_persons.csv", mode="w", newline="", encoding="utf-8") as csvfile:
        fieldnames = [
            "telegram_id",
            "first_name",
            "last_name",
            "about",
            "valid",
            "cleaned_first_name",
            "cleaned_last_name",
            "additional_info",
        ]
        writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
        writer.writeheader()

        for record in records:
            fn, ln = record["first_name"], record["last_name"]

            parsed = llm.parse_name(fn, ln)
            valid = is_valid_person(parsed)

            row = {
                    "telegram_id": record.get("telegram_id", ""),
                    "first_name": fn,
                    "last_name": ln,
                    "about": record.get("about", ""),
                    "valid": valid,
                    "cleaned_first_name": parsed["cleaned_first_name"],
                    "cleaned_last_name": parsed["cleaned_last_name"],
                    "additional_info": parsed["additional_info"],
                }
            writer.writerow(row)

            if valid:
                logger.info(f"✅ Пригодная запись: {parsed['cleaned_first_name']} {parsed['cleaned_last_name']}")
            else:
                logger.info(f"❌ Непригодная запись: {fn} {ln}")
    
    db.close()

def is_valid_person(parsed: dict) -> bool:
    return bool(parsed.get("cleaned_first_name") and parsed.get("cleaned_last_name"))



def test_db():
    db = DatabaseManager()
    if db.is_connected:
        db.test_connection()

        table_info = db.get_table_info()
        logger.info(f"Структура таблицы: {len(table_info)} колонок")
        for column in table_info:
            logger.debug(f"Колонка: {column['column_name']}, Тип: {column['data_type']}")

        persons = db.get_table_person_data(limit=5)
        logger.info(f"Получено {len(persons)} записей для обработки")
    
        if persons:
            first_person = persons[0]
            logger.info("Первая запись для обработки:")
            logger.info(f"{first_person}")

        db.execute_query('SELECT * FROM person_source_data LIMIT %s', (5, ))
    else:
        logger.error("Не удалось подключиться к БД.")
    db.close()

if __name__ == "__main__":
    main()