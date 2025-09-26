import logging
from logger import setup_logging
from db import DatabaseManager
from llm_client import LlmClient

setup_logging(level=logging.INFO)
logger = logging.getLogger(__name__)

def main():
    db = DatabaseManager()
    llm = LlmClient()
    db.create_temp_person_table()
    # records = db.get_temp_person_data(10)
    # for record in records:
    #     fn, ln = record["first_name"], record["last_name"]

    #     parsed = llm.parse_name(fn, ln)
    #     if is_valid_person(parsed):
    #         print("✅ Пригодная запись:")
    #         logger.info(f"ФИ: {parsed["cleaned_first_name"]} {parsed["cleaned_last_name"]}")
    #         logger.info(f"Остальное: {parsed["additional_info"].strip()}\n")
    #     else:
    #         print("❌ Запись непригодна")
    #         logger.info(f"ФИ: {parsed["cleaned_first_name"]} {parsed["cleaned_last_name"]}")
    #         logger.info(f"Остальное: {parsed["additional_info"].strip()}\n")
    
    db.close()

def is_valid_person(parsed: dict) -> bool:
    """
    Проверяет пригодность данных:
    - должны быть и имя, и фамилия
    """
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