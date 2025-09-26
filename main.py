import logging
from logger import setup_logging
from db import DatabaseManager


def main():
    setup_logging(level=logging.DEBUG)
    logger = logging.getLogger(__name__)
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

        a = db.execute_query('SELECT * FROM person_source_data LIMIT %s', (5, ))
    else:
        logger.error("Не удалось подключиться к БД.")
    
    db.close()

if __name__ == "__main__":
    main()