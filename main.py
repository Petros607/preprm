import logging
from logger import setup_logging
from db import DatabaseManager
from llm_client import LlmClient
from perp_client import PerplexityClient
from md_exporter import MarkdownExporter
import csv
import os
import datetime

setup_logging(level=logging.INFO)
logger = logging.getLogger(__name__)

def main():
    perp = PerplexityClient()
    db = DatabaseManager()

    persons = db.get_perp_person_data()
    date_str = datetime.datetime.now().strftime("%Y-%m-%d-%H:%M")
    exporter = MarkdownExporter(f"{date_str}_person_reports")
    count_of_persons = len(persons)
    i = 79
    while i < count_of_persons:
        person = persons[i]
        ans, urls = perp.search_info(
            first_name=person.get("cleaned_first_name"),
            last_name=person.get("cleaned_last_name"),
            additional_info=person.get("additional_info"),
            # birth_date=person.get("additional_info"),,
            personal_channel_name=person.get("personal_channel_username"),
            personal_channel_about=person.get("personal_channel_about"),
            temperature=0.1
        )
        # print("\n" + ans + "\n")
        # print(urls)
        filepath = exporter.export_to_md(
            first_name=person.get("cleaned_first_name", "Unknown"),
            last_name=person.get("cleaned_last_name", "Unknown"),
            content=ans,
            urls=urls,
            personal_channel=person.get('personal_channel_username', '')
        )
        
        if filepath:
            print(f"✅ Создан файл: {filepath}")
        else:
            print("❌ Ошибка создания файла")
        i+=1

def create_table_for_perp():
    db = DatabaseManager() 
    db.execute_query("""CREATE TABLE for_perp_persons AS
SELECT 
    tp.*,
    pp.cleaned_first_name,
    pp.cleaned_last_name, 
    pp.additional_info
FROM temp_person_table tp
INNER JOIN processed_persons pp ON tp.telegram_id = pp.telegram_id
WHERE pp.valid = true;""")

def csv_to_db():
    db = DatabaseManager()
    csv_file = "processed_persons.csv"
    table_name = "processed_persons"
    
    success = db.create_table_from_csv(
            csv_file_path=csv_file,
            table_name=table_name,
            delimiter=',',
            encoding='utf-8'
        )

    if success:
        logger.info(f"Таблица {table_name} успешно создана!")
        table_info = db.get_table_info(table_name)
        logger.info(f"Структура таблицы: {len(table_info)} колонок")
        sample_data = db.execute_query(f"SELECT * FROM {table_name} LIMIT 5")
        logger.info("Пример данных:")
        for row in sample_data:
            logger.info(row)

    db.close()

def FN_LN_About():
    db = DatabaseManager()
    llm = LlmClient()
    
    records = db.get_temp_person_data()
    total = len(records)
    logger.info(f"Получено {total} записей для обработки")

    csv_file = "processed_persons.csv"
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
    file_exists = os.path.exists(csv_file)

    with open(csv_file, mode="a", newline="", encoding="utf-8") as csvfile:
        writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
        if not file_exists:
            writer.writeheader()
        BATCH_SIZE = 15
        i = 0
        while i < total:
            if i + BATCH_SIZE < total:
                chunk = records[i:i + BATCH_SIZE]
            else:
                chunk = records[i:total]
            parsed_chunk = llm.parse_names_batch(chunk, batch_size=len(chunk))

            for record, parsed in zip(chunk, parsed_chunk):
                parsed = parsed_chunk.get(parsed, -1) #TODO
                valid = False
                if not('index' in parsed_chunk.keys()):
                    valid = is_valid_person(parsed)
                else:
                    parsed = {"cleaned_first_name": "", "cleaned_last_name": "", "additional_info": ""}
                    logger.warning(f"Батч с {i} по {i+BATCH_SIZE} не обработался")
                row = {
                    "telegram_id": record.get("telegram_id", ""),
                    "first_name": record.get("first_name", ""),
                    "last_name": record.get("last_name", ""),
                    "about": record.get("about", ""),
                    "valid": valid,
                    "cleaned_first_name": parsed.get("cleaned_first_name", ""),
                    "cleaned_last_name": parsed.get("cleaned_last_name", ""),
                    "additional_info": parsed.get("additional_info", ""),
                }
                writer.writerow(row)

                if valid:
                    logger.debug(f"✅ Пригодная запись: {row['cleaned_first_name']} {row['cleaned_last_name']}")
                else:
                    logger.debug(f"❌ Непригодная запись: {row['first_name']} {row['last_name']}")

            i+=BATCH_SIZE
            logger.info(f"Обработано {min(i, total)} / {total} записей")

    
    db.close()

def is_valid_person(parsed: dict) -> bool:
    return bool((parsed["cleaned_first_name"]) and (parsed["cleaned_last_name"]) and (parsed["additional_info"]))

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