import logging
from logger import setup_logging
from db import DatabaseManager
from llm_client import LlmClient
from perp_client import PerplexityClient
from md_exporter import MarkdownExporter
import datetime
import config

setup_logging(level=logging.INFO)
logger = logging.getLogger(__name__)


def clean_and_create_db():
    """ Первое действие: очистка изначальной бд и создание финальной"""
    db = DatabaseManager()
    
    db.create_cleaned_table(source_table_name=config.source_table_name,
                            new_table_name=config.cleaned_table_name
                            )
    db.create_result_table(source_table_name=config.cleaned_table_name,
                           result_table_name=config.result_table_name
                           )
    db.close()

def test_llm():
    db = DatabaseManager()
    start_position = 0
    row_count = 100
    query = f"SELECT * FROM {config.result_table_name} ORDER BY person_id"
    if row_count and 0 < row_count:
        query += f" LIMIT {row_count}"
    if start_position and 0 < start_position <= row_count:
        query += f" OFFSET {start_position}"
    records = db.execute_query(query)
    total = len(records)

    llm = LlmClient()
    CHUNK_SIZE = 10
    i = 0
    retry = 0
    update_query = f"""
            UPDATE {config.result_table_name} 
            SET meaningful_first_name = %s,
                meaningful_last_name = %s,
                meaningful_about = %s,
                valid = %s
            WHERE person_id = %s
        """.strip()
    while i < total:
        if i + CHUNK_SIZE < total: record = records[i:i + CHUNK_SIZE]
        else: record = records[i:total]
        chunk = transform_record_to_chunk(record)
        parsed_chunk = llm.parse_names_and_about_chunk(chunk)
        count_of_affected_rows = 0
        if not parsed_chunk:
            logger.warning(f"Батч с {i} по {i+CHUNK_SIZE} не обработался ИИ! Попытка №{retry}")
            retry += 1
            if retry > 3: retry = 0
        else:
            if retry: retry = 0
            for index, data in parsed_chunk.items():
                try:
                    params = (
                        data.get('meaningful_first_name', ''),
                        data.get('meaningful_last_name', ''),
                        data.get('meaningful_about', ''),
                        bool(data.get('meaningful_first_name', '') 
                             and data.get('meaningful_last_name', '') 
                             and data.get('meaningful_about', '')),
                        data['person_id']
                    )
                    result = db.execute_query(update_query, params)
                    if result and result[0].get('affected_rows', 0) > 0: count_of_affected_rows+=1
                    else: logger.info(f"Бд не изменила строку с person_id {data['person_id']}")
                except Exception as e:
                    logger.error(f"Ошибка при обновлении person_id {data['person_id']}: {e}")
        if not(retry):
            i+=CHUNK_SIZE
        logger.info(f"Обработано {min(i, total)} / {total} записей, бд сообщила об изменении {count_of_affected_rows} строк")
    db.close()

def transform_record_to_chunk(data):
    result = {}
    for index, row in enumerate(data):
        result[index] = {
            "person_id": row['person_id'],
            "first_name": row['first_name'] or "",
            "last_name": row['last_name'] or "",
            "about": row['about'] or ""
        }
    return result


def test_mdsearch():
    perp = PerplexityClient()
    db = DatabaseManager()

    start_position = -1
    row_count = -1
    query = f"SELECT * FROM {config.result_table_name} WHERE valid ORDER BY person_id"
    if row_count and 0 < row_count:
        query += f" LIMIT {row_count}"
    if start_position and 0 < start_position <= row_count:
        query += f" OFFSET {start_position}"
    persons = db.execute_query(query)
    update_query = f"""
            UPDATE {config.result_table_name} 
            SET summary = %s,
                urls = %s
            WHERE person_id = %s
        """.strip()
    date_str = datetime.datetime.now().strftime("%Y-%m-%d-%H:%M")
    exporter = MarkdownExporter(f"{date_str}_person_reports")
    
    for person in persons:
        if person.get("valid", False):
            ans, urls = perp.search_info(
                first_name=person.get("meaningful_first_name", ""),
                last_name=person.get("meaningful_last_name", ""),
                additional_info=person.get("meaningful_about", ""),
                # birth_date=person.get(""), #SOMEDAY
                personal_channel_name=person.get("personal_channel_username"),
                personal_channel_about=person.get("personal_channel_about"),
                temperature=0.1
            )

            try: db.execute_query(update_query, (ans, urls, person.get('person_id')))
            except Exception as e: logger.error(f"Ошибка при обновлении person_id {person.get('person_id')}: {e}")
            
            filepath = exporter.export_to_md(
                first_name=person.get("meaningful_first_name", "Unknown"),
                last_name=person.get("meaningful_last_name", "Unknown"),
                content=ans,
                urls=urls,
                personal_channel=person.get('personal_channel_username', '')
            )
            
            if filepath: print(f"✅ Создан файл: {filepath}")
            else: print("❌ Ошибка создания файла")
    db.close()

def main():
    # clean_and_create_db()
    # test_llm()
    test_mdsearch()

if __name__ == "__main__":
    main()