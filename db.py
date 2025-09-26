import psycopg2
from psycopg2.extras import RealDictCursor
from typing import List, Dict, Optional
import logging

from config import DatabaseConfig

class DatabaseManager:
    def __init__(self, config: Optional[DatabaseConfig] = None):
        self.config = config or DatabaseConfig()
        self.connection = None
        self.is_connected = False
        self.logger = logging.getLogger(__name__)
        self.connect()
    
    def connect(self) -> bool:
        """Установка соединения с БД"""
        try:
            self.connection = psycopg2.connect(
                host=self.config.host,
                database=self.config.database,
                user=self.config.user,
                password=self.config.password,
                port=self.config.port
            )
            self.is_connected = True
            self.logger.info(
                f"Успешное подключение к БД: {self.config.host}:{self.config.port}/{self.config.database}"
            )
            return True
            
        except psycopg2.OperationalError as e:
            self.logger.error(f"Ошибка подключения к БД: {e}")
            self.is_connected = False
            return False
        except Exception as e:
            self.logger.error(f"Неожиданная ошибка при подключении: {e}")
            self.is_connected = False
            return False
    
    def get_table_person_data(self, limit: Optional[int] = None) -> List[Dict]:
        """Получение данных из таблицы person_source_data"""
        if not self.is_connected or not self.connection:
            self.logger.warning("Попытка получить данные без активного подключения к БД")
            return []
            
        try:
            cursor = self.connection.cursor(cursor_factory=RealDictCursor)
            query = "SELECT * FROM person_source_data"
            params = None
            
            if limit and limit > 0:
                query += " LIMIT %s"
                params = (limit,)
            
            self.logger.debug(f"Выполнение запроса: {query} с параметрами: {params}")
            cursor.execute(query, params)
            results = cursor.fetchall()
            cursor.close()
            
            self.logger.info(f"Успешно получено {len(results)} записей из БД")
            return results
            
        except psycopg2.Error as e:
            self.logger.error(f"Ошибка выполнения SQL запроса: {e}")
            return []
        except Exception as e:
            self.logger.error(f"Неожиданная ошибка при получении данных: {e}")
            return []
        
    def test_connection(self) -> bool:
        """Тестирование подключения к БД"""
        if not self.is_connected or not self.connection:
            self.logger.warning("Нет активного подключения для тестирования")
            return False
            
        try:
            cursor = self.connection.cursor()
            cursor.execute("SELECT version(), current_database(), current_user")
            db_info = cursor.fetchone()
            cursor.close()
            if (db_info):
                self.logger.info(
                    f"Тест подключения успешен. "
                    f"БД: {db_info[1]}, Пользователь: {db_info[2]}, "
                    f"Версия: {db_info[0].split(',')[0]}"
                )
            return True
            
        except psycopg2.Error as e:
            self.logger.error(f"Ошибка тестирования подключения: {e}")
            return False
        except Exception as e:
            self.logger.error(f"Неожиданная ошибка при тестировании: {e}")
            return False
    
    def execute_query(self, query: str, params: Optional[tuple] = None) -> List[Dict]:
        """Универсальный метод для выполнения произвольных SQL запросов"""
        if not self.is_connected or not self.connection:
            self.logger.warning("Попытка выполнить запрос без активного подключения")
            return []
            
        try:
            cursor = self.connection.cursor(cursor_factory=RealDictCursor)
            self.logger.debug(f"Выполнение запроса: {query} с параметрами: {params}")
            cursor.execute(query, params)
            
            if query.strip().upper().startswith('SELECT'):
                results = cursor.fetchall()
                self.logger.info(f"Получено {len(results)} записей")
            else:
                self.connection.commit()
                results = [{"affected_rows": cursor.rowcount}]
                self.logger.info(f"Запрос выполнен, затронуто строк: {cursor.rowcount}")
            
            cursor.close()
            return results
            
        except psycopg2.Error as e:
            self.logger.error(f"Ошибка выполнения запроса: {e}")
            self.connection.rollback()
            return []
        except Exception as e:
            self.logger.error(f"Неожиданная ошибка при выполнении запроса: {e}")
            return []
    
    def get_table_info(self, table_name: str = "person_source_data") -> List[Dict]:
        """Получение информации о структуре таблицы"""
        query = """
        SELECT column_name, data_type, is_nullable, column_default
        FROM information_schema.columns 
        WHERE table_name = %s 
        ORDER BY ordinal_position
        """
        return self.execute_query(query, (table_name,))
    
    def close(self) -> None:
        """Закрытие соединения с БД"""
        if self.connection:
            try:
                self.connection.close()
                self.is_connected = False
                self.logger.info("Соединение с БД успешно закрыто")
            except Exception as e:
                self.logger.error(f"Ошибка при закрытии соединения: {e}")
        else:
            self.logger.debug("Соединение уже закрыто или не было установлено")
