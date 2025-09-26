from dataclasses import dataclass
from typing import Optional
import os
from dotenv import load_dotenv

load_dotenv()

@dataclass
class DatabaseConfig:
    """Конфигурация подключения к базе данных"""
    host: str = os.getenv('DB_HOST', 'localhost')
    database: str = os.getenv('DB_NAME', 'postgres')
    user: str = os.getenv('DB_USER', 'postgres')
    password: Optional[str] = os.getenv('DB_PASSWORD')
    port: int = int(os.getenv('DB_PORT', '5432'))
    