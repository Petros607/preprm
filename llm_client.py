import logging
from openai import OpenAI
from typing import Dict, Optional, Any
import json

from config import LlmConfig

class LlmClient:
    def __init__(self, config: Optional[LlmConfig] = None):
        self.config = config or LlmConfig()
        self.client = OpenAI(
            base_url=self.config.url,
            api_key=self.config.key,
        )
        self.logger = logging.getLogger(__name__)

    def ask_llm(self, prompt: str, response_format: str = "json_object") -> Any:
        """
        Универсальный метод для обращения к LLM.
        """
        try:
            completion = self.client.chat.completions.create(
                # model="qwen/qwen3-14b:free",
                model="qwen/qwen3-14b:free", #TODO
                messages=[{"role": "user", "content": prompt}],
                response_format={"type": response_format},
                n=1,
                temperature=0,
            )
            content = completion.choices[0].message.content
            if response_format == "json_object":
                try:
                    return json.loads(content)
                except json.JSONDecodeError:
                    self.logger.warning(f"LLM вернула невалидный JSON: {content}")
                    return {}
            return content
        except Exception as e:
            self.logger.error(f"Ошибка при вызове LLM: {e}")
            return {}

    def parse_names_and_about_chunk(self, chunk: dict) -> dict:
        """
        Обрабатывает список записей пачками.
        records: список словарей {"first_name": ..., "last_name": ...}
        Возвращает список словарей с результатами в том же порядке.
        """
        prompt = f"""
Ты — ассистент для обработки данных о людях. Твоя задача — корректно извлечь имя, фамилию и дополнительную информацию из предоставленных данных.
## ИНСТРУКЦИИ:
1. **Анализ полей**: Внимательно изучи поля first_name, last_name и about
2. **Извлечение имени и фамилии**:
   - Имя и фамилия должны быть реалистичными (минимум 2 символа)
   - Пропускай односимвольные имена/фамилии (кроме общепринятых как "Al")
   - Если в одном поле содержится и имя и фамилия — раздели их
   - Приоритет: сначала попробуй выделить из first_name + last_name, затем из about
3. **Дополнительная информация (about), которая стоит внимания**:
   - Профессия, должность, специализация
   - Контактные данные (email, сайт, соцсети)
   - Ключевые навыки и компетенции
   - Компания или проект
4. **Фильтрация мусора**: Игнорируй случайные слова, не несущие смысловой нагрузки
## ФОРМАТ ОТВЕТА:
Верни ТОЛЬКО JSON-объект в точном формате:
{{
    "0": {{
        "person_id": 8194,
        "meaningful_first_name": "Ivan",
        "meaningful_last_name": "Ivanov", 
        "meaningful_about": "IT-specialist"
    }},
    "1": {{
        "person_id": 3465,
        "meaningful_first_name": "Петр",
        "meaningful_last_name": "Петров",
        "meaningful_about": "Разработчик, сооснователь компании"
    }}
}}
## ДАННЫЕ ДЛЯ ОБРАБОТКИ (количество: {len(chunk)}):
{json.dumps(chunk)}
"""
        return self.ask_llm(prompt, response_format="json_object")
