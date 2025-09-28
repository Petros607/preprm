import json
from typing import Any

from config import LlmConfig
from llm.base_llm_client import BaseLLMClient


class LlmClient(BaseLLMClient):
    """Конкретный клиент для общих LLM-вызовов.
    Наследуется от BaseLLMClient и использует модель по умолчанию
    из конфигурации для обработки запросов к языковым моделям.
    Attributes:
        config (LlmConfig): Конфигурация LLM клиента
        logger: Логгер для записи событий
    """

    def __init__(self, config: LlmConfig | None = None) -> None:
        """Инициализация LLM клиента.
        Args:
            config: Конфигурация LLM. Если не указана, используется по умолчанию.
        """
        super().__init__(config=config)
        self.logger.debug("LlmClient инициализирован",
                          extra={"default_model": self.config.default_model})

    def ask_llm(self, prompt: str, response_format: str = "json_object",
                temperature: float = 0.0) -> Any:
        """Универсальный метод для обращения к LLM.
        Выполняет запрос к языковой модели с указанными параметрами
        и возвращает результат в заданном формате.
        Args:
            prompt: Текст промпта для отправки в LLM
            response_format: Формат ответа ("json_object" или "text")
            temperature: Температура для генерации (0.0 - детерминированная)
        Returns:
            Any: Словарь если response_format == "json_object", иначе строка.
                 В случае ошибки возвращает {} или "".
        """
        result, _raw = self._request_llm(
            prompt=prompt,
            model=self.config.default_model,
            response_format=response_format,
            temperature=temperature,
        )
        return result

    def parse_names_and_about_chunk(self, chunk: dict[str, Any]) -> dict[str, Any]:
        """Обрабатывает пачку записей для извлечения имен и информации.
        Преобразует chunk данных в JSON, отправляет в LLM для анализа
        и возвращает структурированные данные об именах и информации.
        Args:
            chunk: Словарь с данными для обработки
        Returns:
            Dict[str, Any]: Словарь с обработанными данными или пустой словарь при ошибке
        """
        try:
            chunk_json = json.dumps(chunk, ensure_ascii=False)
        except Exception as exc:
            self.logger.error("Chunk не может быть сериализован в JSON; "
                            "возвращаем пустой результат.", exc_info=exc
                            )
            return {}

        prompt = f"""
Ты — ассистент для обработки данных о людях.
Твоя задача — корректно извлечь имя, фамилию
и дополнительную информацию из предоставленных данных.
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
{chunk_json}
"""
        self.logger.debug("Вызов ask_llm для parse_names_and_about_chunk",
                          extra={"chunk_size": len(chunk)}
                          )
        response = self.ask_llm(prompt, response_format="json_object")
        if not isinstance(response, dict):
            self.logger.warning("Ожидался словарь, но получен другой тип; возвращаем {}.")
            return {}
        return response
