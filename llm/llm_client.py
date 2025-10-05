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
Ты — интеллектуальный фильтр и нормализатор биографий людей.
Твоя задача — из данных first_name / last_name / about / username
выделить три ключевых поля:
1. meaningful_first_name — реальное человеческое имя (если нет, попытайся угадать по username)
2. meaningful_last_name — реальная фамилия (если нельзя понять — оставь пусто)
3. meaningful_about — краткое описание роли человека (профессия / должность / деятельность)
---
Правила обработки:
### 🟢 Имена и фамилии
- Если имя/фамилия указаны — используй их.
- Может быть такое, что в одном поле написано и имя, и фамилия
- Если нет — попытайся угадать из username (например: "sergman" → "Sergey").
- Только реальные человеческие имена. Не надо "Admin", "Support", "CryptoBro".

### 🟡 Описание (meaningful_about)
- Сначала попробуй понять **главную работу / деятельность человека**.
- Сохрани ссылки, названия компаний (к примеру, @DRONICO).
- Формат: **"<роль> в <компания>"** или **"<роль> / <сфера>"**.
  - Пример: "Founder of CardPR.com" → "Основатель CardPR.com"
  - Пример: "COO @amma.family" → "Операционный директор в amma.family"

### 🔴 Что НЕ считается meaningful_about и должно быть отброшено
- Призывы подписаться, цитаты, философия ("живу один раз", "всё будет хорошо")
- "Люблю путешествовать", "мать двоих детей", "блогер" без конкретики
- Просто перечисление интересов без роли ("IT, бизнес, связь")

Если нельзя выделить чёткую профессию / роль — meaningful_about должен быть пустым.
---
ФОРМАТ ОТВЕТА (строго JSON):

{{
    "0": {{
        "person_id": 8194,
        "meaningful_first_name": "Иван",
        "meaningful_last_name": "Иванов",
        "meaningful_about": "Инженер-программист в Google"
    }}
    "1": {{
        "person_id": "412412",
        "meaningful_first_name": "Мария",
        "meaningful_last_name": "Макарова",
        "meaningful_about": ""
    }}
}}

---
ДАННЫЕ ДЛЯ ОБРАБОТКИ (количество: {len(chunk)}):
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
