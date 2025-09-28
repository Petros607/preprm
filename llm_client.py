import json
from typing import Any

from base_llm_client import BaseLLMClient
from config import LlmConfig


class LlmClient(BaseLLMClient):
    """
    Конкретный клиент для "общих" LLM-вызовов.
    Использует модель из config.default_model по умолчанию.
    """

    def __init__(self, config: LlmConfig | None = None) -> None:
        super().__init__(config=config)
        self.logger.debug("LlmClient initialized",
                          extra={"default_model": self.config.default_model})

    def ask_llm(self, prompt: str,
                response_format: str = "json_object", temperature: float = 0.0
                ) -> Any:
        """
        Универсальный метод для обращения к LLM.
        Возвращает dict (если response_format == "json_object") или str.
        В случае ошибки возвращает {} или "".
        """
        result, _raw = self._request_llm(
            prompt=prompt,
            model=self.config.default_model,
            response_format=response_format,
            temperature=temperature,
        )
        return result

    def parse_names_and_about_chunk(self, chunk: dict) -> dict:
        """
        Обрабатывает пачку записей (chunk) — вызывает LLM и ожидает json_object.
        Возвращает dict ({} при ошибке или невалидном JSON).
        """
        try:
            chunk_json = json.dumps(chunk, ensure_ascii=False)
        except Exception as exc:
            self.logger.error("Chunk is not serializable to JSON; "
                            "returning empty result.", exc_info=exc
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
        self.logger.debug("Calling ask_llm for parse_names_and_about_chunk",
                          extra={"chunk_size": len(chunk)}
                          )
        response = self.ask_llm(prompt, response_format="json_object")
        if not isinstance(response, dict):
            self.logger.warning("Expected dict but got different type; returning {}.")
            return {}
        return response
