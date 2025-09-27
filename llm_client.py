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
                model="qwen/qwen3-14b:free",
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

    def parse_name(self, first_name: str, last_name: str) -> Dict[str, str]:
        """
        Нормализация имени и фамилии с помощью LLM.
        Возвращает словарь с ключами:
        - cleaned_first_name
        - cleaned_last_name
        - additional_info
        """
        full_name = f"{first_name or ''} {last_name or ''}".strip()
        prompt = f"""
Ты должен разобрать данные о человеке, которые могут содержать имя, фамилию и дополнительные описания. 
Тебе даются два поля: FN (first_name) и LN (last_name). 
Они могут содержать как имя, так и фамилию, так и любую другую информацию (например: "маг чародей 5-го уровня").

Задача:
1. Попробуй выделить корректные имя и фамилию, в которых ты абсолютно уверен.
2. Всё, что не является частью имени или фамилии, но при этом является полезной информацией — положи в additional_info, иначе не обращай внимания.
3. Если невозможно выделить и имя, и фамилию — оставь их пустыми.

Исходные данные:
FN: "{first_name or ''}"
LN: "{last_name or ''}"

Верни JSON строго в формате:
{{
  "cleaned_first_name": "Имя или пустая строка",
  "cleaned_last_name": "Фамилия или пустая строка",
  "additional_info": "Доп. информация или пустая строка"
}}
"""
        result = self.ask_llm(prompt, response_format="json_object")
        return {
            "cleaned_first_name": result.get("cleaned_first_name", ""),
            "cleaned_last_name": result.get("cleaned_last_name", ""),
            "additional_info": result.get("additional_info", ""),
        }

    def parse_names_batch(self, records: list[dict], batch_size: int = 20) -> dict:
        """
        Обрабатывает список записей пачками.
        records: список словарей {"first_name": ..., "last_name": ...}
        Возвращает список словарей с результатами в том же порядке.
        """
        results = []
        for i in range(0, len(records), batch_size):
            batch = records[i:i + batch_size]
            prompt = f"""
Ты — ассистент для обработки данных о людях. Твоя задача — корректно извлечь имя, фамилию и дополнительную информацию из предоставленных данных.
## ИНСТРУКЦИИ:
1. **Анализ полей**: Внимательно изучи поля first_name, last_name и about
2. **Извлечение имени и фамилии**:
   - Имя и фамилия должны быть реалистичными (минимум 2 символа)
   - Пропускай односимвольные имена/фамилии (кроме общепринятых как "Al")
   - Если в одном поле содержится и имя и фамилия — раздели их
   - Приоритет: сначала попробуй выделить из first_name + last_name, затем из about
3. **Дополнительная информация, которая стоит внимания**:
   - Профессия, должность, специализация
   - Контактные данные (email, сайт, соцсети)
   - Ключевые навыки и компетенции
   - Компания или проект
4. **Фильтрация мусора**: Игнорируй случайные слова, не несущие смысловой нагрузки
## ФОРМАТ ОТВЕТА:
Верни ТОЛЬКО JSON-объект в точном формате:
{{
    "0": {{
        "cleaned_first_name": "Iaroslav",
        "cleaned_last_name": "Isaev", 
        "additional_info": ""
    }},
    "1": {{
        "cleaned_first_name": "",
        "cleaned_last_name": "",
        "additional_info": "On a Vibration Mode"
    }}
}}
## ДАННЫЕ ДЛЯ ОБРАБОТКИ (batch_size: {len(batch)}):
{json.dumps([{"index": idx, "first_name": r.get("first_name",""), "last_name": r.get("last_name",""), "about": r.get("about","")} for idx, r in enumerate(batch)], ensure_ascii=False, indent=2)}
"""
            response = self.ask_llm(prompt, response_format="json_object")
            # if isinstance(response, dict):
                # ordered = sorted(response, key=lambda x: x.get("index", 0))
                # results.extend(ordered)
        results = response # TODO
            # else:
            #     self.logger.warning(f"Некорректный ответ от LLM: {response}")
            #     results.extend({"index": -1:
            #         {
            #         "cleaned_first_name": "",
            #         "cleaned_last_name": "",
            #         "additional_info": ""
            #     } * len(batch))
        return results
