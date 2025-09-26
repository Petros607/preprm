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
