import logging
from openai import OpenAI
from typing import Dict, Optional

from config import LlmConfig

class LlmClient:
    def __init__(self, config: Optional[LlmConfig] = None):
        self.config = config or LlmConfig()
        self.client = OpenAI(
            base_url=self.config.url,
            api_key=self.config.key,
        )
        self.logger = logging.getLogger(__name__)

    def parse_name(self, first_name: str, last_name: str) -> Dict[str, str]:
        """
        Нормализация имени и фамилии с помощью LLM.
        Возвращает словарь с ключами:
        - cleaned_first_name
        - cleaned_last_name
        - additional_info
        """
        try:
            full_name = f"{first_name or ''} {last_name or ''}".strip()
            prompt = f"""
Ты должен разобрать данные о человеке, которые могут содержать имя, фамилию и дополнительные описания. 
Тебе даются два поля: FN (first_name) и LN (last_name). 
Они могут содержать как имя, так и фамилию, так и любую другую информацию (например: "маг чародей 5-го уровня").

Задача:
1. Попробуй выделить корректные имя и фамилию.
2. Всё, что не является частью имени или фамилии — положи в additional_info.
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

            completion = self.client.chat.completions.create(
                model="qwen/qwen3-14b:free",
                messages=[{"role": "user", "content": prompt}],
                response_format={"type": "json_object"},
            )
            content = completion.choices[0].message.content
            return eval(content)  # лучше заменить на json.loads()
        except Exception as e:
            self.logger.error(f"Ошибка при вызове LLM: {e}")
            return {
                "cleaned_first_name": "",
                "cleaned_last_name": "",
                "additional_info": "",
            }
