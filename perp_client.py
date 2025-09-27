import logging
from openai import OpenAI
from typing import Dict, Optional, Any, List
import json

from config import LlmConfig

class PerplexityClient:
    def __init__(self, config: Optional[LlmConfig] = None):
        self.config = config or LlmConfig()
        self.client = OpenAI(
            base_url=self.config.url,
            api_key=self.config.key,
        )
        self.logger = logging.getLogger(__name__)

    def ask_perplexity(self, 
                      prompt: str, 
                      model: str = "perplexity/sonar",
                      response_format: str = "text",
                      temperature: float = 0.1,
                      max_tokens: Optional[int] = None
                      ) -> Any:
        """
        Универсальный метод для обращения к Perplexity через OpenRouter.
        
        Args:
            prompt: Текст запроса
            model: Модель Perplexity (sonar, sonar-debug и т.д.)
            response_format: Формат ответа ("text" или "json_object")
            temperature: Температура генерации
            max_tokens: Максимальное количество токенов
        
        Returns:
            Ответ от модели в указанном формате
        """
        try:
            extra_body = {}
            if max_tokens:
                extra_body["max_tokens"] = max_tokens

            completion = self.client.chat.completions.create(
                # extra_headers=extra_headers,
                extra_body=extra_body,
                model=model,
                messages=[{"role": "user", "content": prompt}],
                response_format={"type": response_format} if response_format == "json_object" else None,
                temperature=temperature,
                n=1,
            )
            
            content = completion.choices[0].message.content
            
            if response_format == "json_object":
                try:
                    return json.loads(content)
                except json.JSONDecodeError:
                    self.logger.warning(f"Perplexity вернула невалидный JSON: {content}")
                    return {}
                    
            return content
            
        except Exception as e:
            self.logger.error(f"Ошибка при вызове Perplexity: {e}")
            return {} if response_format == "json_object" else ""

    def ask_sonar(self, 
                 prompt: str, 
                 response_format: str = "text",
                 temperature: float = 0.1) -> Any:
        """
        Специализированный метод для модели Sonar.
        """
        return self.ask_perplexity(
            prompt=prompt,
            model="perplexity/sonar",
            response_format=response_format,
            temperature=temperature
        )
    
    def search_info(self, 
                 first_name: str,
                 last_name: str,
                 additional_info: str,
                 birth_date: Optional[str] = None,
                 personal_channel_name: Optional[str] = None,
                 personal_channel_about: Optional[str] = None,
                 response_format: str = "text",
                 temperature: float = 0.1) -> Any:
        """
        Генерирует краткую информационную справку о человеке.
        """

        prompt = f"""
    Напиши аналитическую справку о человеке, синтезируя информацию из различных источников.
    ## ДАННЫЕ О ЧЕЛОВЕКЕ:
    - **Имя и фамилия**: {first_name} {last_name}
    - **Дополнительная информация**: {additional_info or "нет данных"}
    - **Дата рождения**: {birth_date or "неизвестна"}
    - **Персональный Telegram-канал**: {personal_channel_name or "отсутствует"}
    - **Описание канала**: {personal_channel_about or "нет описания"}
    ## ИНСТРУКЦИИ:
    1. **Создай лаконичную справку** (3-5 предложений)
    2. **Выдели ключевые аспекты** личности и деятельности
    3. **Используй профессиональный, но живой язык**
    4. **Если данных мало** - сделай выводы на основе того, что есть

    ## ФОРМАТ ОТВЕТА:
    Верни ТОЛЬКО текст справки без заголовков и дополнительных комментариев.
        """.strip()

        return self.ask_perplexity(
            prompt=prompt,
            model="perplexity/sonar",
            response_format=response_format,
            temperature=temperature
        )
