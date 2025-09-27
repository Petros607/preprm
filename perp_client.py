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
            urls = self.extract_urls_from_response(completion)
            
            if response_format == "json_object":
                try:
                    return json.loads(content)
                except json.JSONDecodeError:
                    self.logger.warning(f"Perplexity вернула невалидный JSON: {content}")
                    return {}
                    
            return content, urls
            
        except Exception as e:
            self.logger.error(f"Ошибка при вызове Perplexity: {e}")
            return {} if response_format == "json_object" else ""
        
    def extract_urls_from_response(self, completion_response) -> List[Dict[str, str]]:
        """
        Извлекает все URL из ответа Perplexity
        """
        try:
            message = completion_response.choices[0].message
            urls = []
            if hasattr(message, 'annotations') and message.annotations:
                for annotation in message.annotations:
                    if hasattr(annotation, 'url_citation') and annotation.url_citation:
                        urls.append(annotation.url_citation.url,)
            return urls
        except Exception as e:
            self.logger.error(f"Ошибка извлечения URL: {e}")
            return []

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
    Составь краткую аналитическую справку о человеке, используя имеющиеся данные для поиска дополнительной информации в интернете.
    ## КЛЮЧЕВОЕ ПРАВИЛО:
    Используй ТОЛЬКО содержательные данные. 
    Если какое-либо поле содержит маркеры **"отсутствует" / "нет данных" / "неизвестно"** или пустые фразы (**"Tout passe, tout casse, tout lasse", "Тише едешь..." и т.п.**) — **полностью игнорируй это поле и не упоминай про его отсутствие.**

    ## ДОПУСТИМЫЕ ФОРМУЛИРОВКИ ПРИ МАЛОМ КОЛИЧЕСТВЕ ДАННЫХ:
    Допускаются осторожные выводы, но только с явными маркерами:
    - "Судя по активности..."
    - "Можно предположить..."
    - "Вероятно связан с..."

    **Запрещено:**
    - приписывать конкретные факты без подтверждения;
    - писать воду вроде «профиль недостаточно раскрыт».

    ## ДАННЫЕ О ЧЕЛОВЕКЕ:
    - Имя и фамилия: {first_name} {last_name}
    - Дополнительная информация: {additional_info or "отсутствует"}
    - Дата рождения: {birth_date or "отсутствует"}
    - Персональный Telegram-канал: {personal_channel_name or "отсутствует"}
    - Описание Telegram-канала: {personal_channel_about or "отсутствует"}

    ## ФОРМАТ:
    Объём: 3–5 предложений. Профессиональный, но живой стиль.
    Не упоминай то, чего нет. Не делай итоговых рефлексий. Просто выдай выжимку фактов и аккуратных наблюдений.

    ## ПРИМЕР ХОРОШЕГО ОТВЕТА:
    Pasha Severilov — специалист с опытом работы в области машинного обучения и искусственного интеллекта, ранее связанный с компанией Yandex. Судя по доступным данным, он имеет академическую подготовку в области прикладной математики и информатики, а также опыт преподавания в ведущих российских вузах.
    """.strip()

        return self.ask_perplexity(
            prompt=prompt,
            model="perplexity/sonar",
            response_format=response_format,
            temperature=temperature
        )
