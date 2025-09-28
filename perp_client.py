from typing import Any

from base_llm_client import BaseLLMClient
from config import LlmConfig


class PerplexityClient(BaseLLMClient):
    """Клиент для работы с Perplexity-like моделями (search/sonar).
    Специализированный клиент для моделей с поисковыми возможностями,
    использует модель из конфигурации по умолчанию для Perplexity.
    Attributes:
        config (LlmConfig): Конфигурация LLM клиента
        logger: Логгер для записи событий
    """

    def __init__(self, config: LlmConfig | None = None) -> None:
        """Инициализация клиента Perplexity.
        Args:
            config: Конфигурация LLM. Если не указана, используется по умолчанию.
        """
        super().__init__(config=config)
        self.logger.debug("PerplexityClient инициализирован",
                          extra={"perplexity_model": self.config.perplexity_model}
                        )

    def ask_perplexity(
        self,
        prompt: str,
        model: str | None = None,
        response_format: str = "text",
        temperature: float = 0.1,
        max_tokens: int | None = None,
    ) -> Any:
        """Универсальный метод для обращения к Perplexity через OpenRouter.
        Args:
            prompt: Текст промпта для отправки в модель
            model: Название модели. Если не указана, используется по умолчанию
            response_format: Формат ответа ("text" или "json_object")
            temperature: Температура для генерации
            max_tokens: Максимальное количество токенов в ответе
        Returns:
            Any: Если response_format == "json_object" - словарь (или {} при ошибке),
                 иначе - кортеж (text, urls) где text - строка, urls - список URL
        """
        model_to_use = model or self.config.perplexity_model
        self.logger.debug("Вызов ask_perplexity",
                          extra={"model": model_to_use,
                                 "response_format": response_format
                                 }
                         )

        result, completion = self._request_llm(
            prompt=prompt,
            model=model_to_use,
            response_format=response_format,
            temperature=temperature,
            max_tokens=max_tokens,
        )

        if response_format == "json_object":
            if isinstance(result, dict):
                return result
            self.logger.warning("Ожидался словарь для response_format json_object, "
            "но получен другой тип; возвращаем {}."
            )
            return {}

        urls = self.extract_urls_from_response(completion)
        text_result = result if isinstance(result, str) else ""
        return text_result, urls

    def ask_sonar(self, prompt: str, response_format: str = "text",
                  temperature: float = 0.1) -> Any:
        """Удобный метод для модели Sonar.
        Использует модель Perplexity по умолчанию для упрощенного вызова.
        Args:
            prompt: Текст промпта для отправки в модель
            response_format: Формат ответа ("text" или "json_object")
            temperature: Температура для генерации
        Returns:
            Any: Результат вызова ask_perplexity
        """
        return self.ask_perplexity(prompt=prompt,
                                   model=self.config.perplexity_model,
                                   response_format=response_format,
                                   temperature=temperature
                                   )

    def search_info(
        self,
        first_name: str,
        last_name: str,
        additional_info: str,
        birth_date: str | None = None,
        personal_channel_name: str | None = None,
        personal_channel_about: str | None = None,
        response_format: str = "text",
        temperature: float = 0.1,
    ) -> Any:
        """Генерирует краткую аналитическую справку о человеке.
        Формирует промпт на основе предоставленных данных и вызывает
        Perplexity для поиска дополнительной информации в интернете.
        Args:
            first_name: Имя человека
            last_name: Фамилия человека
            additional_info: Дополнительная информация о человеке
            birth_date: Дата рождения (опционально)
            personal_channel_name: Название личного канала (опционально)
            personal_channel_about: Описание личного канала (опционально)
            response_format: Формат ответа ("text" или "json_object")
            temperature: Температура для генерации
        Returns:
            Any: Результат вызова ask_perplexity
        """
        pieces = [
            f"- Имя: {first_name}",
            f"- Фамилия: {last_name}",
            f"- Доп. информация: {additional_info}",
        ]
        if birth_date:
            pieces.append(f"- Дата рождения: {birth_date}")
        if personal_channel_name:
            pieces.append(f"- Название канала: {personal_channel_name}")
        if personal_channel_about:
            pieces.append(f"- Описание канала: {personal_channel_about}")

        prompt = (
            """Составь краткую аналитическую справку о человеке,
    используя имеющиеся данные для поиска дополнительной информации в интернете.
    ## КЛЮЧЕВОЕ ПРАВИЛО:
    Используй ТОЛЬКО содержательные данные.
    Если какое-либо поле содержит маркеры
    **"отсутствует" / "нет данных" / "неизвестно"**
    или пустые фразы
    (**"Tout passe, tout casse, tout lasse", "Тише едешь..." и т.п.**)
    — **полностью игнорируй это поле и не упоминай про его отсутствие.**

    ## ДОПУСТИМЫЕ ФОРМУЛИРОВКИ ПРИ МАЛОМ КОЛИЧЕСТВЕ ДАННЫХ:
    Допускаются осторожные выводы, но только с явными маркерами:
    - "Судя по активности..."
    - "Можно предположить..."
    - "Вероятно связан с..."

    **Запрещено:**
    - приписывать конкретные факты без подтверждения;
    - писать воду вроде «профиль недостаточно раскрыт».

    ## ДАННЫЕ О ЧЕЛОВЕКЕ:"""
            + "\n".join(pieces)
            + """## ФОРМАТ:
    Объём: 3–5 предложений. Профессиональный, но живой стиль.
    Не упоминай то, чего нет. Не делай итоговых рефлексий.
    Просто выдай выжимку фактов и аккуратных наблюдений.

    ## ПРИМЕР ХОРОШЕГО ОТВЕТА:
    Pasha Severilov — специалист с опытом работы в области машинного обучения
    и искусственного интеллекта, ранее связанный с компанией Yandex.
    Судя по доступным данным, он имеет академическую подготовку в области прикладной
    математики и информатики, а также опыт преподавания в ведущих российских вузах.
    """
        ).strip()

        self.logger.debug("Промпт для search_info подготовлен",
                          extra={"prompt_preview": (prompt[:200] + "...")
                                 if len(prompt) > 200 else prompt}
                          )
        return self.ask_perplexity(prompt=prompt, response_format=response_format,
                                   temperature=temperature
                                   )

    def extract_urls_from_response(self, completion_response: Any | None) -> list[str]:
        """Извлекает URL-ы из сырого ответа completion.
        Осторожно обходит структуру ответа для извлечения аннотаций
        и цитат с URL без выбрасывания исключений.
        Args:
            completion_response: Сырой объект ответа от LLM
        Returns:
            List[str]: Список извлеченных URL или пустой список при ошибке
        """
        if completion_response is None:
            self.logger.debug("Не предоставлен completion_response для извлечения URL")
            return []

        urls: list[str] = []
        try:
            choices = getattr(completion_response, "choices", None) or []
            if not choices:
                self.logger.debug("Нет choices в completion_response при извлечении URL")
            for ch in choices:
                message = getattr(ch, "message", None)
                if not message:
                    continue
                annotations = getattr(message, "annotations", None) or []
                for ann in annotations:
                    url_citation = getattr(ann, "url_citation", None)
                    if url_citation:
                        url = getattr(url_citation, "url", None)
                        if url:
                            urls.append(url)
            self.logger.debug("URL извлечены", extra={"urls_count": len(urls)})
            return urls
        except Exception as exc:
            self.logger.error("Ошибка при извлечении URL из completion response",
                              exc_info=exc
                              )
            return []
