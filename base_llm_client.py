import json
import logging
from typing import Any

from config import LlmConfig
from openai import OpenAI


class BaseLLMClient:
    """
    Базовый класс-обёртка для вызовов LLM через OpenAI/OpenRouter.
    Отвечает за:
      - инициализацию клиента
      - выполнение запроса
      - общую обработку ошибок и логирование
      - безопасный разбор JSON-ответов
    Не поднимает исключения наружу — возвращает безопасные пустые значения.
    """

    def __init__(self, config: LlmConfig | None = None) -> None:
        self.config: LlmConfig = config or LlmConfig()
        self.client = OpenAI(base_url=self.config.url, api_key=self.config.key)
        self.logger = logging.getLogger(f"{__name__}.{self.__class__.__name__}")
        self.logger.debug("Initialized BaseLLMClient", extra={"config": self.config})

    def _safe_parse_json(self, raw: str | None) -> dict[str, Any]:
        """Пытается распарсить JSON; в случае ошибки — логирует и возвращает {}."""
        if not raw:
            self.logger.debug("Empty raw content for JSON parsing.")
            return {}
        try:
            parsed = json.loads(raw)
            self.logger.debug("JSON parsed successfully.",
                              extra={"parsed_type": type(parsed).__name__}
                              )
            return parsed if isinstance(parsed, dict) else {"result": parsed}
        except json.JSONDecodeError as exc:
            self.logger.warning("Invalid JSON from LLM.",
                                exc_info=exc, extra={"raw": raw}
                                )
            return {}
        except Exception as exc:
            self.logger.error("Unexpected error while parsing JSON.",
                              exc_info=exc, extra={"raw": raw}
                              )
            return {}

    def _request_llm(
        self,
        prompt: str,
        model: str,
        response_format: str = "text",
        temperature: float = 0.0,
        n: int = 1,
        max_tokens: int | None = None,
        extra_body: dict[str, Any] | None = None,
    ) -> tuple[Any, Any | None]:
        """
        Универсальный метод выполнения запроса к LLM.
        Возвращает кортеж (parsed_content_or_raw, raw_completion_object_or_None).

        - Если response_format == "json_object" — возвращается Dict (или {} при ошибке).
        - Иначе — возвращается str (или "" при ошибке).
        """
        self.logger.debug(
            "Preparing LLM request",
            extra={
                "model": model,
                "response_format": response_format,
                "temperature": temperature,
                "n": n,
                "has_max_tokens": bool(max_tokens),
                "has_extra_body": bool(extra_body),
            },
        )

        try:
            body = extra_body.copy() if extra_body else {}
            if max_tokens:
                body["max_tokens"] = max_tokens

            rf = {"type": response_format} if response_format == "json_object" else None

            self.logger.debug("Calling OpenAI.chat.completions.create",
                    extra={"body_preview": {k: body.get(k) for k in list(body)[:5]}}
                    )
            completion = self.client.chat.completions.create(
                model=model,
                messages=[{"role": "user", "content": prompt}],
                response_format=rf,
                temperature=temperature,
                n=n,
                extra_body=body or None,
            )

            content = getattr(getattr(completion, "choices", [None])[0], "message", None)
            raw_text: str | None = None
            if content is not None:
                raw_text = getattr(content, "content", None)
            self.logger.debug("Received LLM raw response",
                extra={"raw_text_preview": (raw_text[:500] if raw_text else None)}
                )

            if response_format == "json_object":
                parsed = self._safe_parse_json(raw_text)
                self.logger.info("LLM returned JSON object",
                                 extra={"parsed_keys": list(parsed.keys())}
                                 )
                return parsed, completion

            text_result = raw_text or ""
            self.logger.info("LLM returned text response",
                             extra={"length": len(text_result)}
                             )
            return text_result, completion

        except Exception as exc:
            self.logger.debug("Exception details for LLM request", exc_info=exc)
            self.logger.error("Ошибка при вызове LLM. Возвращаем безопасное значение.",
                              exc_info=False
                              )
            if response_format == "json_object":
                return {}, None
            return "", None
