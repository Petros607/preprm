import datetime
import logging
from pathlib import Path


class MarkdownExporter:
    """Экспортер данных в формат Markdown.
    Класс для создания Markdown файлов с информацией о людях,
    включая краткие справки, ссылки на источники и личные каналы.
    Attributes:
        output_dir (Path): Директория для сохранения отчетов
        logger: Логгер для записи событий
    """

    def __init__(self, output_dir: str = "reports") -> None:
        """Инициализация экспортера Markdown.
        Args:
            output_dir: Директория для сохранения отчетов.
                       По умолчанию "reports".
        """
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(exist_ok=True)
        self.logger = logging.getLogger(__name__)
        self.logger.debug(f"MarkdownExporter инициализирован с директорией: {output_dir}")

    def create_filename(self, first_name: str,
                        last_name: str, extension: str = "md"
                        ) -> str:
        """Создает имя файла в формате: ГГГГ-ММ-ДД_Фамилия_Имя.md.
        Args:
            first_name: Имя человека
            last_name: Фамилия человека
            extension: Расширение файла. По умолчанию "md"
        Returns:
            str: Имя файла в формате дата_фамилия_имя.расширение
        """
        date_str = datetime.datetime.now().strftime("%Y-%m-%d")
        name_clean = self._sanitize_filename(f"{last_name}_{first_name}")
        filename = f"{date_str}_{name_clean}.{extension}"

        self.logger.debug(f"Сгенерировано имя файла: {filename}")
        return filename

    def _sanitize_filename(self, filename: str) -> str:
        """Очищает имя файла от недопустимых символов.
        Args:
            filename: Исходное имя файла
        Returns:
            str: Очищенное имя файла
        """
        sanitized = filename.replace(" ", "_").replace("/", "-")
        sanitized = sanitized.replace("\\", "-").replace(":", "-")
        sanitized = sanitized.replace("*", "-").replace("?", "-")
        sanitized = sanitized.replace("\"", "-").replace("<", "-")
        sanitized = sanitized.replace(">", "-").replace("|", "-")

        return sanitized

    def export_to_md(self,
                    first_name: str,
                    last_name: str,
                    content: str,
                    urls: list[str],
                    personal_channel: str = "") -> str:
        """Создает MD файл с информацией о человеке.
        Args:
            first_name: Имя человека
            last_name: Фамилия человека
            content: Текстовая информация о человеке
            urls: Список URL источников информации
            personal_channel: Username личного Telegram-канала
        Returns:
            str: Путь к созданному файлу или пустая строка в случае ошибки
        """
        self.logger.info(
            f"Начало экспорта данных для {first_name} {last_name}. "
            f"Источников: {len(urls)}, канал: {personal_channel or 'не указан'}"
        )

        filename = self.create_filename(first_name, last_name)
        filepath = self.output_dir / filename

        try:
            md_content = self._generate_md_content(
                first_name, last_name, content, urls, personal_channel
            )

            self._write_md_file(filepath, md_content)
            self.logger.info(f"Успешно создан файл: {filepath}")
            return str(filepath)

        except Exception as e:
            self.logger.error(f"Ошибка создания файла {filename}: {e}")
            return ""

    def _generate_md_content(self,
                           first_name: str,
                           last_name: str,
                           content: str,
                           urls: list[str],
                           personal_channel: str = "") -> str:
        """Генерирует содержимое MD файла.
        Args:
            first_name: Имя человека
            last_name: Фамилия человека
            content: Текстовая информация о человеке
            urls: Список URL источников информации
            personal_channel: Username личного Telegram-канала
        Returns:
            str: Содержимое Markdown файла
        """
        date_str = datetime.datetime.now().strftime("%Y-%m-%d %H:%M")

        md_parts = [
            self._generate_header(first_name, last_name),
            self._generate_summary_section(content),
            self._generate_channel_section(personal_channel),
            self._generate_sources_section(urls),
            self._generate_footer(date_str)
        ]

        return "\n".join(filter(None, md_parts))

    def _generate_header(self, first_name: str, last_name: str) -> str:
        """Генерирует заголовок Markdown файла.
        Args:
            first_name: Имя человека
            last_name: Фамилия человека
        Returns:
            str: Заголовок в формате Markdown
        """
        return f"""# {first_name} {last_name}

---

"""

    def _generate_summary_section(self, content: str) -> str:
        """Генерирует раздел с краткой справкой.
        Args:
            content: Текстовая информация о человеке
        Returns:
            str: Раздел справки в формате Markdown
        """
        return f"""## 📋 Краткая справка

{content}

"""

    def _generate_channel_section(self, personal_channel: str) -> str | None:
        """Генерирует раздел с информацией о Telegram-канале.
        Args:
            personal_channel: Username личного Telegram-канала
        Returns:
            str: Раздел канала в формате Markdown или None если канал не указан
        """
        if not personal_channel:
            return None

        return f"""## 📢 Telegram-канал

[{personal_channel}](https://t.me/{personal_channel})

"""

    def _generate_sources_section(self, urls: list[str]) -> str:
        """Генерирует раздел с источниками информации.
        Args:
            urls: Список URL источников информации
        Returns:
            str: Раздел источников в формате Markdown
        """
        sources_section = f"""## 🔗 Источники информации

Всего найдено источников: **{len(urls)}**

"""

        for i, url in enumerate(urls, 1):
            domain = self._extract_domain(url)
            sources_section += f"{i}. [{domain}]({url})\n"

        return sources_section

    def _extract_domain(self, url: str) -> str:
        """Извлекает домен из URL.
        Args:
            url: URL для обработки
        Returns:
            str: Домен из URL
        """
        try:
            domain = url.split('//')[-1].split('/')[0]
            return domain
        except Exception as e:
            self.logger.warning(f"Ошибка извлечения домена из URL {url}: {e}")
            return url

    def _generate_footer(self, date_str: str) -> str:
        """Генерирует подвал Markdown файла.
        Args:
            date_str: Строка с датой и временем
        Returns:
            str: Подвал в формате Markdown
        """
        return f"""
---

*Сгенерировано с помощью Perplexity AI: {date_str}*
"""

    def _write_md_file(self, filepath: Path, content: str) -> None:
        """Записывает содержимое в Markdown файл.
        Args:
            filepath: Путь к файлу для записи
            content: Содержимое для записи
        Raises:
            IOError: Если произошла ошибка записи файла
            Exception: Другие ошибки при записи
        """
        try:
            filepath.write_text(content, encoding='utf-8')
            self.logger.debug(f"Файл успешно записан: {filepath}")
        except OSError as e:
            self.logger.error(f"Ошибка ввода-вывода при записи файла {filepath}: {e}")
            raise
        except Exception as e:
            self.logger.error(f"Неожиданная ошибка при записи файла {filepath}: {e}")
            raise

    def get_output_directory(self) -> Path:
        """Возвращает путь к директории для сохранения отчетов.
        Returns:
            Path: Путь к директории отчетов
        """
        return self.output_dir
