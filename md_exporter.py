import datetime
from pathlib import Path
import logging

class MarkdownExporter:
    def __init__(self, output_dir: str = "reports"):
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(exist_ok=True)
        self.logger = logging.getLogger(__name__)
    
    def create_filename(self, first_name: str, last_name: str, extension: str = "md") -> str:
        """Создает имя файла в формате: ГГГГ-ММ-ДД_Фамилия_Имя.md"""
        date_str = datetime.datetime.now().strftime("%Y-%m-%d")
        name_clean = f"{last_name}_{first_name}".replace(" ", "_").replace("/", "-")
        return f"{date_str}_{name_clean}.{extension}"
    
    def export_to_md(self, 
                    first_name: str, 
                    last_name: str, 
                    content: str, 
                    urls: list,
                    personal_channel: str = "") -> str:
        """
        Создает MD файл с информацией о человеке
        """
        filename = self.create_filename(first_name, last_name)
        filepath = self.output_dir / filename
        
        md_content = self._generate_md_content(
            first_name, last_name, content, urls, personal_channel
        )
        
        try:
            with open(filepath, 'w', encoding='utf-8') as f:
                f.write(md_content)
            
            self.logger.info(f"Создан файл: {filepath}")
            return str(filepath)
            
        except Exception as e:
            self.logger.error(f"Ошибка создания файла {filename}: {e}")
            return ""
    
    def _generate_md_content(self, first_name, last_name, content, urls, personal_channel):
        """Генерирует содержимое MD файла"""
        date_str = datetime.datetime.now().strftime("%Y-%m-%d %H:%M")
        
        md = f"""# {first_name} {last_name}
---
## 📋 Краткая справка
{content}
"""
        if personal_channel:
            md += f"""## 📢 Telegram-канал
[{personal_channel}](https://t.me/{personal_channel})
"""
        md += f"""## 🔗 Источники информации
Всего найдено источников: **{len(urls)}**
"""
        for i, url in enumerate(urls, 1):
            domain = url.split('//')[-1].split('/')[0]
            md += f"{i}. [{domain}]({url})\n"
        md += f"""
---
*Сгенерировано с помощью Perplexity AI: {date_str}**
"""
        return md
    