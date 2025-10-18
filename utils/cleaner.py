import re
from config import EMOJI_PATTERN, URL_PATTERN, ENRU_CHARS_PATTERN

def normalize_empty(value) -> str | None:
    """ '' / ' ' / None → None """
    if value is None:
        return None
    if isinstance(value, str):
        val = value.strip()
        return val if val else None
    return value

def clean_name_field(value) -> str | None:
    """ Удаляем эмодзи, //, |, спецсимволы """
    if not value:
        return None
    value = EMOJI_PATTERN.sub('', value)
    value = ENRU_CHARS_PATTERN.sub('', value)
    value = re.sub(r'[|/\\\[\]{}(),*+=<>^~"]+', ' ', value)
    value = re.sub(r'\s{2,}', ' ', value)
    value = value.strip()
    return value if len(value) >= 2 else None

def clean_second_name_field(value) -> str | None:
    """ Удаляем эмодзи, //, |, спецсимволы """
    if not value:
        return None
    value = EMOJI_PATTERN.sub('', value)
    # value = re.sub(r'[|/\\\[\]{}(),*+=<>^~"]+', ' ', value)
    value = re.sub(r'[\u200b\u200c\u200d\uFEFF]', '', value)
    value = re.sub(r'\s{2,}', ' ', value)
    value = value.strip()
    return value if len(value) >= 2 else None

def should_move_lastname_to_about(last_name):
    if not last_name:
        return False
    return bool(re.search(r'\.com|@|&|http|www', last_name, re.IGNORECASE))

def clean_summary(text):
    cleaned = re.sub(r'\s*\[\d+\]\s*', ' ', text)
    cleaned = re.sub(r'\s+', ' ', cleaned)
    return cleaned.strip()

def merge_about_fields(*fields):
    # Берём уникальные, без эмодзи
    texts = []
    for f in fields:
        if f:
            f = EMOJI_PATTERN.sub('', f)
            f = re.sub(r'\s{2,}', ' ', f)
            f = f.strip()
            if len(f) > 2 and f not in texts:
                texts.append(f)
    return ' | '.join(texts) if texts else None
