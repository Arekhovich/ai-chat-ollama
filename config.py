# config.py
# === КОНФИГУРАЦИЯ ===
# Здесь мы храним настройки отдельно от логики.
import os

# Режимы работы:
# "local"        — локальная модель Ollama на вашем ПК
# "ollama_cloud" — облачная модель внутри Ollama (с суффиксом -cloud)
# "openrouter"   — внешнее облако через OpenRouter (нужен API-ключ)

MODE = "ollama_cloud"  # поменяйте на "ollama_cloud" или "openrouter" чтобы переключить режим

# --- 1) ЛОКАЛЬНАЯ OLLAMA ---
# Те же настройки, что использовались в уроках 2.2–2.5
LOCAL_MODEL = "ollama_chat/qwen2.5:3b"
LOCAL_API_BASE = "http://localhost:11434"

# --- 2) OLLAMA CLOUD (через ту же Ollama, но модель другая) ---
# Важно: api_base остаётся локальным, меняется только имя модели
OLLAMA_CLOUD_MODEL = "ollama_chat/gpt-oss:20b-cloud"
OLLAMA_CLOUD_API_BASE = "http://localhost:11434"

# --- 3) OPENROUTER (внешнее облако) ---
OPENROUTER_MODEL = "openrouter/openrouter/auto"
OPENROUTER_API_BASE = "https://openrouter.ai/api/v1"
OPENROUTER_API_KEY = os.getenv("OPENROUTER_API_KEY")

# Системная инструкция модели (как ей отвечать) — единая для всех режимов
SYSTEM_PROMPT = "Ты Senior Python Developer"
TEMPERATURE = 0.4
MAX_TOKENS = 5000
STREAM_MODE = False
LOG_USAGE = True

# ENABLE_HISTORY: True = модель помнит контекст беседы, False = каждый запрос независим
ENABLE_HISTORY = True

# Максимальное количество сообщений в истории
# Считаются только сообщения user/assistant, system не входит.
# При превышении лимита старые пары user+assistant удаляются первыми.
MAX_HISTORY_MESSAGES = 20

HISTORY_FILE = "history.json"
AUTO_SAVE_HISTORY = True
CONTEXT_MESSAGES_LIMIT = 8



