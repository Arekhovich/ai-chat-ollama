import json
import os
import config
from llm_client import send_request_to_llm


def load_history(filepath: str) -> list:
    """
    Загружает историю диалога из JSON-файла.
    Если файла нет или он повреждён — возвращает пустой список.
    """
    if not os.path.exists(filepath):
        return []

    try:
        with open(filepath, "r", encoding="utf-8") as f:
            history = json.load(f)

        if not isinstance(history, list):
            return []

        cleaned_history = []
        for item in history:
            if (
                    isinstance(item, dict)
                    and item.get("role") in ("user", "assistant")
                    and isinstance(item.get("content"), str)
            ):
                cleaned_history.append(
                    {"role": item["role"], "content": item["content"]}
                )

        return cleaned_history

    except (json.JSONDecodeError, OSError):
        return []


def save_history(filepath: str, history: list) -> None:
    """
    Сохраняет историю диалога в JSON-файл.
    """
    try:
        with open(filepath, "w", encoding="utf-8") as f:
            json.dump(history, f, ensure_ascii=False, indent=2)
    except OSError as e:
        print(f"\n⚠️ Не удалось сохранить историю: {e}\n")

def get_context_window(history: list, limit: int) -> list:
    """
    Возвращает только последние сообщения из истории
    для отправки в модель.

    Не удаляет ничего из самой истории — только делает срез.
    history[-limit:] возвращает последние limit элементов.
    Если история короче limit — вернёт весь список.
    """
    if not history or limit <= 0:
        return []

    return history[-limit:]

def build_summary_message(summary_text: str) -> dict:
    """
    Превращает текст сводки в сообщение,
    которое можно передать модели как часть контекста.
    """
    return {
        "role": "assistant",
        "content": f"Краткое резюме предыдущей части диалога: {summary_text}"
    }

def summarize_old_history(old_history: list) -> str:
    """
    Создаёт краткую сводку старой части истории.
    Если что-то пошло не так — возвращает пустую строку.
    """
    if not old_history:
        return ""

    result = send_request_to_llm(
        user_message=(
            "Сделай краткое резюме старой части диалога. "
            "Сохрани только важные факты о пользователе, цели, предпочтения, "
            "незавершённые темы и полезный контекст для продолжения разговора."
        ),
        system_prompt=config.SUMMARY_SYSTEM_PROMPT,
        conversation_history=old_history,
        stream_mode=False,
        log_request=False
    )

    if result and result.get("error") is None and result.get("content"):
        return result["content"].strip()

    return ""

def build_history_for_request(history: list, summary_text: str) -> list | None:
    """
    Собирает рабочий контекст для нового запроса:
    сначала сводка, потом последние сообщения.
    """
    context = []

    if summary_text:
        context.append(build_summary_message(summary_text))

    context.extend(history)

    return context if context else None


def print_response_details(result: dict) -> None:
    """Печатает текст ответа и метаданные. Используется в обычном режиме."""
    print(f"\n🦙 ИИ: {result['content']}")
    print_response_meta(result)


def print_response_meta(result: dict) -> None:
    """Печатает только метаданные. Используется в потоковом режиме."""
    print("\n📊 Метаданные:")

    mode_label = {
        "local": "🏠 Локальная Ollama (ваш ПК)",
        "ollama_cloud": "☁️ Ollama Cloud (через Ollama)",
        "openrouter": "☁️ OpenRouter (внешнее облако)",
    }.get(config.MODE, "❓ Неизвестный режим")

    print(f"   Режим: {mode_label}")
    print(f"   Модель: {result['model']}")
    print(f"   Время генерации: {result['duration_sec']} сек")
    print(f"   Причина остановки: {result['finish_reason']}")

    if result["total_tokens"] is None:
        print("   Токены: недоступно (для некоторых конфигураций это нормально)")
    else:
        print(f"   Токены запроса: {result['prompt_tokens']}")
        print(f"   Токены ответа: {result['completion_tokens']}")
        print(f"   Всего токенов: {result['total_tokens']}")

    print("-" * 40)


def print_error_message(error_type: str) -> None:
    messages = {
        "timeout": "\n⏱ Таймаут! Модель не ответила вовремя. Упростите запрос или увеличьте request_timeout.",
        "rate_limit": "\n🚧 Слишком много запросов за короткое время. Подождите 1–2 минуты.",
        "model_not_found": "\n🤷 Модель не найдена. Проверьте название модели и список: ollama list",
        "connection": "\n🔌 Сервер недоступен. Для Ollama проверьте, что она запущена: http://localhost:11434",
        "no_api_key": "\n🔑 API-ключ не настроен. Добавьте OPENROUTER_API_KEY в config.py.",
        "bad_mode": "\n⚙️ Неверный MODE в config.py. Используйте: local / ollama_cloud / openrouter.",
    }
    default_msg = "\n❌ Неизвестная ошибка. Попробуйте снова."
    print(messages.get(error_type, default_msg))


def main() -> None:
    print("🦙 AI-помощник (с памятью контекста) запущен!\n")
    print(
        f"📊 Режим: {config.MODE} | "
        f"history={config.ENABLE_HISTORY} | "
        f"max_history={config.MAX_HISTORY_MESSAGES} | "
        f"context_limit={config.CONTEXT_MESSAGES_LIMIT} | "
        f"summary={config.ENABLE_SUMMARY} | "
        f"summary_trigger={config.SUMMARY_TRIGGER_MESSAGES} | "
        f"auto_save={config.AUTO_SAVE_HISTORY} | "
        f"stream={config.STREAM_MODE} | "
        f"log={config.LOG_USAGE}\n"
    )

    print("Введите ваш вопрос (или 'выход' для завершения):\n")

    conversation_history = []

    if config.ENABLE_HISTORY:
        conversation_history = load_history(config.HISTORY_FILE)

        if conversation_history:
            print(f"📚 Загружено сообщений из файла: {len(conversation_history)}\n")
        else:
            print("📚 История пуста\n")

    conversation_summary = ""

    total_requests = 0
    total_tokens = 0

    while True:
        user_input = input("👤 Вы: ").strip()

        if user_input.lower() in ("выход", "exit", "quit"):
            print("\n👋 До встречи!")

            if total_requests > 0:
                print("\n📊 Статистика сессии:")
                print(f"   Запросов: {total_requests}")
                if total_tokens > 0:
                    avg = total_tokens // total_requests
                    print(f"   Токенов всего: {total_tokens}")
                    print(f"   В среднем: {avg} токенов/запрос")
                else:
                    print("   Токены: n/a (стриминг или модель не вернула usage)")
                print(f"   Сообщений в истории: {len(conversation_history)}")
                print(f"   Сводка используется: {'да' if conversation_summary else 'нет'}")

            break

        if user_input.lower() == "/clear":
            conversation_history.clear()

            if config.ENABLE_HISTORY:
                save_history(config.HISTORY_FILE, conversation_history)

            print("\n🧹 История диалога очищена\n")
            continue

        if not user_input:
            print("⚠️ Введите текст вопроса\n")
            continue

        print("\n🤔 Модель думает...")

        if config.ENABLE_HISTORY:
            recent_history = get_context_window(
                conversation_history,
                config.CONTEXT_MESSAGES_LIMIT
            )

            if (
                config.ENABLE_SUMMARY
                and len(conversation_history) > config.SUMMARY_TRIGGER_MESSAGES
                and len(conversation_history) > config.CONTEXT_MESSAGES_LIMIT
            ):
                old_history = conversation_history[:-config.CONTEXT_MESSAGES_LIMIT]
                conversation_summary = summarize_old_history(old_history)

            history_to_send = build_history_for_request(
                recent_history,
                conversation_summary
            )
        else:
            history_to_send = None

        result = send_request_to_llm(
            user_input,
            conversation_history=history_to_send
        )

        if result is None:
            print("\n⚠️ Не удалось получить ответ.\n")

        elif result["error"] is not None:
            print_error_message(result["error"])

        else:
            total_requests += 1
            if result["total_tokens"] is not None:
                total_tokens += result["total_tokens"]

            if config.STREAM_MODE:
                print_response_meta(result)
            else:
                print_response_details(result)

            if config.ENABLE_HISTORY and result.get("new_message"):
                conversation_history.append({"role": "user", "content": user_input})
                conversation_history.append(result["new_message"])

                while len(conversation_history) > config.MAX_HISTORY_MESSAGES:
                    conversation_history.pop(0)
                    if conversation_history:
                        conversation_history.pop(0)

                if config.AUTO_SAVE_HISTORY:
                    save_history(config.HISTORY_FILE, conversation_history)


if __name__ == "__main__":
    main()
