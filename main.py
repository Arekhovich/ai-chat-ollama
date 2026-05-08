import json
import os
import config
from llm_client import send_request_to_llm


def load_memory_state(filepath: str) -> tuple[list, str]:
    """
    Загружает память приложения из JSON-файла.
    Возвращает кортеж (history, summary).
    Поддерживает старый формат: если в файле просто список — это история, сводка пустая.
    """
    if not os.path.exists(filepath):
        return [], ""

    try:
        with open(filepath, "r", encoding="utf-8") as f:
            data = json.load(f)

        # Поддержка старого формата (history.json из урока 4.2–4.5)
        if isinstance(data, list):
            cleaned_history = []
            for item in data:
                if (
                    isinstance(item, dict)
                    and item.get("role") in ("user", "assistant")
                    and isinstance(item.get("content"), str)
                ):
                    cleaned_history.append(
                        {"role": item["role"], "content": item["content"]}
                    )
            return cleaned_history, ""

        if not isinstance(data, dict):
            return [], ""

        raw_history = data.get("history", [])
        raw_summary = data.get("summary", "")

        if not isinstance(raw_history, list):
            raw_history = []
        if not isinstance(raw_summary, str):
            raw_summary = ""

        cleaned_history = []
        for item in raw_history:
            if (
                isinstance(item, dict)
                and item.get("role") in ("user", "assistant")
                and isinstance(item.get("content"), str)
            ):
                cleaned_history.append(
                    {"role": item["role"], "content": item["content"]}
                )

        return cleaned_history, raw_summary

    except (json.JSONDecodeError, OSError):
        return [], ""


def save_memory_state(filepath: str, history: list, summary: str) -> None:
    """
    Сохраняет состояние памяти приложения в JSON-файл.
    Формат: {"history": [...], "summary": "..."}
    """
    data = {
        "history": history,
        "summary": summary,
    }

    try:
        with open(filepath, "w", encoding="utf-8") as f:
            json.dump(data, f, ensure_ascii=False, indent=2)
    except OSError as e:
        print(f"\n⚠️ Не удалось сохранить память: {e}\n")


def get_context_window(history: list, limit: int) -> list:
    """
    Возвращает только последние сообщения из истории,
    которые нужно отправить модели.
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


def summarize_old_history(old_history: list, previous_summary: str = "") -> str:
    """
    Создаёт обновлённую сводку на основе старой части истории
    и уже существующей сводки (если есть).
    При ошибке возвращает previous_summary без изменений.
    """
    if not old_history:
        return previous_summary

    summary_context = []
    if previous_summary:
        summary_context.append(build_summary_message(previous_summary))
    summary_context.extend(old_history)

    result = send_request_to_llm(
        user_message=(
            "Сделай краткое резюме прошлой части диалога. "
            "Сохрани только важные факты о пользователе, цели, предпочтения, "
            "незавершённые темы и полезный контекст для продолжения беседы. "
            "Ответ дай короткой связной сводкой без приветствия."
        ),
        system_prompt=config.SUMMARY_SYSTEM_PROMPT,
        conversation_history=summary_context,
        stream_mode=False,
        log_request=False,
        expect_json=False
    )

    if result and result.get("error") is None and result.get("content"):
        return result["content"].strip()

    return previous_summary


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


def compress_history_if_needed(
    history: list, summary_text: str
) -> tuple[list, str]:
    """
    Если история превысила порог — сжимает старую часть в сводку.
    Возвращает (обновлённая_история, обновлённая_сводка).
    Если сжатие не нужно или не удалось — возвращает исходные данные.
    """
    if not config.ENABLE_SUMMARY:
        return history, summary_text

    if len(history) <= config.SUMMARY_TRIGGER_MESSAGES:
        return history, summary_text

    keep_count = config.RECENT_MESSAGES_AFTER_SUMMARY

    if keep_count <= 0:
        old_part = history
        recent_part = []
    else:
        old_part = history[:-keep_count]
        recent_part = history[-keep_count:]

    if not old_part:
        return history, summary_text

    new_summary = summarize_old_history(old_part, summary_text)

    if new_summary:
        return recent_part, new_summary

    return history, summary_text


def print_json_result(json_data) -> None:
    """
    Красиво печатает разобранный JSON-ответ модели.
    """
    print("\n🧩 JSON-ответ:")
    print(json.dumps(json_data, ensure_ascii=False, indent=2))


def print_response_details(result: dict) -> None:
    print(f"\n🦙 ИИ: {result['content']}")
    print_response_meta(result)


def print_response_meta(result: dict) -> None:
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
        "timeout":         "\n⏱ Таймаут! Модель не ответила вовремя. Упростите запрос или увеличьте request_timeout.",
        "rate_limit":      "\n🚧 Слишком много запросов за короткое время. Подождите 1–2 минуты.",
        "model_not_found": "\n🤷 Модель не найдена. Проверьте название модели и список: ollama list",
        "connection":      "\n🔌 Сервер недоступен. Для Ollama проверьте, что она запущена и доступен http://localhost:11434",
        "no_api_key":      "\n🔑 API-ключ не настроен. Добавьте OPENROUTER_API_KEY в config.py (MODE=openrouter).",
        "bad_mode":        "\n⚙️ Неверный MODE в config.py. Используйте: local / ollama_cloud / openrouter.",
    }
    default_msg = "\n❌ Неизвестная ошибка. Попробуйте снова."
    print(messages.get(error_type, default_msg))


def main() -> None:
    print("🦙 AI-помощник (с памятью и JSON-режимом) запущен!\n")
    print(
        f"📊 Режим: {config.MODE} | "
        f"history={config.ENABLE_HISTORY} | "
        f"max_history={config.MAX_HISTORY_MESSAGES} | "
        f"context_limit={config.CONTEXT_MESSAGES_LIMIT} | "
        f"summary={config.ENABLE_SUMMARY} | "
        f"summary_trigger={config.SUMMARY_TRIGGER_MESSAGES} | "
        f"recent_after_summary={config.RECENT_MESSAGES_AFTER_SUMMARY} | "
        f"auto_save={config.AUTO_SAVE_MEMORY} | "
        f"json_mode={config.ENABLE_JSON_MODE} | "
        f"stream={config.STREAM_MODE} | "
        f"log={config.LOG_USAGE}\n"
    )

    conversation_history = []
    conversation_summary = ""

    if config.ENABLE_HISTORY:
        conversation_history, conversation_summary = load_memory_state(config.MEMORY_FILE)

        if conversation_history or conversation_summary:
            print(
                f"📚 Память загружена: "
                f"сообщений={len(conversation_history)}, "
                f"сводка={'есть' if conversation_summary else 'нет'}\n"
            )
        else:
            print("📚 Память пуста\n")

    print("Введите ваш вопрос (или 'выход' для завершения):\n")

    total_requests = 0
    total_tokens   = 0

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

                print(f"   Сообщений в свежей истории: {len(conversation_history)}")
                print(f"   Сводка памяти: {'есть' if conversation_summary else 'нет'}")

            break

        if user_input.lower() == "/clear":
            conversation_history.clear()
            conversation_summary = ""

            if config.ENABLE_HISTORY:
                save_memory_state(
                    config.MEMORY_FILE,
                    conversation_history,
                    conversation_summary
                )

            print("\n🧹 История диалога и сводка памяти очищены\n")
            continue

        if user_input.lower() == "/memory":
            if conversation_summary:
                print(f"\n🧠 Сводка памяти:\n{conversation_summary}\n")
            else:
                print("\n🧠 Сводка памяти пока пуста\n")
            continue

        if user_input.lower() == "/save":
            if config.ENABLE_HISTORY:
                save_memory_state(
                    config.MEMORY_FILE,
                    conversation_history,
                    conversation_summary
                )
                print(f"\n💾 Память сохранена в {config.MEMORY_FILE}\n")
            continue

        if not user_input:
            print("⚠️ Введите текст вопроса\n")
            continue

        print("\n🤔 Модель думает...")

        if config.ENABLE_HISTORY:
            history_to_send = build_history_for_request(
                get_context_window(conversation_history, config.CONTEXT_MESSAGES_LIMIT),
                conversation_summary
            )
        else:
            history_to_send = None

        result = send_request_to_llm(
            user_input,
            conversation_history=history_to_send,
            expect_json=config.ENABLE_JSON_MODE
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

            if config.ENABLE_JSON_MODE:
                if result.get("json_data") is not None:
                    print_json_result(result["json_data"])
                else:
                    print("\n⚠️ Модель ответила текстом, но JSON разобрать не удалось.")

            if config.ENABLE_HISTORY and result.get("new_message"):
                conversation_history.append({"role": "user", "content": user_input})
                conversation_history.append(result["new_message"])

                conversation_history, conversation_summary = compress_history_if_needed(
                    conversation_history,
                    conversation_summary
                )

                while len(conversation_history) > config.MAX_HISTORY_MESSAGES:
                    conversation_history.pop(0)
                    if conversation_history:
                        conversation_history.pop(0)

                if config.AUTO_SAVE_MEMORY:
                    save_memory_state(
                        config.MEMORY_FILE,
                        conversation_history,
                        conversation_summary
                    )


if __name__ == "__main__":
    main()
