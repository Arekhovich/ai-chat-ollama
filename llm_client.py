# llm_client.py
import time
from datetime import datetime
import config  # читаем настройки из config.py
from litellm import completion
from litellm.exceptions import (  # исключения — из litellm.exceptions, не из litellm!
    Timeout,
    RateLimitError,
    NotFoundError,
    APIConnectionError,
)


def _get_usage_value(usage, key: str):
    """Безопасно достаёт значение из поля usage (токены).
    Работает как со словарём, так и с объектом — в зависимости от провайдера."""
    if usage is None:
        return None
    if isinstance(usage, dict):
        return usage.get(key)
    return getattr(usage, key, None)


def log_usage(result: dict) -> None:
    """
    Записывает статистику одного успешного вызова в log.txt.
    Формат строки: timestamp | model | tokens | duration

    Вызывается из send_request_to_llm() после каждого успешного ответа.
    Если LOG_USAGE = False в config.py — ничего не записывает.
    """
    if not config.LOG_USAGE:
        return  # логирование выключено — выходим сразу

    ts = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    model = result.get("model", "unknown")
    tokens = result.get("total_tokens")  # может быть None в режиме стриминга
    duration = result.get("duration_sec", 0)

    tokens_str = str(tokens) if tokens is not None else "n/a"

    with open("log.txt", "a", encoding="utf-8") as f:
        # "a" = append: старые записи не стираются, новая дописывается в конец
        f.write(f"{ts} | {model} | tokens={tokens_str} | {duration}s\n")

def send_request_to_llm(
    user_message: str,
    system_prompt: str = config.SYSTEM_PROMPT,
    conversation_history: list | None = None,
    stream_mode: bool | None = None,
    log_request: bool = True
) -> dict | None:
    """
    Отправляет запрос к модели и возвращает словарь с результатом.

    Что изменилось по сравнению с уроком 2.5:
    - функция читает MODE из config.py и сама выбирает нужные model_name / api_base
    - добавлена проверка: если MODE="openrouter" без ключа — сразу возвращаем ошибку no_api_key
    - добавлена обработка ошибок bad_mode и no_api_key
    - всё остальное (messages, completion, метаданные, словарь результата) — без изменений
    """

    # Проверяем: если хотим openrouter, но ключ не задан — сразу говорим об этом
    if config.MODE == "openrouter" and not hasattr(config, "OPENROUTER_API_KEY"):
        return {"error": "no_api_key", "content": None, "new_message": None}

    messages = []
    messages.append({"role": "system", "content": system_prompt})
    if conversation_history:
        messages.extend(conversation_history)
    messages.append({"role": "user", "content": user_message})

    use_stream = config.STREAM_MODE if stream_mode is None else stream_mode

    try:
        # Выбираем параметры подключения в зависимости от режима из config.py
        if config.MODE == "local":
            model_name = config.LOCAL_MODEL
            api_base = config.LOCAL_API_BASE
            headers = None  # локальный режим — без заголовков

        elif config.MODE == "ollama_cloud":
            model_name = config.OLLAMA_CLOUD_MODEL
            api_base = config.OLLAMA_CLOUD_API_BASE
            headers = None  # адрес локальный, заголовки не нужны

        elif config.MODE == "openrouter":
            model_name = config.OPENROUTER_MODEL
            api_base = config.OPENROUTER_API_BASE
            headers = {"Authorization": f"Bearer {config.OPENROUTER_API_KEY}"}

        else:
            # MODE задан неверно — возвращаем понятную ошибку
            return {"error": "bad_mode", "content": None, "new_message": None}

        start_time = time.time()

        if use_stream:
            response_stream = completion(
                model=model_name,
                messages=messages,
                api_base=api_base,
                request_timeout=120,
                headers=headers,
                temperature=config.TEMPERATURE,
                max_tokens=config.MAX_TOKENS,
                stream=True,                   # ключевой параметр — включаем стриминг
            )

            content = ""                       # сюда будем собирать полный ответ
            print("\n🦙 ИИ: ", end="", flush=True)  # печатаем начало без перевода строки

            for chunk in response_stream:      # проходим по потоку чанков
                if hasattr(chunk, "choices") and len(chunk.choices) > 0:
                    delta = chunk.choices[0].delta        # в стриминге текст — в delta, не в message
                    if hasattr(delta, "content") and delta.content:
                        content += delta.content          # сохраняем кусочек в общий ответ
                        print(delta.content, end="", flush=True)  # сразу показываем пользователю

            print()                            # перевод строки после завершения ответа
            duration = time.time() - start_time

            result = {
                "content": content,
                "model": model_name,
                "finish_reason": None,         # в стриминге эти поля обычно недоступны
                "duration_sec": round(duration, 2),
                "prompt_tokens": None,
                "completion_tokens": None,
                "total_tokens": None,
                "error": None,
                "new_message":   {"role": "assistant", "content": content},
            }
            if log_request:
                log_usage(result)  # записываем в лог до return
            return result
        else:
            response = completion(
                model=model_name,
                messages=messages,
                api_base=api_base,
                request_timeout=120,
                headers=headers,  # None для local/ollama_cloud, Bearer-токен для openrouter
                temperature=config.TEMPERATURE,
                max_tokens=config.MAX_TOKENS,
            )

            duration = time.time() - start_time

            content = response.choices[0].message.content
            finish_reason = getattr(response.choices[0], "finish_reason", None)
            usage = getattr(response, "usage", None)

            prompt_tokens = _get_usage_value(usage, "prompt_tokens")
            completion_tokens = _get_usage_value(usage, "completion_tokens")
            total_tokens = _get_usage_value(usage, "total_tokens")

            result = {
                "content": content,
                "model": getattr(response, "model", "unknown"),
                "finish_reason": finish_reason,
                "duration_sec": round(duration, 2),
                "prompt_tokens": prompt_tokens,
                "completion_tokens": completion_tokens,
                "total_tokens": total_tokens,
                "error": None,
                "new_message":   {"role": "assistant", "content": content},
            }
            if log_request:
                log_usage(result)
            return result

    except Timeout:
        return {"error": "timeout", "content": None, "new_message": None}
    except RateLimitError:
        return {"error": "rate_limit", "content": None, "new_message": None}
    except NotFoundError:
        return {"error": "model_not_found", "content": None, "new_message": None}
    except APIConnectionError:
        return {"error": "connection", "content": None, "new_message": None}
    except Exception as e:
        return {"error": str(e), "content": None, "new_message": None}


