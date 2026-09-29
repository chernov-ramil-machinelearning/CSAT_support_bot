import os
import json
import time
import asyncio
from pathlib import Path
from dotenv import load_dotenv

from aiogram import Bot, Dispatcher, F
from aiogram.filters import CommandStart
from aiogram.types import Message, CallbackQuery, InlineKeyboardMarkup, InlineKeyboardButton
from aiogram.client.session.aiohttp import AiohttpSession
from proxy import generate_answer

BASE_DIR = Path(__file__).resolve().parent
load_dotenv(BASE_DIR / ".env")
TOKEN = os.getenv("BOT_TOKEN")
PROXY = os.getenv("TELEGRAM_PROXY")

if not TOKEN:
    raise ValueError("Токен BOT_TOKEN не найден в .env файле")

LOG_FILE = BASE_DIR / "feedback_log.jsonl"

if PROXY:
    session = AiohttpSession(proxy=PROXY)
    bot = Bot(token=TOKEN, session=session)
else:
    for var in ("HTTP_PROXY", "HTTPS_PROXY", "ALL_PROXY", "http_proxy", "https_proxy", "all_proxy"):
        os.environ.pop(var, None)
    bot = Bot(token=TOKEN)

dp = Dispatcher()

# Семафор: на одноплатнике только 1 генерация единовременно
inference_lock = asyncio.Lock()

# Хранилище последних ответов в памяти: message_id -> {question, answer, user_id}
dialog_cache: dict[int, dict] = {}


def get_csat_keyboard() -> InlineKeyboardMarkup:
    return InlineKeyboardMarkup(
        inline_keyboard=[
            [
                InlineKeyboardButton(text="👍 Помогло", callback_data="csat:like"),
                InlineKeyboardButton(text="👎 Не помогло", callback_data="csat:dislike"),
            ]
        ]
    )


def log_feedback(user_id: int, question: str, answer: str, score: int, feedback: str) -> None:
    entry = {
        "timestamp": int(time.time()),
        "user_id": user_id,
        "question": question,
        "answer": answer,
        "score": score,
        "feedback": feedback,
    }
    with open(LOG_FILE, "a", encoding="utf-8") as f:
        f.write(json.dumps(entry, ensure_ascii=False) + "\n")


@dp.message(CommandStart())
async def handle_start(message: Message) -> None:
    if message.chat.type != "private":
        return
    print(f"\n[Telegram] Пользователь {message.from_user.id} нажал /start")
    await message.answer(
        "Привет! Я бот поддержки, который собирает CSAT-метрику.\n\n"
        "Задай мне вопрос, а после ответа нажми кнопку с оценкой — это поможет улучшать ответы."
    )


@dp.message(F.text)
async def handle_message(message: Message) -> None:
    # Строго игнорируем любые группы и каналы — работаем только в личке
    if message.chat.type != "private":
        return

    user_text = message.text.strip()
    if not user_text:
        return

    print(f"\n[Telegram] Вопрос от {message.from_user.id}: {user_text}")

    # Индикация набора текста без краша от лимитов Telegram
    try:
        await bot.send_chat_action(chat_id=message.chat.id, action="typing")
    except Exception:
        pass

    if inference_lock.locked():
        wait_msg = await message.answer("⏳ Сейчас генерирую ответ на другой запрос, подождите...")
    else:
        wait_msg = None

    async with inference_lock:
        if wait_msg:
            try:
                await wait_msg.delete()
            except Exception:
                pass
        print("[Telegram] Генерация ответа в ONNX...")
        t0 = time.time()
        answer = await asyncio.to_thread(generate_answer, user_text)
        print(f"[Telegram] Ответ сгенерирован за {time.time() - t0:.2f}с: {answer[:60]}...")

    sent_msg = await message.answer(
        text=answer,
        reply_markup=get_csat_keyboard()
    )

    dialog_cache[sent_msg.message_id] = {
        "user_id": message.from_user.id if message.from_user else 0,
        "question": user_text,
        "answer": answer,
    }


@dp.callback_query(F.data.startswith("csat:"))
async def handle_csat_feedback(callback: CallbackQuery) -> None:
    action = callback.data.split(":")[1]
    score = 1 if action == "like" else 0
    feedback_text = "👍 Оценка учтена, спасибо!" if score == 1 else "👎 Зафиксировали ошибку, учтём!"

    msg_id = callback.message.message_id if callback.message else None
    dialog_data = dialog_cache.pop(msg_id, None) if msg_id else None

    if dialog_data:
        log_feedback(
            user_id=dialog_data["user_id"],
            question=dialog_data["question"],
            answer=dialog_data["answer"],
            score=score,
            feedback=action,
        )
    elif callback.message and callback.message.text:
        log_feedback(
            user_id=callback.from_user.id if callback.from_user else 0,
            question="[Предыдущая сессия]",
            answer=callback.message.text,
            score=score,
            feedback=action,
        )

    try:
        if callback.message:
            await callback.message.edit_reply_markup(reply_markup=None)
        await callback.answer(feedback_text)
        if callback.message:
            await callback.message.reply(feedback_text)
    except Exception:
        pass


async def main() -> None:
    print("Бот запущен. Сбрасываем старый бэклог и слушаем только ЛС...")
    await dp.start_polling(bot, drop_pending_updates=True)


if __name__ == "__main__":
    asyncio.run(main())
