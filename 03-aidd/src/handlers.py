import logging
from aiogram import Router
from aiogram.filters import Command
from aiogram.types import Message
from src.llm import get_response
from src.config import config

logger = logging.getLogger(__name__)
router = Router()

# Глобальный словарь для хранения историй диалогов
chat_conversations: dict[int, list[dict]] = {}

# Максимальная длина сообщения пользователя
MAX_MESSAGE_LENGTH = 4000

@router.message(Command("start"))
async def cmd_start(message: Message):
    logger.info(f"User {message.chat.id} started the bot")
    
    # Инициализируем историю с системным промптом
    chat_conversations[message.chat.id] = [
        {"role": "system", "content": config.SYSTEM_PROMPT}
    ]
    
    await message.answer(
        "Привет! Я банковский ассистент.\n\n"
        "Я могу:\n"
        "• Отвечать на вопросы по банковским услугам\n"
        "• Помогать с финансами\n"
        "• Поддерживать диалог с учетом контекста\n\n"
        "Используйте /start для начала нового диалога."
    )

@router.message()
async def handle_message(message: Message):
    # Игнорируем сообщения без текста (стикеры, фото и т.д.)
    if not message.text:
        await message.answer("Извините, я работаю только с текстовыми сообщениями.")
        return
    
    # Проверяем длину сообщения
    if len(message.text) > MAX_MESSAGE_LENGTH:
        await message.answer(
            f"Извините, ваше сообщение слишком длинное ({len(message.text)} символов). "
            f"Максимальная длина: {MAX_MESSAGE_LENGTH} символов."
        )
        return
    
    logger.info(f"Message from {message.chat.id}: {message.text[:100]}...")
    
    # Инициализируем историю если её нет
    if message.chat.id not in chat_conversations:
        chat_conversations[message.chat.id] = [
            {"role": "system", "content": config.SYSTEM_PROMPT}
        ]
    
    # Добавляем сообщение пользователя в историю
    chat_conversations[message.chat.id].append(
        {"role": "user", "content": message.text}
    )
    
    try:
        # Получаем ответ LLM со всей историей
        response = await get_response(chat_conversations[message.chat.id])
        
        # Добавляем ответ LLM в историю
        chat_conversations[message.chat.id].append(
            {"role": "assistant", "content": response}
        )
        
        await message.answer(response)
    except Exception as e:
        logger.error(f"Error in handle_message: {e}")
        await message.answer(
            "Произошла ошибка при обработке вашего сообщения. "
            "Попробуйте еще раз или используйте /start для начала нового диалога."
        )

