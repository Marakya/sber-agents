import logging
from openai import AsyncOpenAI
from src.config import config

logger = logging.getLogger(__name__)

client = AsyncOpenAI(
    api_key=config.OPENAI_API_KEY,
    base_url=config.OPENAI_BASE_URL
)

async def get_response(messages: list[dict]) -> str:
    try:
        response = await client.chat.completions.create(
            model=config.MODEL,
            messages=messages
        )
        return response.choices[0].message.content
    except Exception as e:
        logger.error(f"Error calling LLM: {e}")
        return "Извините, произошла ошибка при обращении к LLM. Попробуйте позже."

