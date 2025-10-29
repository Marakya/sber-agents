import logging
from typing import List, Dict
from openai import AsyncOpenAI
from src.config import config

logger = logging.getLogger(__name__)

# Reusable async client configured for OpenRouter
client = AsyncOpenAI(
    api_key=config.openai_api_key,
    base_url=config.openai_base_url,
)

async def get_response(messages: List[Dict[str, str]]) -> str:
    """Call the LLM and return assistant text. Raises on failure."""
    try:
        resp = await client.chat.completions.create(
            model=config.model,
            messages=messages,
            temperature=0.7,
            max_tokens=800,
        )
        return resp.choices[0].message.content
    except Exception:
        logger.exception("Error calling LLM")
        raise
