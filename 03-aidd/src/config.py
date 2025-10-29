import os
from dotenv import load_dotenv

load_dotenv()

class Config:
    """App configuration loaded from environment variables."""
    def __init__(self):
        self.telegram_token = self._get_required("TELEGRAM_TOKEN")
        # API key used for Openrouter (stored in env)
        self.openai_api_key = self._get_required("OPENAI_API_KEY")
        self.openai_base_url = os.getenv("OPENAI_BASE_URL", "https://openrouter.ai/api/v1")
        self.model = os.getenv("MODEL", "openai/gpt-3.5-turbo")
        self.system_prompt = os.getenv("SYSTEM_PROMPT", "Ты дружелюбный банковский ассистент.")

    def _get_required(self, key: str) -> str:
        value = os.getenv(key)
        if not value:
            raise ValueError(f"{key} is required in .env")
        return value

config = Config()

# Backwards-compatible aliases (some modules use uppercase names)
config.TELEGRAM_TOKEN = config.telegram_token
config.OPENAI_API_KEY = config.openai_api_key
config.OPENAI_BASE_URL = config.openai_base_url
config.MODEL = config.model
config.SYSTEM_PROMPT = config.system_prompt

