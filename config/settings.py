import os
from pathlib import Path
from dotenv import load_dotenv

# Base project paths
BASE_DIR = Path(__file__).resolve().parent.parent
ENV_PATH = BASE_DIR / ".env"

if ENV_PATH.exists():
    load_dotenv(ENV_PATH)
else:
    load_dotenv()


class Settings:
    # API Keys
    GROQ_API_KEY: str | None = os.getenv("GROQ_API_KEY")
    OPENAI_API_KEY: str | None = os.getenv("OPENAI_API_KEY")
    ANTHROPIC_API_KEY: str | None = os.getenv("ANTHROPIC_API_KEY")
    GOOGLE_API_KEY: str | None = os.getenv("GOOGLE_API_KEY")

    # Defaults
    DEFAULT_PROVIDER: str = os.getenv("DEFAULT_PROVIDER", "groq")
    DEFAULT_MODEL: str = os.getenv("DEFAULT_MODEL", "llama-3.3-70b-versatile")

    # Persona
    ASSISTANT_NAME: str = os.getenv("ASSISTANT_NAME", "Jarvis")
    USER_TITLE: str = os.getenv("USER_TITLE", "Sir")

    # Workspace & Storage
    WORKSPACE_DIR: Path = BASE_DIR
    DATA_DIR: Path = BASE_DIR / "data"
    MEMORY_FILE_PATH: Path = Path(os.getenv("JARVIS_MEMORY_PATH", str(BASE_DIR / "data" / "jarvis_memory.json")))


settings = Settings()
