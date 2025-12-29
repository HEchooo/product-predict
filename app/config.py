"""
Configuration management using pydantic-settings.
Loads from environment variables and .env file.
"""

from pydantic import field_validator
from pydantic_settings import BaseSettings
from typing import Optional, Any
import logging

logger = logging.getLogger(__name__)


class TranslateConfig(BaseSettings):
    """
    Translation service settings.
    
    These settings can be overridden with environment variables.
    """
    # API settings
    API_V1_STR: str = "/api/v1"
    PROJECT_NAME: str = "Product Translate API"
    
    # Logging
    LOG_LEVEL: str = "INFO"
    
    # CORS settings (comma-separated string)
    BACKEND_CORS_ORIGINS: str = "*"
    
    # GCP Configuration
    GCP_PROJECT: str = "ai-agent-461123"
    GCP_LOCATION: str = "us-east4"
    GEMINI_MODEL: str = "gemini-2.0-flash-001"
    GEMINI_API_KEY: Optional[str] = None  # If set, use API Key instead of VertexAI
    
    # OCR Configuration
    DEFAULT_OCR_BACKEND: str = "rapid"
    PADDLE_LANG: str = "ch"
    PADDLE_DEVICE: Optional[str] = None
    OCR_AUGMENT: int = 2
    
    # Inpaint Configuration (use opencv on Mac due to LaMa CUDA model issue)
    DEFAULT_INPAINT_BACKEND: str = "opencv"
    LAMA_DEVICE: str = "auto"
    
    # Font Configuration
    MIN_FONT_PX: int = 10
    MAX_FONT_PX: int = 64
    MIN_GOOD_FONT_PX: int = 14
    
    # Processing options
    ALLOW_EXTRA_LINE: bool = True
    MAX_EXTRA_LINES: int = 1
    GATE_MODE: str = "auto"
    
    # Erase refinement
    ERASE_REFINE_ITERS: int = 2
    ERASE_REFINE_RADIUS: int = 7
    ERASE_REFINE_METHOD: str = "ns"
    ERASE_RESIDUAL_DELTA_E: float = 12.0
    ERASE_RESIDUAL_GRAD: float = 40.0
    ERASE_RESIDUAL_MAX_AREA: int = 2000
    
    # Safety limits
    MAX_MASK_RATIO: float = 0.75
    MAX_SINGLE_POLY_AREA_RATIO: float = 0.70
    
    # Paths
    SCHEMA_PATH: str = "chart_translate_schema.json"
    OUTPUT_DIR: str = "output"
    DEBUG_DIR: Optional[str] = None
    
    @field_validator("GCP_PROJECT", mode="before")
    @classmethod
    def validate_gcp_project(cls, v: Optional[str]) -> Optional[str]:
        """
        Validate GCP project.
        """
        if not v:
            logger.warning("GCP_PROJECT is not set. Translation features will not work.")
        return v or ""
    
    class Config:
        case_sensitive = True
        env_file = ".env"
        env_file_encoding = "utf-8"
        extra = "ignore"  # Allow extra fields in .env


class Settings(TranslateConfig):
    """
    Combined application settings.
    """
    pass


# Create settings instance
settings = Settings()


def get_settings() -> Settings:
    """Get settings instance."""
    return settings
