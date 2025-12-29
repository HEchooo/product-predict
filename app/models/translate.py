"""
Pydantic models for translate API request/response.
"""

from typing import Optional, Literal
from pydantic import BaseModel, Field


class TranslateOptions(BaseModel):
    """Options for image translation processing."""
    
    ocr_backend: Literal["rapid", "paddle", "auto"] = Field(
        default="rapid",
        description="OCR backend to use"
    )
    inpaint_backend: Literal["lama", "opencv", "none"] = Field(
        default="lama", 
        description="Inpainting backend for text removal"
    )
    gate_mode: Literal["auto", "off", "strict"] = Field(
        default="auto",
        description="Chart detection gating mode"
    )
    min_font_px: int = Field(default=10, ge=6, le=100, description="Minimum font size in pixels")
    max_font_px: int = Field(default=64, ge=10, le=200, description="Maximum font size in pixels")
    allow_extra_line: bool = Field(default=True, description="Allow extra lines for translation")
    max_extra_lines: int = Field(default=1, ge=0, le=3, description="Maximum extra lines allowed")
    return_base64: bool = Field(default=True, description="Return image as base64 string")


class TranslateUrlRequest(BaseModel):
    """Request model for URL-based translation."""
    
    image_url: str = Field(..., description="URL of the image to translate")
    options: TranslateOptions = Field(default_factory=TranslateOptions)


class TranslateResultData(BaseModel):
    """Data returned from successful translation."""
    
    status: Literal["processed", "skipped", "error"] = Field(
        description="Processing status"
    )
    reason: str = Field(description="Status reason/description")
    chinese_tokens: int = Field(default=0, description="Number of Chinese text tokens detected")
    total_tokens: int = Field(default=0, description="Total OCR tokens detected")
    regions: int = Field(default=0, description="Number of text regions processed")
    time_ms: int = Field(default=0, description="Processing time in milliseconds")
    ocr_backend: str = Field(default="", description="OCR backend used")
    inpaint_backend: str = Field(default="", description="Inpainting backend used")
    output_image_base64: Optional[str] = Field(
        default=None, 
        description="Translated image as base64 string (if return_base64=true)"
    )
    output_filename: Optional[str] = Field(
        default=None,
        description="Output filename"
    )


class TranslateResponse(BaseModel):
    """Standard API response for translation endpoints."""
    
    success: bool = Field(description="Whether the request was successful")
    data: Optional[TranslateResultData] = Field(
        default=None,
        description="Translation result data"
    )
    error: Optional[str] = Field(
        default=None,
        description="Error message if success=false"
    )


class HealthResponse(BaseModel):
    """Health check response."""
    
    status: str = Field(default="ok")
    version: str = Field(default="0.1.0")
