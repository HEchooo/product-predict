"""
Translation API endpoints.
"""

from typing import Annotated

from fastapi import APIRouter, File, Form, UploadFile, HTTPException, Depends

from app.models.translate import (
    TranslateOptions,
    TranslateUrlRequest,
    TranslateResponse,
    TranslateResultData,
    HealthResponse,
)
from app.services.translate_service import TranslateService, get_translate_service


router = APIRouter()


@router.get("/health", response_model=HealthResponse, tags=["Health"])
async def health_check() -> HealthResponse:
    """
    Health check endpoint.
    Returns service status and version.
    """
    return HealthResponse(status="ok", version="0.1.0")


@router.post(
    "/api/v1/translate",
    response_model=TranslateResponse,
    tags=["Translation"],
    summary="Translate image (multipart upload)",
    description="Upload an image containing Chinese text and get the translated version with English text."
)
async def translate_image(
    file: Annotated[UploadFile, File(description="Image file to translate")],
    ocr_backend: Annotated[str, Form()] = "rapid",
    inpaint_backend: Annotated[str, Form()] = "lama",
    gate_mode: Annotated[str, Form()] = "auto",
    min_font_px: Annotated[int, Form()] = 10,
    max_font_px: Annotated[int, Form()] = 64,
    allow_extra_line: Annotated[bool, Form()] = True,
    max_extra_lines: Annotated[int, Form()] = 1,
    return_base64: Annotated[bool, Form()] = True,
    service: TranslateService = Depends(get_translate_service),
) -> TranslateResponse:
    """
    Translate Chinese text in an uploaded image to English.
    
    The image is processed through:
    1. OCR to detect Chinese text
    2. Inpainting to remove original text
    3. Translation via Gemini
    4. Rendering English text in the same style/position
    
    Returns the processed image as base64 (if return_base64=true).
    """
    # Validate file type
    if file.content_type and not file.content_type.startswith("image/"):
        raise HTTPException(
            status_code=400,
            detail=f"Invalid file type: {file.content_type}. Expected image/*"
        )
    
    # Build options
    options = TranslateOptions(
        ocr_backend=ocr_backend,  # type: ignore
        inpaint_backend=inpaint_backend,  # type: ignore
        gate_mode=gate_mode,  # type: ignore
        min_font_px=min_font_px,
        max_font_px=max_font_px,
        allow_extra_line=allow_extra_line,
        max_extra_lines=max_extra_lines,
        return_base64=return_base64,
    )
    
    try:
        # Read image bytes
        image_bytes = await file.read()
        
        # Process translation
        result = service.translate_image(
            image_bytes=image_bytes,
            options=options,
            filename=file.filename
        )
        
        return TranslateResponse(success=True, data=result)
        
    except Exception as e:
        return TranslateResponse(
            success=False,
            error=str(e),
            data=TranslateResultData(
                status="error",
                reason=str(e)
            )
        )


@router.post(
    "/api/v1/translate/url",
    response_model=TranslateResponse,
    tags=["Translation"],
    summary="Translate image from URL",
    description="Provide an image URL containing Chinese text and get the translated version."
)
async def translate_image_from_url(
    request: TranslateUrlRequest,
    service: TranslateService = Depends(get_translate_service),
) -> TranslateResponse:
    """
    Translate Chinese text in an image from URL to English.
    
    Downloads the image from the provided URL and processes it.
    """
    try:
        result = await service.translate_image_from_url(
            image_url=request.image_url,
            options=request.options
        )
        
        return TranslateResponse(success=True, data=result)
        
    except Exception as e:
        return TranslateResponse(
            success=False,
            error=str(e),
            data=TranslateResultData(
                status="error",
                reason=str(e)
            )
        )
