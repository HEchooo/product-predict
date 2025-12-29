"""
Translation service that wraps preserve_style_translate.py functionality.
Provides singleton instances for OCR, translator, and inpainter.
"""

import os
import base64
import tempfile
import uuid
from pathlib import Path
from typing import Optional, Tuple, Any, Dict

import cv2
import numpy as np

from app.config import get_settings
from app.models.translate import TranslateOptions, TranslateResultData

# Import from preserve_style_translate
import preserve_style_translate as pst


class TranslateService:
    """
    Service for translating Chinese text in images to English.
    Manages singleton instances of OCR, translator, and inpainter.
    """
    
    _instance: Optional["TranslateService"] = None
    
    def __new__(cls) -> "TranslateService":
        if cls._instance is None:
            cls._instance = super().__new__(cls)
            cls._instance._initialized = False
        return cls._instance
    
    def __init__(self) -> None:
        if self._initialized:
            return
        
        self._settings = get_settings()
        self._schema: Dict[str, Any] = {}
        
        # Lazy-loaded components
        self._ocr_rapid: Optional[pst.BaseOCR] = None
        self._ocr_paddle: Optional[pst.BaseOCR] = None
        self._translator: Optional[pst.GeminiTranslator] = None
        self._inpainter_lama: Optional[pst.BaseInpaint] = None
        self._inpainter_opencv: Optional[pst.BaseInpaint] = None
        self._lama_device_used: str = "auto"
        
        # Load schema
        self._load_schema()
        
        self._initialized = True
    
    def _load_schema(self) -> None:
        """Load translation schema from JSON file."""
        schema_path = self._settings.SCHEMA_PATH
        if os.path.exists(schema_path):
            self._schema = pst.load_schema(schema_path)
        else:
            # Default schema
            self._schema = {
                "translation": {
                    "system_prompt": "You are a professional translator for product size charts."
                },
                "gate": {
                    "keywords": ["尺码", "尺寸"],
                    "min_chinese_tokens": 3
                }
            }
    
    def _get_ocr(self, backend: str) -> pst.BaseOCR:
        """Get or create OCR backend."""
        if backend == "rapid":
            if self._ocr_rapid is None:
                self._ocr_rapid = pst.RapidOCRBackend()
            return self._ocr_rapid
        elif backend == "paddle":
            if self._ocr_paddle is None:
                self._ocr_paddle = pst.PaddleOCRBackend(
                    lang=self._settings.PADDLE_LANG,
                    device=self._settings.PADDLE_DEVICE,
                    suppress_init_logs=True,
                    suppress_run_logs=True
                )
            return self._ocr_paddle
        else:  # auto - default to rapid
            if self._ocr_rapid is None:
                self._ocr_rapid = pst.RapidOCRBackend()
            return self._ocr_rapid
    
    def _get_translator(self) -> Optional[pst.GeminiTranslator]:
        """Get or create Gemini translator."""
        if self._translator is None:
            # API Key takes priority over VertexAI
            api_key = self._settings.GEMINI_API_KEY
            if not api_key and not self._settings.GCP_PROJECT:
                return None
            self._translator = pst.GeminiTranslator(
                project=self._settings.GCP_PROJECT,
                location=self._settings.GCP_LOCATION,
                model=self._settings.GEMINI_MODEL,
                schema=self._schema,
                temperature=0.0,
                api_key=api_key
            )
        return self._translator
    
    def _get_inpainter(self, backend: str) -> Tuple[Optional[pst.BaseInpaint], str]:
        """Get or create inpainter backend."""
        if backend == "none":
            return None, "none"
        
        if backend == "lama":
            if self._inpainter_lama is None:
                self._inpainter_lama, self._lama_device_used = pst.build_inpainter(
                    "lama", self._settings.LAMA_DEVICE
                )
            return self._inpainter_lama, self._lama_device_used
        
        if backend == "opencv":
            if self._inpainter_opencv is None:
                self._inpainter_opencv = pst.OpenCVInpaint(radius=3)
            return self._inpainter_opencv, "cpu"
        
        # Default to lama
        if self._inpainter_lama is None:
            self._inpainter_lama, self._lama_device_used = pst.build_inpainter(
                "lama", self._settings.LAMA_DEVICE
            )
        return self._inpainter_lama, self._lama_device_used
    
    def translate_image(
        self,
        image_bytes: bytes,
        options: TranslateOptions,
        filename: Optional[str] = None
    ) -> TranslateResultData:
        """
        Translate Chinese text in an image to English.
        
        Args:
            image_bytes: Raw image bytes
            options: Translation options
            filename: Optional original filename
        
        Returns:
            TranslateResultData with processing results
        """
        settings = self._settings
        
        # Determine file extension
        ext = ".jpg"
        if filename:
            ext = os.path.splitext(filename)[1] or ".jpg"
        
        # Create temp file for image
        with tempfile.NamedTemporaryFile(suffix=ext, delete=False) as tmp:
            tmp.write(image_bytes)
            tmp_path = tmp.name
        
        try:
            # Get components
            ocr = self._get_ocr(options.ocr_backend)
            translator = self._get_translator()
            inpainter, lama_device = self._get_inpainter(options.inpaint_backend)
            
            # Ensure output directory exists
            output_dir = settings.OUTPUT_DIR
            pst.ensure_dir(output_dir)
            
            # Process image
            result = pst.process_one(
                path=tmp_path,
                ocr=ocr,
                schema=self._schema,
                translator=translator,
                inpainter=inpainter,
                outdir=output_dir,
                min_font_px=options.min_font_px,
                max_font_px=options.max_font_px,
                gate_mode=options.gate_mode,
                ocr_augment=settings.OCR_AUGMENT,
                debug_dir=settings.DEBUG_DIR,
                lama_device_used=lama_device,
                allow_extra_line=options.allow_extra_line,
                max_extra_lines=options.max_extra_lines,
                min_good_font_px=settings.MIN_GOOD_FONT_PX,
                refine_radius=settings.ERASE_REFINE_RADIUS,
                refine_method=settings.ERASE_REFINE_METHOD,
                residual_deltaE_thr=settings.ERASE_RESIDUAL_DELTA_E,
                residual_grad_thr=settings.ERASE_RESIDUAL_GRAD,
                residual_max_cc_area=settings.ERASE_RESIDUAL_MAX_AREA,
                refine_iters=settings.ERASE_REFINE_ITERS,
                max_mask_ratio=settings.MAX_MASK_RATIO,
                max_single_poly_area_ratio=settings.MAX_SINGLE_POLY_AREA_RATIO,
            )
            
            # Build response
            response_data = TranslateResultData(
                status=result.status,
                reason=result.reason,
                chinese_tokens=result.chinese_tokens,
                total_tokens=result.total_tokens,
                regions=result.regions,
                time_ms=result.time_ms,
                ocr_backend=result.ocr_backend,
                inpaint_backend=result.inpaint_backend,
            )
            
            # Include output image if requested and available
            if options.return_base64 and result.out_path and os.path.exists(result.out_path):
                with open(result.out_path, "rb") as f:
                    response_data.output_image_base64 = base64.b64encode(f.read()).decode("utf-8")
                response_data.output_filename = os.path.basename(result.out_path)
            
            return response_data
            
        finally:
            # Cleanup temp file
            if os.path.exists(tmp_path):
                os.unlink(tmp_path)
    
    async def translate_image_from_url(
        self,
        image_url: str,
        options: TranslateOptions
    ) -> TranslateResultData:
        """
        Download image from URL and translate.
        
        Args:
            image_url: URL of the image
            options: Translation options
        
        Returns:
            TranslateResultData with processing results
        """
        import httpx
        
        async with httpx.AsyncClient(timeout=30.0) as client:
            response = await client.get(image_url)
            response.raise_for_status()
            image_bytes = response.content
        
        # Extract filename from URL
        filename = image_url.split("/")[-1].split("?")[0]
        if not filename or "." not in filename:
            filename = f"image_{uuid.uuid4().hex[:8]}.jpg"
        
        # Process synchronously (heavy computation)
        return self.translate_image(image_bytes, options, filename)


# Singleton accessor
def get_translate_service() -> TranslateService:
    """Get singleton TranslateService instance."""
    return TranslateService()
