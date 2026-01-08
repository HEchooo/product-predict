"""
FastAPI application entry point for image translation service.
"""

# 在最开始加载 .env 文件，确保所有服务能读取环境变量
from dotenv import load_dotenv
load_dotenv()

import logging
import uvicorn
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from app.api.translate import router as translate_router
from app.config import get_settings

# 配置日志级别
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
)


def create_app() -> FastAPI:
    """Create and configure FastAPI application."""
    
    settings = get_settings()
    
    app = FastAPI(
        title="Product Translate API",
        description=(
            "API service for translating Chinese text in product images "
            "(especially size charts) to English while preserving the original style."
        ),
        version="0.1.0",
        docs_url="/docs",
        redoc_url="/redoc",
    )
    
    # CORS middleware
    cors_origins = [o.strip() for o in settings.BACKEND_CORS_ORIGINS.split(",") if o.strip()]
    app.add_middleware(
        CORSMiddleware,
        allow_origins=cors_origins if cors_origins else ["*"],
        allow_credentials=True,
        allow_methods=["*"],
        allow_headers=["*"],
    )
    
    # Include routers
    app.include_router(translate_router)
    
    @app.on_event("startup")
    async def startup_event():
        """Initialize services on startup."""
        import logging
        logger = logging.getLogger(__name__)
        
        # Pre-warm services: initialize OCR and inpainter
        try:
            logger.info("Pre-loading translation services...")
            from app.services.translate_service import get_translate_service
            service = get_translate_service()
            
            # Trigger inpainter initialization
            service._get_inpainter(settings.DEFAULT_INPAINT_BACKEND)
            logger.info("Translation services ready.")
        except Exception as e:
            logger.warning(f"Pre-loading failed (will retry on first request): {e}")
    
    return app


# Application instance
app = create_app()


def run():
    """Run the application with uvicorn."""
    uvicorn.run(
        "main:app",
        host="0.0.0.0",
        port=8000,
        reload=True,
    )


if __name__ == "__main__":
    run()
