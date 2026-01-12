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
        
        # Initialize database
        try:
            logger.info("Initializing database...")
            from app.database import init_db
            init_db()
            logger.info("Database initialized successfully")
        except Exception as e:
            logger.error(f"Failed to initialize database: {e}")
            raise
        
        # Start background worker
        try:
            logger.info("Starting background worker...")
            from app.worker import start_worker
            await start_worker()
            logger.info("Background worker started")
        except Exception as e:
            logger.error(f"Failed to start worker: {e}")
            raise
        
        # Pre-warm translation services (optional, for sync translate)
        try:
            logger.info("Pre-loading translation services...")
            from app.services.translate_service import get_translate_service
            service = get_translate_service()
            service._get_inpainter(settings.DEFAULT_INPAINT_BACKEND)
            logger.info("Translation services ready.")
        except Exception as e:
            logger.warning(f"Pre-loading failed (will retry on first request): {e}")
    
    @app.on_event("shutdown")
    async def shutdown_event():
        """Cleanup on shutdown."""
        import logging
        logger = logging.getLogger(__name__)
        
        # Stop background worker
        try:
            logger.info("Stopping background worker...")
            from app.worker import stop_worker
            await stop_worker()
            logger.info("Background worker stopped")
        except Exception as e:
            logger.warning(f"Error stopping worker: {e}")
        
        # Close database connections
        try:
            from app.database import close_db
            await close_db()
            logger.info("Database connections closed")
        except Exception as e:
            logger.warning(f"Error closing database: {e}")
    
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
