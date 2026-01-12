"""
Database connection and session management.
Uses SQLAlchemy 2.0 with async support.
"""

import logging
from contextlib import asynccontextmanager
from typing import AsyncGenerator

from sqlalchemy import create_engine, text
from sqlalchemy.ext.asyncio import AsyncSession, async_sessionmaker, create_async_engine
from sqlalchemy.orm import DeclarativeBase, sessionmaker

from app.config import get_settings

logger = logging.getLogger(__name__)


class Base(DeclarativeBase):
    """Base class for all SQLAlchemy models."""
    pass


# Engines and session factories (initialized lazily)
_sync_engine = None
_async_engine = None
_sync_session_factory = None
_async_session_factory = None


def get_database_url(async_mode: bool = False) -> str:
    """Get database URL from settings."""
    settings = get_settings()
    if async_mode:
        return (
            f"mysql+aiomysql://{settings.MYSQL_USER}:{settings.MYSQL_PASSWORD}"
            f"@{settings.MYSQL_HOST}:{settings.MYSQL_PORT}/{settings.MYSQL_DATABASE}"
        )
    return (
        f"mysql+pymysql://{settings.MYSQL_USER}:{settings.MYSQL_PASSWORD}"
        f"@{settings.MYSQL_HOST}:{settings.MYSQL_PORT}/{settings.MYSQL_DATABASE}"
    )


def get_sync_engine():
    """Get or create synchronous database engine."""
    global _sync_engine
    if _sync_engine is None:
        _sync_engine = create_engine(
            get_database_url(async_mode=False),
            pool_pre_ping=True,
            pool_recycle=3600,
            echo=False,
        )
    return _sync_engine


def get_async_engine():
    """Get or create async database engine."""
    global _async_engine
    if _async_engine is None:
        _async_engine = create_async_engine(
            get_database_url(async_mode=True),
            pool_pre_ping=True,
            pool_recycle=3600,
            echo=False,
        )
    return _async_engine


def get_sync_session_factory():
    """Get or create synchronous session factory."""
    global _sync_session_factory
    if _sync_session_factory is None:
        _sync_session_factory = sessionmaker(
            bind=get_sync_engine(),
            expire_on_commit=False,
        )
    return _sync_session_factory


def get_async_session_factory():
    """Get or create async session factory."""
    global _async_session_factory
    if _async_session_factory is None:
        _async_session_factory = async_sessionmaker(
            bind=get_async_engine(),
            expire_on_commit=False,
        )
    return _async_session_factory


@asynccontextmanager
async def get_async_session() -> AsyncGenerator[AsyncSession, None]:
    """Get async database session (context manager)."""
    session = get_async_session_factory()()
    try:
        yield session
        await session.commit()
    except Exception:
        await session.rollback()
        raise
    finally:
        await session.close()


def init_db() -> None:
    """
    Initialize database: create database if not exists, then create all tables.
    This uses synchronous operations for startup.
    """
    settings = get_settings()
    
    # First, connect without database to create it if needed
    base_url = (
        f"mysql+pymysql://{settings.MYSQL_USER}:{settings.MYSQL_PASSWORD}"
        f"@{settings.MYSQL_HOST}:{settings.MYSQL_PORT}"
    )
    
    try:
        temp_engine = create_engine(base_url)
        with temp_engine.connect() as conn:
            # Create database if not exists
            conn.execute(text(f"CREATE DATABASE IF NOT EXISTS {settings.MYSQL_DATABASE}"))
            conn.commit()
        temp_engine.dispose()
        logger.info(f"Database '{settings.MYSQL_DATABASE}' ensured to exist")
    except Exception as e:
        logger.error(f"Failed to create database: {e}")
        raise
    
    # Now create tables
    try:
        # Import models to register them with Base
        from app.models.task import TranslateTask  # noqa: F401
        
        engine = get_sync_engine()
        Base.metadata.create_all(bind=engine)
        logger.info("Database tables created successfully")
    except Exception as e:
        logger.error(f"Failed to create tables: {e}")
        raise


async def close_db() -> None:
    """Close database connections."""
    global _sync_engine, _async_engine
    
    if _async_engine:
        await _async_engine.dispose()
        _async_engine = None
    
    if _sync_engine:
        _sync_engine.dispose()
        _sync_engine = None
    
    logger.info("Database connections closed")
