from __future__ import annotations

from datetime import datetime
from typing import Optional
from sqlalchemy import String, Integer, Date, DateTime, ForeignKey, Text, UniqueConstraint, Float
from sqlalchemy.orm import Mapped, mapped_column, relationship
from .db import Base


class Source(Base):
    __tablename__ = "sources"

    id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)
    name: Mapped[str] = mapped_column(String(100), unique=True, index=True)
    base_url: Mapped[Optional[str]] = mapped_column(String(500), nullable=True)

    documents: Mapped[list[Document]] = relationship("Document", back_populates="source")


class Document(Base):
    __tablename__ = "documents"
    __table_args__ = (
        UniqueConstraint("source_id", "external_id", name="uq_source_external"),
    )

    id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)
    source_id: Mapped[int] = mapped_column(ForeignKey("sources.id"), index=True)

    external_id: Mapped[Optional[str]] = mapped_column(String(200), nullable=True, index=True)
    title: Mapped[Optional[str]] = mapped_column(String(1000), nullable=True)
    date: Mapped[Optional[Date]] = mapped_column(Date, nullable=True, index=True)

    url: Mapped[Optional[str]] = mapped_column(String(1000), nullable=True)
    sha256: Mapped[Optional[str]] = mapped_column(String(64), nullable=True, index=True)
    file_path: Mapped[Optional[str]] = mapped_column(String(1000), nullable=True)

    pages: Mapped[Optional[int]] = mapped_column(Integer, nullable=True)

    created_at: Mapped[datetime] = mapped_column(DateTime, default=datetime.utcnow)

    source: Mapped[Source] = relationship("Source", back_populates="documents")
    page_texts: Mapped[list[Page]] = relationship("Page", back_populates="document", cascade="all, delete-orphan")


class Page(Base):
    __tablename__ = "pages"
    __table_args__ = (
        UniqueConstraint("document_id", "page_no", name="uq_doc_page"),
    )

    id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)
    document_id: Mapped[int] = mapped_column(ForeignKey("documents.id"), index=True)

    page_no: Mapped[int] = mapped_column(Integer)
    text: Mapped[Optional[str]] = mapped_column(Text, nullable=True)
    ocr_confidence: Mapped[Optional[float]] = mapped_column(Float, nullable=True)
    image_path: Mapped[Optional[str]] = mapped_column(String(1000), nullable=True)

    document: Mapped[Document] = relationship("Document", back_populates="page_texts")


class IngestionJob(Base):
    __tablename__ = "ingestion_jobs"

    id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)
    source: Mapped[str] = mapped_column(String(100), index=True)
    status: Mapped[str] = mapped_column(String(50), default="running")
    started_at: Mapped[datetime] = mapped_column(DateTime, default=datetime.utcnow)
    finished_at: Mapped[Optional[datetime]] = mapped_column(DateTime, nullable=True)
    error: Mapped[Optional[str]] = mapped_column(Text, nullable=True)
    stats: Mapped[Optional[str]] = mapped_column(Text, nullable=True)  # JSON string for simplicity
