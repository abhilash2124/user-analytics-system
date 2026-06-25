import uuid
from sqlalchemy import Column, String, DateTime, JSON, ForeignKey, func
from sqlalchemy.dialects.postgresql import UUID
from pgvector.sqlalchemy import Vector
from app.database import Base


class User(Base):
    __tablename__ = "users"

    id = Column(String, primary_key=True, index=True)
    created_at = Column(DateTime, server_default=func.now())


class Event(Base):
    __tablename__ = "events"

    id = Column(UUID(as_uuid=True), primary_key=True, default=uuid.uuid4)
    user_id = Column(String, ForeignKey("users.id", ondelete="CASCADE"), nullable=False)
    event_type = Column(String, nullable=False, index=True)
    page = Column(String, nullable=False)
    metadata_json = Column(JSON, default={}, name="metadata")
    timestamp = Column(DateTime, server_default=func.now())
    embedding = Column(Vector(768), nullable=True)
