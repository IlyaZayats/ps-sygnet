from sqlalchemy import (
    Column,
    Integer,
    String,
    Text,
    Boolean,
    ForeignKey,
    Numeric,
    DateTime,
    func
)
from sqlalchemy.orm import relationship, declarative_base

Base = declarative_base()

class User(Base):
    __tablename__ = "user"

    user_id = Column(Integer, primary_key=True, autoincrement=True)
    name = Column(String(60), nullable=False)
    login = Column(String(30), nullable=False, unique=True)
    pwd = Column(String(60), nullable=False)

    create_dt = Column(DateTime, server_default=func.now(), nullable=False)
    update_dt = Column(DateTime, server_default=func.now(), onupdate=func.now(), nullable=False)

    researches = relationship("Research", back_populates="user", cascade="all, delete-orphan")


class Research(Base):
    __tablename__ = "research"

    research_id = Column(Integer, primary_key=True, autoincrement=True)
    user_id = Column(Integer, ForeignKey("user.user_id", ondelete="CASCADE"), nullable=False)

    classification_bin_new = Column(Numeric(1, 3), nullable=True)
    classification_bin_old = Column(Numeric(1, 3), nullable=True)
    file_path = Column(Text, nullable=True)

    create_dt = Column(DateTime, server_default=func.now(), nullable=False)
    update_dt = Column(DateTime, server_default=func.now(), onupdate=func.now(), nullable=False)

    user = relationship("User", back_populates="researches")
    logs = relationship("ResearchLog", back_populates="research", cascade="all, delete-orphan")


class ResearchLog(Base):
    __tablename__ = "research_log"

    log_id = Column(Integer, primary_key=True, autoincrement=True)
    research_id = Column(Integer, ForeignKey("research.research_id", ondelete="CASCADE"), nullable=False)

    is_error = Column(Boolean, nullable=False, default=False)
    msg = Column(Text, nullable=True)

    create_dt = Column(DateTime, server_default=func.now(), nullable=False)
    update_dt = Column(DateTime, server_default=func.now(), onupdate=func.now(), nullable=False)

    research = relationship("Research", back_populates="logs")