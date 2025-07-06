from sqlalchemy.orm import Session
from models import User, Research, ResearchLog

def login(db: Session, login: str, pwd: str) -> (int, bool):
    user = db.query(User).filter(User.login == login).first()
    if user is None:
        return -1, False

    if user.pwd != pwd:
        return -1, False

    return user.user_id, True

def add_research(db: Session, research: Research):
    db.add(research)
    db.commit()
    db.refresh(research)

def add_research_log(db: Session, log: ResearchLog):
    db.add(log)
    db.commit()
    db.refresh(log)