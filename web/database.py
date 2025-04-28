from sqlalchemy import create_engine, Column, Integer, String, Float, DateTime, Boolean
from sqlalchemy.ext.declarative import declarative_base
from sqlalchemy.orm import sessionmaker
import os
from datetime import datetime

# Create the database directory if it doesn't exist
db_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'data')
os.makedirs(db_dir, exist_ok=True)

# Database configuration
DATABASE_URL = f"sqlite:///{os.path.join(db_dir, 'security.db')}"
engine = create_engine(DATABASE_URL)
SessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=engine)
Base = declarative_base()

# Models
class KnownFace(Base):
    __tablename__ = "known_faces"
    
    id = Column(Integer, primary_key=True, index=True)
    name = Column(String, index=True)
    date = Column(String)
    time = Column(String)
    confidence = Column(Float)
    snapshot = Column(String)
    status = Column(String)

class UnknownFace(Base):
    __tablename__ = "unknown_faces"
    
    id = Column(Integer, primary_key=True, index=True)
    date = Column(String)
    time = Column(String)
    confidence = Column(Float)
    snapshot = Column(String)
    status = Column(String)

class MotionEvent(Base):
    __tablename__ = "motion_events"
    
    id = Column(Integer, primary_key=True, index=True)
    date = Column(String)
    time = Column(String)
    confidence = Column(Float)
    snapshot = Column(String)
    status = Column(String)

class EmailLog(Base):
    __tablename__ = "email_logs"
    
    id = Column(Integer, primary_key=True, index=True)
    date = Column(String)
    time = Column(String)
    recipient = Column(String)
    subject = Column(String)
    status = Column(String)

# Database operations
def get_db():
    db = SessionLocal()
    try:
        yield db
    finally:
        db.close()

def init_db():
    Base.metadata.create_all(bind=engine)

def get_statistics(db):
    stats = {
        'known_faces': db.query(KnownFace).count(),
        'unknown_faces': db.query(UnknownFace).count(),
        'motion_events': db.query(MotionEvent).count(),
        'email_notifications': db.query(EmailLog).count()
    }
    return stats

def get_recent_events(db, limit=10):
    events = []
    
    # Get known faces
    known_faces = db.query(KnownFace).order_by(KnownFace.id.desc()).limit(limit).all()
    for face in known_faces:
        events.append({
            'type': 'known_face',
            'name': face.name,
            'date': face.date,
            'time': face.time,
            'confidence': f"{face.confidence:.2f}",
            'snapshot': face.snapshot,
            'status': face.status
        })
    
    # Get unknown faces
    unknown_faces = db.query(UnknownFace).order_by(UnknownFace.id.desc()).limit(limit).all()
    for face in unknown_faces:
        events.append({
            'type': 'unknown_face',
            'name': 'Unknown Person',
            'date': face.date,
            'time': face.time,
            'confidence': f"{face.confidence:.2f}",
            'snapshot': face.snapshot,
            'status': face.status
        })
    
    # Get motion events
    motion_events = db.query(MotionEvent).order_by(MotionEvent.id.desc()).limit(limit).all()
    for event in motion_events:
        events.append({
            'type': 'motion',
            'name': 'Motion Event',
            'date': event.date,
            'time': event.time,
            'confidence': f"{event.confidence:.2f}",
            'snapshot': event.snapshot,
            'status': event.status
        })
    
    # Sort all events by date and time
    events.sort(key=lambda x: (x['date'], x['time']), reverse=True)
    return events[:limit]

def get_known_faces(db):
    return db.query(KnownFace).all()

def get_snapshots(db, limit=20):
    snapshots = []
    
    # Get snapshots from known faces
    known_faces = db.query(KnownFace).order_by(KnownFace.id.desc()).limit(limit).all()
    for face in known_faces:
        snapshots.append({
            'type': 'known_face',
            'name': face.name,
            'date': face.date,
            'time': face.time,
            'path': face.snapshot
        })
    
    # Get snapshots from unknown faces
    unknown_faces = db.query(UnknownFace).order_by(UnknownFace.id.desc()).limit(limit).all()
    for face in unknown_faces:
        snapshots.append({
            'type': 'unknown_face',
            'name': 'Unknown Person',
            'date': face.date,
            'time': face.time,
            'path': face.snapshot
        })
    
    # Get snapshots from motion events
    motion_events = db.query(MotionEvent).order_by(MotionEvent.id.desc()).limit(limit).all()
    for event in motion_events:
        snapshots.append({
            'type': 'motion',
            'name': 'Motion Event',
            'date': event.date,
            'time': event.time,
            'path': event.snapshot
        })
    
    # Sort all snapshots by date and time
    snapshots.sort(key=lambda x: (x['date'], x['time']), reverse=True)
    return snapshots[:limit]

def get_email_logs(db, limit=50):
    return db.query(EmailLog).order_by(EmailLog.id.desc()).limit(limit).all()

def clear_events(db, event_type):
    if event_type == 'known_faces':
        db.query(KnownFace).delete()
    elif event_type == 'unknown_faces':
        db.query(UnknownFace).delete()
    elif event_type == 'motion':
        db.query(MotionEvent).delete()
    elif event_type == 'emails':
        db.query(EmailLog).delete()
    db.commit() 