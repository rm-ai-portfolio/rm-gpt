import json
import uuid
import logging
from pathlib import Path
from typing import Dict, List, Optional
from pydantic import ValidationError
from filelock import FileLock
from models import ChatSession, Message
from config import settings

logger = logging.getLogger(__name__)

class SessionManager:
    def __init__(self):
        self.sessions: Dict[str, ChatSession] = {}
        self.active_session_id: Optional[str] = None
        self._lock = FileLock(str(settings.sessions_file) + ".lock")
        self._load_sessions()
        
        if not self.active_session_id:
            if self.sessions:
                self.active_session_id = next(iter(self.sessions))
            else:
                self.create_new_session()

    def _load_sessions(self) -> None:
        sessions_path = Path(settings.sessions_file)
        if not sessions_path.exists():
            return
        try:
            with open(sessions_path, "r", encoding="utf-8") as f:
                data = json.load(f)
            for sid, sdata in data.items():
                self.sessions[sid] = ChatSession(**sdata)
        except (json.JSONDecodeError, ValidationError, IOError, KeyError) as e:
            logger.error(f"Failed to load sessions: {e}")

    def _save_sessions(self) -> None:
        try:
            with self._lock:
                data = {sid: session.model_dump() for sid, session in self.sessions.items()}
                with open(settings.sessions_file, "w", encoding="utf-8") as f:
                    json.dump(data, f, indent=4)
                    logger.info("Session saved sucessfully")
        except Exception as e:
            logger.exception("Save session failed", e)

    def create_new_session(self) -> str:
        session_id = f"chat_{uuid.uuid4().hex[:8]}"
        count = len(self.sessions) + 1
        self.sessions[session_id] = ChatSession(
            id=session_id,
            title=f"New Chat {count}"
        )
        self.active_session_id = session_id
        self._save_sessions()
        return session_id

    def get_active_session(self) -> Optional[ChatSession]:
        return self.sessions.get(self.active_session_id)

    def set_active_session(self, session_id: str) -> None:
        if session_id in self.sessions:
            self.active_session_id = session_id

    def add_message(self, role: str, content: str) -> None:
        session = self.get_active_session()
        if session:
            session.messages.append(Message(role=role, content=content))
            self._save_sessions()

    def update_session_files(self, file_names: List[str]) -> None:
        try:
            session = self.get_active_session()
            if session:
                session.file_names = file_names
                session.has_vector_store = True
                self._save_sessions()
                logger.info("Session updated successfully")
        except Exception as e:
            logger.exception("Updates session file fAILED", e)
