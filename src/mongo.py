"""MongoDB layer: chat + prediction + scenario logs. Always best-effort.

DB down / no server / bad URI  ->  every function reports ok=False and the app
keeps working on session logs. The UI always shows which store is active.
"""
from __future__ import annotations
from datetime import datetime

_CLIENTS = {}


def get_db(uri: str, timeout_ms: int = 3000):
    """Returns (db_or_None, error_or_None). Never raises."""
    try:
        from pymongo import MongoClient
        uri = (uri or "").strip()
        if not uri:
            return None, "empty URI"
        if uri not in _CLIENTS:
            _CLIENTS[uri] = MongoClient(uri, serverSelectionTimeoutMS=timeout_ms,
                                        connectTimeoutMS=timeout_ms)
        client = _CLIENTS[uri]
        db = client.get_database("agripredict")
        db.command("ping")
        return db, None
    except Exception as e:
        _CLIENTS.pop(uri, None)
        return None, f"{type(e).__name__}: {str(e)[:160]}"


def _insert(db, coll: str, rec: dict) -> bool:
    try:
        rec = {**rec, "at": datetime.now().strftime("%Y-%m-%d %H:%M:%S")}
        db[coll].insert_one(rec)
        return True
    except Exception:
        return False


def log_chat(db, role: str, text: str, intent: str = "", mode: str = "") -> bool:
    if db is None:
        return False
    return _insert(db, "chats", {"role": role, "text": str(text)[:2000], "intent": intent, "mode": mode})


def log_prediction(db, rec: dict) -> bool:
    return False if db is None else _insert(db, "predictions", rec)


def log_scenario(db, rec: dict) -> bool:
    return False if db is None else _insert(db, "scenarios", rec)


def recent_chats(db, n: int = 20) -> list:
    try:
        return list(db["chats"].find({}, {"_id": 0}).sort("at", -1).limit(n))[::-1]
    except Exception:
        return []


def counts(db) -> dict:
    """Document counts per collection — empty dict when unreachable."""
    try:
        return {c: int(db[c].count_documents({})) for c in ("chats", "predictions", "scenarios")}
    except Exception:
        return {}
