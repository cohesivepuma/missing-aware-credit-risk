"""Private local artifact storage and transactional SQLite application metadata."""
from __future__ import annotations

from contextlib import contextmanager
from datetime import datetime, timezone
import json
import os
from pathlib import Path
import re
import sqlite3
from threading import RLock
from typing import Any
import uuid


def now() -> str:
    return datetime.now(timezone.utc).isoformat()


def new_id() -> str:
    return uuid.uuid4().hex


class Store:
    def __init__(self, root: Path) -> None:
        self.root = root.resolve()
        self.root.mkdir(parents=True, exist_ok=True, mode=0o700)
        os.chmod(self.root, 0o700)
        self.lock = RLock()
        self.database = self.root / 'workbench.sqlite3'
        with self.connection() as connection:
            connection.executescript('''
                PRAGMA journal_mode=WAL;
                CREATE TABLE IF NOT EXISTS objects (
                  kind TEXT NOT NULL, id TEXT NOT NULL, payload TEXT NOT NULL,
                  created_at TEXT NOT NULL, PRIMARY KEY(kind,id));
                CREATE TABLE IF NOT EXISTS settings (key TEXT PRIMARY KEY,value TEXT NOT NULL);
                CREATE TABLE IF NOT EXISTS sessions (token_hash TEXT PRIMARY KEY,expires_at REAL NOT NULL);
                CREATE TABLE IF NOT EXISTS audit (
                  id INTEGER PRIMARY KEY AUTOINCREMENT,action TEXT NOT NULL,
                  detail TEXT NOT NULL,created_at TEXT NOT NULL);
            ''')
        os.chmod(self.database, 0o600)

    @contextmanager
    def connection(self):
        with self.lock:
            connection = sqlite3.connect(self.database, timeout=20)
            connection.row_factory = sqlite3.Row
            try:
                with connection:
                    yield connection
            finally:
                connection.close()

    def get(self, kind: str, identifier: str) -> dict | None:
        with self.connection() as connection:
            row = connection.execute('SELECT payload FROM objects WHERE kind=? AND id=?', (kind, identifier)).fetchone()
        return json.loads(row['payload']) if row else None

    def put(self, kind: str, payload: dict) -> dict:
        with self.connection() as connection:
            connection.execute('INSERT INTO objects(kind,id,payload,created_at) VALUES(?,?,?,?) ON CONFLICT(kind,id) DO UPDATE SET payload=excluded.payload',
                               (kind, payload['id'], json.dumps(payload, ensure_ascii=False, allow_nan=False), payload['created_at']))
        return payload

    def update(self, kind: str, identifier: str, **values: Any) -> dict:
        with self.lock:
            current = self.get(kind, identifier)
            if current is None:
                raise KeyError(identifier)
            current.update(values)
            return self.put(kind, current)

    def list(self, kind: str) -> list[dict]:
        with self.connection() as connection:
            rows = connection.execute('SELECT payload FROM objects WHERE kind=? ORDER BY created_at DESC', (kind,)).fetchall()
        return [json.loads(row['payload']) for row in rows]

    def setting(self, key: str) -> str | None:
        with self.connection() as connection:
            row = connection.execute('SELECT value FROM settings WHERE key=?', (key,)).fetchone()
        return row['value'] if row else None

    def set_initial_password(self, encoded: str) -> bool:
        with self.connection() as connection:
            cursor = connection.execute('INSERT OR IGNORE INTO settings(key,value) VALUES(?,?)', ('password', encoded))
            return cursor.rowcount == 1

    def add_activity(self, action: str, detail: str) -> None:
        with self.connection() as connection:
            connection.execute('INSERT INTO audit(action,detail,created_at) VALUES(?,?,?)', (action, detail[:500], now()))

    def activities(self, limit: int = 12) -> list[dict]:
        with self.connection() as connection:
            return [dict(row) for row in connection.execute('SELECT * FROM audit ORDER BY id DESC LIMIT ?', (limit,))]

    def path(self, kind: str, identifier: str, extension: str) -> Path:
        if kind not in {'datasets', 'runs', 'predictions'} or not re.fullmatch(r'[a-f0-9]{32}', identifier):
            raise ValueError('Invalid artifact identifier.')
        if extension not in {'.csv', '.joblib'}:
            raise ValueError('Invalid artifact type.')
        directory = self.root / kind
        directory.mkdir(exist_ok=True, mode=0o700)
        return directory / (identifier + extension)
