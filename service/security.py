"""Single-user password sessions; no default credentials or model-upload trust."""
import hashlib
import hmac
import secrets
import time
from collections import defaultdict
from threading import Lock

from fastapi import HTTPException, Request, Response

from service.storage import Store

COOKIE = 'aware_session'
SESSION_SECONDS = 12 * 60 * 60


def password_hash(password: str) -> str:
    salt = secrets.token_bytes(16)
    digest = hashlib.scrypt(password.encode(), salt=salt, n=16384, r=8, p=1)
    return salt.hex() + ':' + digest.hex()


def verify_password(password: str, stored: str) -> bool:
    salt, digest = stored.split(':')
    actual = hashlib.scrypt(password.encode(), salt=bytes.fromhex(salt), n=16384, r=8, p=1)
    return hmac.compare_digest(actual.hex(), digest)


class Sessions:
    def __init__(self, store: Store, *, secure_cookie: bool = False) -> None:
        self.store = store
        self.secure_cookie = secure_cookie
        self.failures: dict[str, list[float]] = defaultdict(list)
        self.lock = Lock()

    def authenticated(self, request: Request) -> bool:
        token = request.cookies.get(COOKIE, '')
        if not token or len(token) > 256:
            return False
        digest = hashlib.sha256(token.encode()).hexdigest()
        with self.store.connection() as connection:
            row = connection.execute('SELECT expires_at FROM sessions WHERE token_hash=?', (digest,)).fetchone()
        return bool(row and row['expires_at'] > time.time())

    def issue(self, response: Response) -> None:
        token = secrets.token_urlsafe(32)
        with self.store.connection() as connection:
            connection.execute('DELETE FROM sessions WHERE expires_at < ?', (time.time(),))
            connection.execute('INSERT INTO sessions VALUES(?,?)', (hashlib.sha256(token.encode()).hexdigest(), time.time() + SESSION_SECONDS))
        response.set_cookie(COOKIE, token, httponly=True, secure=self.secure_cookie, samesite='strict', max_age=SESSION_SECONDS, path='/')

    def logout(self, request: Request, response: Response) -> None:
        token = request.cookies.get(COOKIE, '')
        with self.store.connection() as connection:
            connection.execute('DELETE FROM sessions WHERE token_hash=?', (hashlib.sha256(token.encode()).hexdigest(),))
        response.delete_cookie(COOKIE, path='/', secure=self.secure_cookie, httponly=True, samesite='strict')

    def check_attempt(self, host: str) -> None:
        with self.lock:
            self.failures[host] = [t for t in self.failures[host] if t > time.time() - 900]
            if len(self.failures[host]) >= 8:
                raise HTTPException(429, '尝试次数过多，请 15 分钟后重试。')

    def failed_attempt(self, host: str) -> None:
        with self.lock:
            self.failures[host].append(time.time())
