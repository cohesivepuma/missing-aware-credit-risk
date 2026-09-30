"""Cap streamed request bodies before multipart parsing can spool unlimited data."""
from starlette.responses import JSONResponse


class RequestLimitExceeded(Exception):
    pass


class BodyLimitMiddleware:
    def __init__(self, app, max_bytes: int) -> None:
        self.app = app
        self.max_bytes = max_bytes

    async def __call__(self, scope, receive, send):
        if scope['type'] != 'http':
            return await self.app(scope, receive, send)
        received, exceeded, started = 0, False, False

        async def bounded_receive():
            nonlocal received, exceeded
            message = await receive()
            if message['type'] == 'http.request':
                received += len(message.get('body', b''))
                if received > self.max_bytes:
                    exceeded = True
                    raise RequestLimitExceeded()
            return message

        async def guarded_send(message):
            nonlocal started
            # Multipart parsers may translate the read exception to HTTP 400;
            # replace that response with the accurate size-limit status.
            if exceeded and not started:
                return
            if message['type'] == 'http.response.start':
                started = True
            await send(message)

        try:
            await self.app(scope, bounded_receive, guarded_send)
        except Exception:
            if not exceeded or started:
                raise
        if exceeded and not started:
            await JSONResponse({'detail': '请求总大小不能超过 20 MiB（另含少量表单元数据）。'}, status_code=413)(scope, receive, send)
