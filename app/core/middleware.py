"""Small, process-local safeguards for a password-protected interview demo."""

import asyncio
import base64
import binascii
import logging
import secrets
import time
from collections import deque

from starlette.responses import JSONResponse

from app.core.logging import generate_request_id, request_id_ctx

logger = logging.getLogger(__name__)


class DemoMiddleware:
    def __init__(self, app, settings):
        self.app = app
        self.settings = settings
        self.requests = deque()
        self.uploading = False
        if settings.require_auth and len(settings.demo_password.get_secret_value()) < 16:
            raise RuntimeError("Public demo requires APP_DEMO_PASSWORD with at least 16 characters.")

    def authorized(self, headers):
        if not self.settings.require_auth:
            return True
        try:
            scheme, value = headers.get(b"authorization", b"").decode("ascii").split(" ", 1)
            if scheme.lower() != "basic":
                return False
            username, password = base64.b64decode(value, validate=True).decode("utf-8").split(":", 1)
            user_ok = secrets.compare_digest(username.encode(), self.settings.demo_username.encode())
            password_ok = secrets.compare_digest(
                password.encode(), self.settings.demo_password.get_secret_value().encode())
            return user_ok and password_ok
        except (ValueError, UnicodeError, binascii.Error):
            return False

    async def __call__(self, scope, receive, send):
        if scope["type"] != "http":
            return await self.app(scope, receive, send)
        request_id = generate_request_id()
        token = request_id_ctx.set(request_id)
        started = time.monotonic()
        status = 500

        async def send_response(message):
            nonlocal status
            if message["type"] == "http.response.start":
                status = message["status"]
                headers = list(message.get("headers", []))
                headers = [(k, v) for k, v in headers if k.lower() != b"x-request-id"]
                headers.extend([
                    (b"x-request-id", request_id.encode()),
                    (b"x-content-type-options", b"nosniff"),
                    (b"cache-control", b"no-store"),
                    (b"referrer-policy", b"no-referrer"),
                    (b"content-security-policy", b"default-src 'self'; script-src 'self'; "
                     b"style-src 'self'; img-src 'self' blob:; frame-src blob:; "
                     b"object-src 'none'; base-uri 'none'; frame-ancestors 'none'; form-action 'self'"),
                ])
                message = {**message, "headers": headers}
            await send(message)

        async def reject(code, detail, headers=None):
            await JSONResponse({"detail": detail}, code, headers=headers)(scope, receive, send_response)

        try:
            headers = dict(scope["headers"])
            if scope["path"] not in ("/live", "/ready") and not self.authorized(headers):
                return await reject(401, "Demo login required.", {"WWW-Authenticate": 'Basic realm="Document demo", charset="UTF-8"'})
            if scope["method"] != "POST" or scope["path"] != "/correct-orientation":
                return await self.app(scope, receive, send_response)
            now = time.monotonic()
            while self.requests and self.requests[0] <= now - 60:
                self.requests.popleft()
            if len(self.requests) >= self.settings.requests_per_minute:
                return await reject(429, "Demo request limit reached. Try again in a minute.", {"Retry-After": "60"})
            self.requests.append(now)
            # Bound the body before Starlette parses multipart uploads, including chunked bodies.
            limit = self.settings.max_image_size_mb * 1024 * 1024 + 64 * 1024
            try:
                declared = int(headers.get(b"content-length", b"0"))
                if declared < 0:
                    raise ValueError
            except ValueError:
                return await reject(400, "Invalid Content-Length.")
            if declared > limit:
                return await reject(413, f"File too large: upload exceeds the {self.settings.max_image_size_mb} MB limit.")
            if self.uploading:
                return await reject(429, "Another upload is in progress. Try again shortly.", {"Retry-After": "2"})
            self.uploading = True
            try:
                body = bytearray()
                async with asyncio.timeout(self.settings.upload_timeout_seconds):
                    while True:
                        message = await receive()
                        if message["type"] == "http.disconnect":
                            return
                        body.extend(message.get("body", b""))
                        if len(body) > limit:
                            return await reject(413, f"File too large: upload exceeds the {self.settings.max_image_size_mb} MB limit.")
                        if not message.get("more_body", False):
                            break

                async def bounded_receive():
                    nonlocal body
                    if body is not None:
                        data, body = bytes(body), None
                        return {"type": "http.request", "body": data, "more_body": False}
                    return await receive()

                await self.app(scope, bounded_receive, send_response)
            except TimeoutError:
                await reject(408, "Upload timed out.")
            finally:
                self.uploading = False
        finally:
            if scope["path"] not in ("/live", "/ready"):
                logger.info("request method=%s path=%s status=%s duration_ms=%.1f",
                            scope["method"], scope["path"], status,
                            (time.monotonic() - started) * 1000)
            request_id_ctx.reset(token)
