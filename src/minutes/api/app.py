import re
import time
from collections.abc import Awaitable, Callable

from fastapi import FastAPI, Request, Response
from fastapi.exceptions import RequestValidationError
from fastapi.responses import FileResponse, JSONResponse

from minutes.api import routes_core, routes_labels
from minutes.config import ROOT
from minutes.errors import (
    DatabaseError,
    LabelRuleError,
    LabelsFrozenError,
    LabelsNotFrozenError,
    LabelValidationError,
    LLMQuotaError,
    LLMTransportError,
    MinutesError,
    NotFoundError,
    ValidationError,
)
from minutes.log import bind_correlation_id, get_logger, new_correlation_id

CORRELATION_HEADER = "X-Correlation-ID"
CORRELATION_ID = re.compile(r"^[0-9a-f]{32}$")
WEB_DIST = ROOT / "web" / "dist"

# status, error code, and whether the message is sent as "detail"
ERROR_RESPONSES: dict[type[MinutesError], tuple[int, str, bool]] = {
    ValidationError: (422, "validation_error", True),
    NotFoundError: (404, "not_found", True),
    DatabaseError: (503, "database_unavailable", False),
    LLMTransportError: (502, "llm_unavailable", False),
    LLMQuotaError: (503, "llm_quota", False),
    LabelsFrozenError: (409, "labels_frozen", False),
    LabelsNotFrozenError: (409, "labels_not_frozen", False),
    LabelValidationError: (422, "label_invalid", True),
    LabelRuleError: (409, "label_rules", True),
}

log = get_logger("api")


def create_app() -> FastAPI:
    app = FastAPI(title="Minutes")
    app.include_router(routes_core.router)
    app.include_router(routes_labels.router)

    @app.middleware("http")
    async def correlation_id(
        request: Request, call_next: Callable[[Request], Awaitable[Response]]
    ) -> Response:
        supplied = request.headers.get(CORRELATION_HEADER, "")
        cid = supplied if CORRELATION_ID.match(supplied) else new_correlation_id()
        request.state.correlation_id = cid
        bind_correlation_id(cid)
        started = time.perf_counter()
        response = await call_next(request)
        response.headers[CORRELATION_HEADER] = cid
        fields = {
            "method": request.method,
            "path": request.url.path,
            "status": response.status_code,
            "ms": int((time.perf_counter() - started) * 1000),
        }
        log.info("request done", extra={"event": "request_end", "fields": fields})
        return response

    @app.exception_handler(MinutesError)
    async def known_error(request: Request, err: MinutesError) -> JSONResponse:
        if type(err) not in ERROR_RESPONSES:
            return await unknown_error(request, err)
        status, code, with_detail = ERROR_RESPONSES[type(err)]
        body: dict[str, object] = {"error": code}
        if with_detail:
            body["detail"] = (
                err.args[0] if isinstance(err, (LabelValidationError, LabelRuleError)) else str(err)
            )
        return JSONResponse(body, status_code=status)

    @app.exception_handler(RequestValidationError)
    async def invalid_request(request: Request, err: RequestValidationError) -> JSONResponse:
        first = err.errors()[0]
        detail = ".".join(str(part) for part in first["loc"]) + ": " + first["msg"]
        return JSONResponse({"error": "validation_error", "detail": detail}, status_code=422)

    @app.exception_handler(Exception)
    async def unknown_error(request: Request, err: Exception) -> JSONResponse:
        # for an unexpected exception this runs outside the middleware, so it sets the header
        cid = getattr(request.state, "correlation_id", "-")
        log.error("request failed", exc_info=err, extra={"event": "request_failed"})
        return JSONResponse(
            {"error": "internal_error", "correlation_id": cid},
            status_code=500,
            headers={CORRELATION_HEADER: cid},
        )

    if WEB_DIST.is_dir():

        @app.get("/{path:path}", include_in_schema=False)
        def web(path: str) -> Response:
            if path.startswith("api/"):
                return JSONResponse({"error": "not_found", "detail": path}, status_code=404)
            file = WEB_DIST / path
            if path and file.is_file() and file.resolve().is_relative_to(WEB_DIST.resolve()):
                return FileResponse(file)
            return FileResponse(WEB_DIST / "index.html")

    return app
