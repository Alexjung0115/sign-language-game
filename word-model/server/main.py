from __future__ import annotations

import asyncio
from functools import partial
from http import HTTPStatus
import json
import os

from websockets.asyncio.server import (
    Request,
    Response,
    ServerConnection,
    serve,
)

from .frame_adapter import request_to_model_frame
from .messages import (
    ClientRequest,
    GetCapabilitiesRequest,
    HandNotDetectedRequest,
    ProtocolError,
    ResetSequenceRequest,
    WordLandmarkFrameRequest,
    capabilities_message,
    error_message,
    json_ready,
    parse_request,
)
from .recognition_session import RecognitionSession
from wordmodel.adapter import WordModelAdapter


HOST = os.getenv("HANDPRACTICE_WORD_HOST", "localhost")
PORT = int(os.getenv("HANDPRACTICE_WORD_PORT", "8767"))
PATH = os.getenv("HANDPRACTICE_WORD_PATH", "/word")
INFERENCE_WORKERS = int(
    os.getenv("HANDPRACTICE_WORD_INFERENCE_WORKERS", "1"),
)


def normalise_path(raw_path: str) -> str:
    path = raw_path.split("?", 1)[0].split("#", 1)[0]
    return path.rstrip("/") or "/"


def check_path(
    connection: ServerConnection,
    request: Request,
) -> Response | None:
    if normalise_path(request.path) == normalise_path(PATH):
        return None
    body = (
        f"Not found: {request.path}\n"
        f"This is the word-sign model server. "
        f"Connect to ws://{HOST}:{PORT}{PATH}\n"
    )
    return connection.respond(HTTPStatus.NOT_FOUND, body)


def process_parsed_request(
    session: RecognitionSession,
    request: ClientRequest,
) -> list[dict[str, object]]:
    if isinstance(request, GetCapabilitiesRequest):
        contract = session.contract
        config = session.config
        return [
            capabilities_message(
                model_version=contract.model_version,
                labels=contract.labels,
                sequence_length=contract.sequence_length,
                minimum_frames=contract.minimum_frames,
                recommended_fps=config.recommended_fps,
                confidence_threshold=config.confidence_threshold,
            ),
        ]
    if isinstance(request, WordLandmarkFrameRequest):
        return session.process_model_frame(
            frame_id=request.frame_id,
            captured_at=request.captured_at,
            frame=request_to_model_frame(request),
        )
    if isinstance(request, HandNotDetectedRequest):
        return session.process_hand_not_detected(
            request.captured_at,
        )
    if isinstance(request, ResetSequenceRequest):
        session.reset()
        return []
    raise ProtocolError(
        "UNSUPPORTED_MESSAGE",
        "Unsupported message type",
    )


def process_request(
    session: RecognitionSession,
    raw_message: str,
) -> list[dict[str, object]]:
    return process_parsed_request(
        session,
        parse_request(raw_message),
    )


async def websocket_handler(
    connection: ServerConnection,
    adapter: WordModelAdapter,
    inference_slots: asyncio.Semaphore,
) -> None:
    session = RecognitionSession(adapter)
    frame_count = 0
    missing_count = 0
    connection_label = f"{id(connection):x}"
    print(f"[word:{connection_label}] connected", flush=True)
    try:
        async for raw_message in connection:
            try:
                if not isinstance(raw_message, str):
                    raise ProtocolError(
                        "INVALID_MESSAGE",
                        "Binary messages are not supported",
                    )
                request = parse_request(raw_message)
                if isinstance(request, WordLandmarkFrameRequest):
                    frame_count += 1
                    async with inference_slots:
                        responses = await asyncio.to_thread(
                            process_parsed_request,
                            session,
                            request,
                        )
                    if frame_count == 1 or frame_count % 30 == 0:
                        response_types = ",".join(
                            str(item.get("type"))
                            for item in responses
                        ) or "NONE"
                        print(
                            f"[word:{connection_label}] "
                            f"landmarks={frame_count} "
                            f"responses={response_types}",
                            flush=True,
                        )
                else:
                    responses = process_parsed_request(
                        session,
                        request,
                    )
                    if isinstance(request, HandNotDetectedRequest):
                        missing_count += 1
            except ProtocolError as error:
                responses = [
                    error_message(error.code, error.message),
                ]
            except ValueError as error:
                responses = [
                    error_message("INVALID_INPUT", str(error)),
                ]
            except Exception as error:
                print(
                    f"[word:{connection_label}] "
                    f"internal-error={type(error).__name__}",
                    flush=True,
                )
                responses = [
                    error_message(
                        "INTERNAL_ERROR",
                        "Recognition request failed",
                    ),
                ]
            for response in responses:
                await connection.send(
                    json.dumps(
                        json_ready(response),
                        ensure_ascii=False,
                    ),
                )
    finally:
        session.reset()
        print(
            f"[word:{connection_label}] disconnected "
            f"landmarks={frame_count} hand-missing={missing_count}",
            flush=True,
        )


async def run_server(
    adapter: WordModelAdapter,
    host: str | None = None,
    port: int | None = None,
) -> None:
    if INFERENCE_WORKERS <= 0:
        raise ValueError(
            "HANDPRACTICE_WORD_INFERENCE_WORKERS must be positive",
        )
    selected_host = host if host is not None else HOST
    selected_port = port if port is not None else PORT
    inference_slots = asyncio.Semaphore(INFERENCE_WORKERS)
    handler = partial(
        websocket_handler,
        adapter=adapter,
        inference_slots=inference_slots,
    )
    async with serve(
        handler,
        selected_host,
        selected_port,
        process_request=check_path,
    ):
        print(
            "Word AI WebSocket server listening on "
            f"ws://{selected_host}:{selected_port}{PATH}",
        )
        await asyncio.get_running_loop().create_future()


def main() -> None:
    adapter = WordModelAdapter()
    asyncio.run(run_server(adapter))


if __name__ == "__main__":
    main()
