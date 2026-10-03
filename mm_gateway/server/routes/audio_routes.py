"""Provider-neutral speech synthesis and reusable voice clone resources."""

from __future__ import annotations

from typing import Annotated

from fastapi import APIRouter, Body, Depends, Path, Request

from mm_gateway.config import KeyConfig
from mm_gateway.schemas.api import AudioRequest, AudioTaskResponse, VoiceCloneRequest, VoiceListResponse, VoiceResponse
from mm_gateway.server.auth import get_api_key
from mm_gateway.server.routes._audio_resources import (
    create_resource, get_audio_resource, get_voice_resource, list_voice_resources,
)
from mm_gateway.server.routes._resources import (
    POLL_HEADERS, RESOURCE_HEADERS, IdempotencyKeyHeader, IfNoneMatchHeader,
    render_conditional_json, render_resource,
)

router = APIRouter(tags=["audio"])


@router.post("/v1/audio", name="create_audio", operation_id="createAudio", summary="Create a speech task",
             status_code=202, response_model=AudioTaskResponse, response_model_exclude_none=True,
             responses={202: {"description": "Speech synthesis accepted.", "headers": RESOURCE_HEADERS}})
async def create_audio(request: Request, body: Annotated[AudioRequest, Body()],
                       key: Annotated[KeyConfig, Depends(get_api_key)],
                       idempotency_key: IdempotencyKeyHeader = None):
    resource, replayed = await create_resource(request.app.state, body, key, idempotency_key,
                      lambda id: str(request.url_for("get_audio", audio_id=id)))
    return render_resource(resource, request, resource_url=resource.links.self_url, created=True, replayed=replayed)


@router.get("/v1/audio/{audio_id}", name="get_audio", operation_id="getAudio", summary="Retrieve a speech task",
            response_model=AudioTaskResponse, response_model_exclude_none=True,
            responses={200: {"description": "Current speech task state.", "headers": POLL_HEADERS},
                       304: {"description": "Unchanged speech task."}})
async def get_audio(request: Request, audio_id: Annotated[str, Path(description="Opaque speech task id.")],
                    key: Annotated[KeyConfig, Depends(get_api_key)], if_none_match: IfNoneMatchHeader = None):
    url = str(request.url_for("get_audio", audio_id=audio_id))
    resource = await get_audio_resource(request.app.state, key, audio_id, url)
    return render_resource(resource, request, resource_url=url)


@router.post("/v1/voices", name="create_voice", operation_id="createVoice", summary="Clone a reusable voice",
             status_code=202, response_model=VoiceResponse, response_model_exclude_none=True,
             responses={202: {"description": "Voice cloning accepted.", "headers": RESOURCE_HEADERS}})
async def create_voice(request: Request, body: Annotated[VoiceCloneRequest, Body()],
                       key: Annotated[KeyConfig, Depends(get_api_key)],
                       idempotency_key: IdempotencyKeyHeader = None):
    resource, replayed = await create_resource(request.app.state, body, key, idempotency_key,
                      lambda id: str(request.url_for("get_voice", voice_id=id)), voice=True)
    return render_resource(resource, request, resource_url=resource.links.self_url, created=True, replayed=replayed)


@router.get("/v1/voices", name="list_voices", operation_id="listVoices", summary="List usable voice presets and owned clones",
            response_model=VoiceListResponse, response_model_exclude_none=True,
            responses={200: {"description": "Voices available to this key.", "headers": {"ETag": RESOURCE_HEADERS["ETag"]}},
                       304: {"description": "Unchanged voice list."}})
async def list_voices(request: Request, key: Annotated[KeyConfig, Depends(get_api_key)],
                      if_none_match: IfNoneMatchHeader = None):
    voices = await list_voice_resources(request.app.state, key,
                        lambda id: str(request.url_for("get_voice", voice_id=id)))
    body = VoiceListResponse(data=voices).model_dump(mode="json", by_alias=True, exclude_none=True)
    return render_conditional_json(body, request)


@router.get("/v1/voices/{voice_id}", name="get_voice", operation_id="getVoice", summary="Retrieve a voice or clone task",
            response_model=VoiceResponse, response_model_exclude_none=True,
            responses={200: {"description": "Current voice state.", "headers": POLL_HEADERS},
                       304: {"description": "Unchanged voice."}})
async def get_voice(request: Request, voice_id: Annotated[str, Path(description="Gateway voice id.")],
                    key: Annotated[KeyConfig, Depends(get_api_key)], if_none_match: IfNoneMatchHeader = None):
    url = str(request.url_for("get_voice", voice_id=voice_id))
    resource = await get_voice_resource(request.app.state, key, voice_id, url)
    return render_resource(resource, request, resource_url=url)
