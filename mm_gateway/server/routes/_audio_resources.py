"""Speech resources shared by REST and MCP, including voice ownership checks."""

from __future__ import annotations

from mm_gateway.core.exceptions import TaskNotFoundError
from mm_gateway.schemas.api import AudioTaskResponse, ResourceLinks, VoiceResponse
from mm_gateway.server.auth import authorize_task_access
from mm_gateway.server.routes._resources import (
    find_idempotent_record, new_record, remember_create_response, replay_resource,
    request_fingerprint, served_model,
)
from mm_gateway.translators.rest import from_audio_request, from_voice_request, to_audio_response, to_voice_response


async def create_resource(state, body, key, idempotency_key, url_for, *, voice=False):
    modality = "voice" if voice else "audio"
    service = state.voice_service if voice else state.audio_service
    translate = from_voice_request if voice else from_audio_request
    respond = to_voice_response if voice else to_audio_response
    response_type = VoiceResponse if voice else AudioTaskResponse
    store = state.task_store
    fingerprint = request_fingerprint(body)
    async with store.idempotency_guard(key.id, modality, idempotency_key):
        record = await find_idempotent_record(store, owner_key_id=key.id, modality=modality,
                                             idempotency_key=idempotency_key, fingerprint=fingerprint)
        if record:
            authorize_task_access(state.registry, key, record)
            return replay_resource(record, response_type, resource_url=url_for(record.task_id)), True
        task = await service.create(translate(body), key=key, routing=body.routing, wait=False)
        record = new_record("voice" if voice else "aud", task, model=served_model(body.model, task),
                            modality=modality, metadata=body.metadata, owner_key_id=key.id,
                            idempotency_key=idempotency_key,
                            request_fingerprint=fingerprint if idempotency_key else None)
        resource = respond(task, record, self_url=url_for(record.task_id))
        remember_create_response(record, resource)
        await store.put(record)
        return resource, False


async def get_audio_resource(state, key, id, self_url):
    record = await state.task_store.get(id)
    if record is None or record.modality != "audio":
        raise TaskNotFoundError("Audio task not found.")
    authorize_task_access(state.registry, key, record)
    task = await state.audio_service.get(record.provider_task_id or id, backend_name=record.provider)
    return to_audio_response(task, record, self_url=self_url)


async def get_voice_resource(state, key, id, self_url):
    record = await state.task_store.get(id)
    if record is not None and record.modality == "voice":
        authorize_task_access(state.registry, key, record)
        task = await state.voice_service.get(record.provider_task_id or id, backend_name=record.provider)
        return to_voice_response(task, record, self_url=self_url)
    if not id.startswith("voice_"):
        for name in state.registry.usable_for_modality(key, "audio"):
            if any(id in provider.voice_presets() for _, provider in state.registry.accounts_of(name)):
                return VoiceResponse(id=id, kind="preset", name=id, status="succeeded",
                                     links=ResourceLinks(self=self_url))
    raise TaskNotFoundError("Voice not found.")


async def list_voice_resources(state, key, url_for):
    presets = set()
    usable = state.registry.usable_for_modality(key, "audio")
    for name in usable:
        for _, provider in state.registry.accounts_of(name):
            presets.update(provider.voice_presets())
    resources = [VoiceResponse(id=id, kind="preset", name=id, status="succeeded",
                               links=ResourceLinks(self=url_for(id))) for id in sorted(presets)]
    for record in await state.task_store.list(key.id, "voice"):
        if record.provider in usable:
            resources.append(await get_voice_resource(state, key, record.task_id, url_for(record.task_id)))
    return resources
