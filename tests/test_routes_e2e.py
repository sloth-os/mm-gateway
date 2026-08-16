"""End-to-end tests for the public REST resource contracts."""

from __future__ import annotations

from mm_gateway.core.exceptions import ProviderRequestError


def _poll_until_done(client, url: str) -> dict:
    for _ in range(10):
        response = client.get(url)
        assert response.status_code == 200, response.text
        body = response.json()
        if body["status"] in {"succeeded", "failed", "cancelled", "expired"}:
            return body
    raise AssertionError(f"task at {url} did not reach a terminal state")


def _text(value: str) -> list[dict[str, str]]:
    return [{"type": "text", "text": value}]


# -- Meta routes ------------------------------------------------------------ #


def test_health(client):
    response = client.get("/health")
    assert response.status_code == 200
    assert response.json() == {"status": "ok"}


def test_models_can_be_filtered_by_modality(client):
    response = client.get("/v1/models", params={"modality": "image"})
    assert response.status_code == 200
    assert response.json()["object"] == "list"
    assert response.json()["data"]
    assert {model["modality"] for model in response.json()["data"]} == {"image"}
    assert all(set(model) == {"id", "object", "modality"}
               for model in response.json()["data"])
    assert response.headers["cache-control"] == "private, max-age=60"

    unchanged = client.get(
        "/v1/models",
        params={"modality": "image"},
        headers={"if-none-match": response.headers["etag"]},
    )
    assert unchanged.status_code == 304
    assert unchanged.content == b""


def test_model_limits_endpoint_returns_limits(client):
    response = client.get("/v1/models/limits")
    assert response.status_code == 200
    body = response.json()
    assert body["object"] == "list"
    assert body["data"]
    # Every entry carries a neutral limits object with at least its modality.
    for model in body["data"]:
        assert set(model) == {"id", "object", "modality", "limits"}
        assert model["limits"]["modality"] == model["modality"]
    # The same conditional-retrieval contract as /v1/models.
    assert response.headers["cache-control"] == "private, max-age=60"
    unchanged = client.get(
        "/v1/models/limits",
        headers={"if-none-match": response.headers["etag"]},
    )
    assert unchanged.status_code == 304
    assert unchanged.content == b""


def test_model_limits_can_be_filtered_by_modality(client):
    response = client.get("/v1/models/limits", params={"modality": "music"})
    assert response.status_code == 200
    data = response.json()["data"]
    assert data
    assert {model["modality"] for model in data} == {"music"}


def test_metrics_exposes_prometheus(client):
    client.post("/v1/images", json={"model": "fake-image-1", "input": _text("x")})
    response = client.get("/metrics")
    assert response.status_code == 200
    assert "gateway_requests_total" in response.text


def test_request_id_middleware(client):
    response = client.get("/health", headers={"x-request-id": "abc-123"})
    assert response.headers["x-request-id"] == "abc-123"


# -- Image resources -------------------------------------------------------- #


def test_image_create_returns_task_resource_and_rest_headers(client, fake_provider):
    response = client.post("/v1/images", json={
        "model": "fake-image-1",
        "input": _text("a cat"),
        "parameters": {
            "output_count": 2,
            "dimensions": {"width": 1024, "height": 1024},
            "delivery": "inline",
        },
        "metadata": {"job": "cover"},
    })
    assert response.status_code == 202, response.text
    body = response.json()
    assert body["id"].startswith("img_")
    assert body["object"] == "image"
    assert body["model"] == "fake-image-1"
    assert body["status"] == "pending"
    assert body["outputs"] == []
    assert body["metadata"] == {"job": "cover"}
    assert body["links"]["self"] == response.headers["location"]
    assert response.headers["location"].endswith(f"/v1/images/{body['id']}")
    assert response.headers["retry-after"] == "1"
    assert response.headers["cache-control"] == "private, no-cache"
    assert response.headers["etag"].startswith('"')

    request = fake_provider.image_calls[0]
    assert request.prompt() == "a cat"
    assert request.n == 2
    assert request.width == 1024
    assert request.height == 1024
    assert request.response_format == "b64_json"


def test_image_auto_routes_when_model_is_omitted(client, fake_provider):
    # No "model" field -> the gateway auto-routes to a fitting backend. The
    # response echoes the resolved model id (not "auto" / None).
    response = client.post("/v1/images", json={"input": _text("auto-routed cat")})
    assert response.status_code == 202, response.text
    body = response.json()
    assert body["model"] == "fake-image-1"
    assert fake_provider.image_calls[-1].model == "fake-image-1"


def test_image_auto_routes_when_model_is_auto(client, fake_provider):
    response = client.post(
        "/v1/images", json={"model": "auto", "input": _text("auto string")}
    )
    assert response.status_code == 202, response.text
    assert response.json()["model"] == "fake-image-1"


def test_idempotent_replay_across_omitted_and_auto_model(client, fake_provider):
    # Omitting `model` and sending "auto" are documented as equivalent, so the
    # same Idempotency-Key must replay across the two spellings (not 409).
    headers = {"idempotency-key": "auto-equiv-1"}
    first = client.post("/v1/images", json={"input": _text("a cat")}, headers=headers)
    replay = client.post(
        "/v1/images",
        json={"model": "auto", "input": _text("a cat")},
        headers=headers,
    )
    assert first.status_code == replay.status_code == 202
    assert first.json() == replay.json()
    assert replay.headers["idempotency-replayed"] == "true"
    assert len(fake_provider.image_calls) == 1


def test_auto_route_no_fit_returns_422_validation_error():
    # When no configured model can serve an auto-routed request, the gateway
    # returns 422 validation_error — not 404 model_not_found — because no
    # specific model id was requested; the request's input is incompatible.
    from fastapi.testclient import TestClient

    from mm_gateway.config import BackendConfig, KeyConfig, Settings
    from mm_gateway.server.app import create_app
    from tests.conftest import FakeProvider

    settings = Settings(
        backends=[BackendConfig(name="fake", type="fake", api_key="test")],
        keys=[KeyConfig(id="test", key="")],
    )
    app = create_app(settings)
    provider = FakeProvider(settings.backends[0])
    provider.image_models = ["dall-e-3"]  # text-only -> rejects image input
    app.state.registry._backends["fake"] = provider
    app.state.registry._configs["fake"] = settings.backends[0]

    response = TestClient(app).post("/v1/images", json={
        "model": "auto",
        "input": [
            {"type": "text", "text": "edit this"},
            {"type": "image", "uri": "https://example.test/a.png"},
        ],
    })
    assert response.status_code == 422, response.text
    assert response.headers["content-type"].startswith("application/problem+json")
    assert response.json()["code"] == "validation_error"


def test_create_is_idempotent(client, fake_provider):
    payload = {
        "model": "fake-image-1",
        "input": _text("a cat"),
        "metadata": {"job": "cover"},
    }
    headers = {"idempotency-key": "cover-001"}

    first = client.post("/v1/images", json=payload, headers=headers)
    replay = client.post("/v1/images", json=payload, headers=headers)

    assert first.status_code == replay.status_code == 202
    assert first.json() == replay.json()
    assert first.headers["location"] == replay.headers["location"]
    assert replay.headers["idempotency-replayed"] == "true"
    assert len(fake_provider.image_calls) == 1


def test_idempotency_key_cannot_be_reused_for_a_different_request(
    client, fake_provider
):
    headers = {"idempotency-key": "cover-001"}
    first = client.post(
        "/v1/images",
        json={"model": "fake-image-1", "input": _text("a cat")},
        headers=headers,
    )
    conflict = client.post(
        "/v1/images",
        json={"model": "fake-image-1", "input": _text("a dog")},
        headers=headers,
    )

    assert first.status_code == 202
    assert conflict.status_code == 409
    assert conflict.headers["content-type"].startswith("application/problem+json")
    assert conflict.json()["code"] == "idempotency_conflict"
    assert len(fake_provider.image_calls) == 1


def test_image_accepts_ordered_text_and_multiple_images(client, fake_provider):
    response = client.post("/v1/images", json={
        "model": "fake-image-1",
        "input": [
            {"type": "image", "uri": "https://example.test/one.png"},
            {"type": "text", "text": "combine these references"},
            {"type": "image", "uri": "data:image/png;base64,AAAA"},
        ],
    })
    assert response.status_code == 202, response.text
    request = fake_provider.image_calls[0]
    assert request.prompt() == "combine these references"
    assert len(request.input_images()) == 2
    assert request.input_images()[0].url == "https://example.test/one.png"
    assert request.input_images()[1].data == "AAAA"


def test_image_poll_returns_normalized_outputs(client):
    created = client.post(
        "/v1/images", json={"model": "fake-image-1", "input": _text("a cat")}
    )
    task_id = created.json()["id"]
    body = _poll_until_done(client, created.headers["location"])
    assert body["id"] == task_id
    assert body["object"] == "image"
    assert body["outputs"] == [{
        "uri": "https://example.test/out.png",
        "revised_prompt": "a cat",
    }]
    assert body["links"]["self"].endswith(f"/v1/images/{task_id}")


def test_poll_supports_etag_revalidation(client):
    created = client.post(
        "/v1/images", json={"model": "fake-image-1", "input": _text("a cat")}
    )
    first = client.get(created.headers["location"])
    completed = client.get(created.headers["location"])
    assert first.json()["created_at"] == completed.json()["created_at"]
    assert completed.json()["status"] == "succeeded"

    unchanged = client.get(
        created.headers["location"],
        headers={"if-none-match": completed.headers["etag"]},
    )
    assert unchanged.status_code == 304
    assert unchanged.content == b""
    assert unchanged.headers["etag"] == completed.headers["etag"]


def test_image_unknown_task_is_404(client):
    response = client.get("/v1/images/img_unknown")
    assert response.status_code == 404
    assert response.json()["code"] == "task_not_found"


# -- Video resources -------------------------------------------------------- #


def test_video_accepts_text_images_audio_and_video(client, fake_provider):
    response = client.post("/v1/videos", json={
        "model": "fake-video-1",
        "input": [
            {"type": "text", "text": "cut between all references"},
            {"type": "image", "uri": "https://example.test/first.png", "role": "first_frame"},
            {"type": "image", "uri": "https://example.test/ref.png", "role": "reference_image"},
            {"type": "audio", "uri": "data:audio/wav;base64,AAAA", "role": "reference_audio"},
            {"type": "audio", "uri": "https://example.test/score.mp3", "role": "reference_audio"},
            {"type": "video", "uri": "https://example.test/ref.mp4", "role": "reference_video"},
            {"type": "video", "uri": "data:video/webm;base64,BBBB", "role": "reference_video"},
        ],
        "parameters": {
            "duration_seconds": 5,
            "dimensions": {"width": 1280, "height": 720},
            "include_audio": True,
            "camera_motion": "fixed",
            "enhance_prompt": True,
            "guidance_scale": 4.5,
            "motion_intensity": 140,
            "frame_count": 121,
            "file_format": "mp4",
        },
    })
    assert response.status_code == 202, response.text
    assert response.json()["id"].startswith("vid_")
    assert response.json()["object"] == "video"

    request = fake_provider.video_calls[0]
    assert request.prompt() == "cut between all references"
    assert request.first_image() == "https://example.test/first.png"
    assert request.reference_images() == ["https://example.test/ref.png"]
    assert request.reference_audios()[0].startswith("data:audio/wav;base64,")
    assert request.reference_audios()[1] == "https://example.test/score.mp3"
    assert request.reference_videos()[0] == "https://example.test/ref.mp4"
    assert request.reference_videos()[1].startswith("data:video/webm;base64,")
    assert request.duration == 5
    assert request.width == 1280
    assert request.height == 720
    assert request.generate_audio is True
    assert request.camera_fixed is True
    assert request.prompt_extend is True
    assert request.guidance_scale == 4.5
    assert request.motion_intensity == 140
    assert request.frame_count == 121
    assert request.output_format == "mp4"
    assert request.extra == {}


def test_video_poll_returns_normalized_outputs_and_usage(client):
    created = client.post(
        "/v1/videos", json={"model": "fake-video-1", "input": _text("a cat playing")}
    )
    body = _poll_until_done(client, created.headers["location"])
    assert body["object"] == "video"
    assert body["outputs"] == [{"uri": "https://example.test/out.mp4"}]
    assert body["usage"] == {"cost": 0.01, "output_count": 1}


def test_task_id_cannot_be_used_on_a_different_collection(client):
    created = client.post(
        "/v1/videos", json={"model": "fake-video-1", "input": _text("x")}
    )
    response = client.get(f"/v1/images/{created.json()['id']}")
    assert response.status_code == 404


# -- Music resources -------------------------------------------------------- #


def test_music_accepts_text_image_and_audio_inputs(client, fake_provider):
    response = client.post("/v1/music", json={
        "model": "fake-music-1",
        "input": [
            {"type": "text", "text": "verse"},
            {"type": "text", "text": "chorus"},
            {"type": "lyrics", "text": "we sing together"},
            {"type": "lyrics", "text": "under one sky"},
            {"type": "image", "uri": "https://example.test/cover.png"},
            {"type": "image", "uri": "data:image/jpeg;base64,BBBB"},
            {"type": "audio", "uri": "https://example.test/reference.wav", "role": "reference_audio"},
            {"type": "audio", "uri": "data:audio/mpeg;base64,CCCC", "role": "continuation_audio"},
        ],
        "parameters": {
            "file_format": "wav",
            "sample_rate_hz": 44100,
            "bitrate_kbps": 192,
            "duration_seconds": 30,
            "style": "cinematic pop",
            "instrumental": False,
            "output_count": 2,
            "enhance_lyrics": True,
            "voice": "warm-alto",
            "vocal_gender": "female",
            "style_strength": 0.8,
            "novelty": 0.3,
            "reference_audio_strength": 0.7,
            "inference_steps": 30,
            "respect_section_durations": True,
            "provenance": True,
        },
    })
    assert response.status_code == 202, response.text
    assert response.json()["id"].startswith("mus_")
    request = fake_provider.music_calls[0]
    assert request.prompt() == "verse\nchorus"
    assert request.lyrics == "we sing together\nunder one sky"
    assert request.reference_images()[0] == "https://example.test/cover.png"
    assert request.reference_images()[1].startswith("data:image/jpeg;base64,")
    assert request.reference_audios() == ["https://example.test/reference.wav"]
    assert request.continuation_audio().startswith("data:audio/mpeg;base64,")
    assert request.audio_format == "wav"
    assert request.sample_rate_hz == 44100
    assert request.bitrate_kbps == 192
    assert request.duration == 30
    assert request.style == "cinematic pop"
    assert request.is_instrumental is False
    assert request.n == 2
    assert request.enhance_lyrics is True
    assert request.voice == "warm-alto"
    assert request.vocal_gender == "female"
    assert request.style_strength == 0.8
    assert request.novelty == 0.3
    assert request.reference_audio_strength == 0.7
    assert request.inference_steps == 30
    assert request.respect_section_durations is True
    assert request.provenance is True
    assert request.extra == {}


def test_music_poll_returns_audio_lyrics_and_usage(client):
    created = client.post(
        "/v1/music", json={"model": "fake-music-1", "input": _text("a sad ballad")}
    )
    body = _poll_until_done(client, created.headers["location"])
    assert body["object"] == "music"
    assert body["outputs"] == [{
        "uri": "data:audio/wav;base64,AAAA",
        "mime_type": "audio/wav",
    }]
    assert body["lyrics"] == "la la la"
    assert body["usage"] == {
        "cost": 0.01,
        "output_count": 1,
        "duration_seconds": 8.0,
    }


# -- Validation ------------------------------------------------------------- #


def test_validation_uses_problem_details(client):
    # The envelope is strict: an unknown top-level field yields 422. (``model``
    # is optional now — omitting it triggers auto-routing, not validation.)
    response = client.post("/v1/images", json={"input": _text("strict envelope"), "bogus": 1})
    assert response.status_code == 422
    assert response.headers["content-type"].startswith("application/problem+json")
    problem = response.json()
    assert problem["type"] == "urn:mm-gateway:problem:validation_error"
    assert problem["title"] == "Validation Error"
    assert problem["status"] == 422
    assert problem["code"] == "validation_error"
    assert problem["instance"] == "/v1/images"
    assert problem["request_id"] == response.headers["x-request-id"]
    assert problem["errors"]


def test_upstream_failures_are_sanitized_problem_details(
    client, fake_provider, monkeypatch
):
    async def fail_create(_request, **_kwargs):
        raise ProviderRequestError(
            "provider wire error with private context",
            provider="private-provider",
            details={"upstream_body": "secret"},
        )

    monkeypatch.setattr(fake_provider, "create_image_task", fail_create)
    response = client.post(
        "/v1/images",
        json={"model": "fake-image-1", "input": _text("x")},
    )
    problem = response.json()

    assert response.status_code == 502
    assert response.headers["content-type"].startswith("application/problem+json")
    assert problem["code"] == "generation_service_error"
    assert problem["detail"] == "The generation service returned an error."
    assert "provider" not in problem
    assert "secret" not in response.text


def test_top_level_unknown_fields_are_rejected(client):
    response = client.post("/v1/images", json={
        "model": "fake-image-1",
        "input": _text("x"),
        "size": "1024x1024",
    })
    assert response.status_code == 422


def test_provider_specific_parameters_are_rejected(client):
    response = client.post("/v1/videos", json={
        "model": "fake-video-1",
        "input": _text("x"),
        "parameters": {"service_tier": "provider-specific"},
    })
    assert response.status_code == 422


def test_routing_rejects_provider_or_backend_selectors(client):
    response = client.post("/v1/images", json={
        "model": "fake-image-1",
        "input": _text("x"),
        "routing": {"backend": "fake", "tag": "prod"},
    })
    assert response.status_code == 422


def test_visual_dimensions_have_one_canonical_shape(client, fake_provider):
    accepted = client.post("/v1/images", json={
        "model": "fake-image-1",
        "input": _text("x"),
        "parameters": {"dimensions": {"width": 1024, "height": 768}},
    })
    missing_height = client.post("/v1/images", json={
        "model": "fake-image-1",
        "input": _text("x"),
        "parameters": {"dimensions": {"width": 1024}},
    })
    legacy_fields = ("size", "width", "height", "aspect_ratio", "resolution")
    legacy_responses = [
        client.post("/v1/images", json={
            "model": "fake-image-1",
            "input": _text("x"),
            "parameters": {field: 1024 if field in {"width", "height"} else "legacy"},
        })
        for field in legacy_fields
    ]
    assert accepted.status_code == 202
    assert fake_provider.image_calls[0].width == 1024
    assert fake_provider.image_calls[0].height == 768
    assert missing_height.status_code == 422
    assert all(response.status_code == 422 for response in legacy_responses)


def test_media_parts_require_one_absolute_uri_field(client):
    valid = client.post("/v1/images", json={
        "model": "fake-image-1",
        "input": [{"type": "image", "uri": "data:image/png;base64,AAAA"}],
    })
    legacy_source = client.post("/v1/images", json={
        "model": "fake-image-1",
        "input": [{"type": "image", "url": "https://example.test/image.png"}],
    })
    relative_uri = client.post("/v1/images", json={
        "model": "fake-image-1",
        "input": [{"type": "image", "uri": "/image.png"}],
    })
    raw_base64 = client.post("/v1/images", json={
        "model": "fake-image-1",
        "input": [{"type": "image", "uri": "AAAA"}],
    })
    assert valid.status_code == 202
    assert legacy_source.status_code == 422
    assert relative_uri.status_code == 422
    assert raw_base64.status_code == 422


def test_duration_is_explicitly_seconds_and_positive(client, fake_provider):
    accepted = client.post("/v1/videos", json={
        "model": "fake-video-1",
        "input": _text("x"),
        "parameters": {"duration_seconds": 2.5},
    })
    rejected = client.post("/v1/videos", json={
        "model": "fake-video-1",
        "input": _text("x"),
        "parameters": {"duration_seconds": 0},
    })
    legacy_name = client.post("/v1/videos", json={
        "model": "fake-video-1",
        "input": _text("x"),
        "parameters": {"duration": 2.5},
    })
    assert accepted.status_code == 202
    assert fake_provider.video_calls[0].duration == 2.5
    assert rejected.status_code == 422
    assert legacy_name.status_code == 422


def test_input_requires_a_non_empty_typed_parts_array(client):
    resources = (
        ("/v1/images", "fake-image-1"),
        ("/v1/videos", "fake-video-1"),
        ("/v1/music", "fake-music-1"),
    )
    for path, model in resources:
        string_input = client.post(path, json={"model": model, "input": "x"})
        empty_input = client.post(path, json={"model": model, "input": []})
        assert string_input.status_code == 422
        assert empty_input.status_code == 422
