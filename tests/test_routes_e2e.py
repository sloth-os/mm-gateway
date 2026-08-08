"""End-to-end route tests with a fake in-memory provider.

These exercise the full FastAPI stack: HTTP request -> translator -> service ->
fake provider -> translator -> JSON response, with no network calls.
"""

from __future__ import annotations

# -- Meta routes ------------------------------------------------------------ #

def test_health(client):
    r = client.get("/health")
    assert r.status_code == 200
    assert r.json() == {"status": "ok"}


def test_models_lists_fake_models(client):
    r = client.get("/v1/models")
    assert r.status_code == 200
    ids = [m["id"] for m in r.json()["data"]]
    assert "fake-image-1" in ids
    assert "fake-video-1" in ids


def test_metrics_exposes_prometheus(client):
    # Make a real request first so the metrics store has a counter to render.
    client.post("/v1/images", json={"model": "fake-image-1", "input": "x"})
    r = client.get("/metrics")
    assert r.status_code == 200
    assert "gateway_requests_total" in r.text


def test_request_id_middleware(client):
    r = client.get("/health", headers={"x-request-id": "abc-123"})
    assert r.headers["x-request-id"] == "abc-123"


# -- Image routes ------------------------------------------------------------ #

def test_image_create_sync_returns_just_id(client, fake_provider):
    # image_sync_default defaults to True, so create blocks until success and
    # returns only the Gemini interaction id.
    r = client.post("/v1/images", json={"model": "fake-image-1", "input": "a cat", "n": 1})
    assert r.status_code == 200, r.text
    assert r.json() == {"id": "img-1"}
    assert len(fake_provider.image_calls) == 1
    assert fake_provider.image_calls[0].model == "fake-image-1"
    assert fake_provider.image_calls[0].prompt() == "a cat"


def test_image_async_then_poll_returns_steps_content(client, fake_provider):
    # Respond-async: create returns immediately without waiting.
    r = client.post("/v1/images", json={"model": "fake-image-1", "input": "a cat"},
                    headers={"prefer": "respond-async"})
    assert r.status_code == 200, r.text
    task_id = r.json()["id"]

    # Poll until terminal.
    for _ in range(10):
        poll = client.get(f"/v1/images/{task_id}")
        assert poll.status_code == 200, poll.text
        if poll.json()["status"] == "succeeded":
            break
    body = poll.json()
    assert body["id"] == task_id
    assert body["status"] == "succeeded"
    # The image rides a model_output step's content array as a typed block.
    step = body["steps"][0]
    assert step["type"] == "model_output"
    blocks = step["content"]
    assert any(b["type"] == "image" and b["url"] == "https://example.test/out.png" for b in blocks)
    # Convenience accessor mirrors the SDK's interaction.output_image / output_image_url.
    assert body["output_image_url"] == "https://example.test/out.png"
    # revised_prompt surfaces as a text block.
    assert any(b["type"] == "text" and b["text"] == "a cat" for b in blocks)


def test_image_missing_model_is_rejected(client):
    r = client.post("/v1/images", json={"input": "no model"})
    assert r.status_code in (400, 422)


def test_image_poll_unknown_task_is_404(client):
    r = client.get("/v1/images/no-such-task")
    assert r.status_code == 404


# -- Video routes ------------------------------------------------------------ #

def test_video_seedance_create_sync_returns_completed(client, fake_provider):
    # video_sync_default is True, so the create call blocks until success.
    r = client.post("/v1/videos", json={
        "model": "fake-video-1",
        "content": [{"type": "text", "text": "a cat playing"}],
    })
    assert r.status_code == 200, r.text
    out = r.json()
    # Seedance create only returns the id per its contract.
    assert out == {"id": "task-1"}


def test_video_seedance_async_then_poll(client, fake_provider):
    # Respond-async: create returns a handle without waiting.
    r = client.post("/v1/videos", json={
        "model": "fake-video-1",
        "content": [{"type": "text", "text": "a cat playing"}],
    }, headers={"prefer": "respond-async"})
    assert r.status_code == 200, r.text
    task_id = r.json()["id"]

    # Poll until terminal.
    for _ in range(10):
        poll = client.get(f"/v1/videos/{task_id}")
        assert poll.status_code == 200, poll.text
        if poll.json()["status"] == "succeeded":
            break
    body = poll.json()
    assert body["status"] == "succeeded"
    assert body["id"] == task_id
    # The video url rides the content envelope.
    assert body["content"]["video_url"] == "https://example.test/out.mp4"


# -- Music routes (Gemini Lyria 3 shape) ------------------------------------- #


def test_music_create_sync_returns_just_id(client, fake_provider):
    # music_sync_default defaults to True, so create blocks until success and
    # returns only the Lyria interaction id.
    r = client.post("/v1/music", json={"model": "fake-music-1", "input": "an upbeat pop song"})
    assert r.status_code == 200, r.text
    assert r.json() == {"id": "music-1"}
    assert len(fake_provider.music_calls) == 1
    assert fake_provider.music_calls[0].prompt() == "an upbeat pop song"


def test_music_async_then_poll_returns_steps_content(client, fake_provider):
    # Respond-async: create returns immediately without waiting.
    r = client.post("/v1/music", json={"model": "fake-music-1", "input": "a sad ballad"},
                    headers={"prefer": "respond-async"})
    assert r.status_code == 200, r.text
    task_id = r.json()["id"]

    # Poll until terminal.
    for _ in range(10):
        poll = client.get(f"/v1/music/{task_id}")
        assert poll.status_code == 200, poll.text
        if poll.json()["status"] == "succeeded":
            break
    body = poll.json()
    assert body["id"] == task_id
    assert body["status"] == "succeeded"
    # The audio + lyrics ride a model_output step's content array as typed blocks.
    step = body["steps"][0]
    assert step["type"] == "model_output"
    blocks = step["content"]
    assert any(b["type"] == "audio" and b["data"] == "AAAA" for b in blocks)
    assert any(b["type"] == "text" and b["text"] == "la la la" for b in blocks)
    # Convenience accessors mirror the SDK's interaction.output_audio / output_text.
    assert body["output_audio"] == "AAAA"
    assert body["output_text"] == "la la la"


def test_music_parts_input_round_trips(client, fake_provider):
    # ``input`` as a parts array (Lyria native) is accepted and concatenates.
    r = client.post("/v1/music", json={
        "model": "fake-music-1",
        "input": [{"type": "text", "text": "verse"}, {"type": "text", "text": "chorus"}],
        "response_format": {"type": "audio"},
    })
    assert r.status_code == 200, r.text
    assert fake_provider.music_calls[0].prompt() == "verse\nchorus"
    # response_format {"type":"audio"} selects wav output.
    assert fake_provider.music_calls[0].audio_format == "wav"


def test_music_missing_model_is_rejected(client):
    r = client.post("/v1/music", json={"input": "x"})
    assert r.status_code in (400, 422)


def test_music_poll_unknown_task_is_404(client):
    # No record in the task store -> the service surfaces a not-found error.
    r = client.get("/v1/music/no-such-task")
    assert r.status_code == 404
