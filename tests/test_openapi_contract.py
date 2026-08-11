"""Assertions for the published provider-neutral REST contract."""

from __future__ import annotations

from mm_gateway.config import Settings
from mm_gateway.server.app import create_app

EXPECTED_OPERATIONS = {
    ("/health", "get"): "getHealth",
    ("/metrics", "get"): "getMetrics",
    ("/v1/models", "get"): "listModels",
    ("/v1/images", "post"): "createImage",
    ("/v1/images/{image_id}", "get"): "getImage",
    ("/v1/videos", "post"): "createVideo",
    ("/v1/videos/{video_id}", "get"): "getVideo",
    ("/v1/music", "post"): "createMusic",
    ("/v1/music/{music_id}", "get"): "getMusic",
}


def _spec() -> dict:
    return create_app(Settings()).openapi()


def _walk_schema(value, path=()):
    if isinstance(value, dict):
        yield path, value
        for key, child in value.items():
            yield from _walk_schema(child, (*path, key))
    elif isinstance(value, list):
        for index, child in enumerate(value):
            yield from _walk_schema(child, (*path, index))


def test_openapi_has_only_the_intended_rest_paths_and_operation_ids():
    spec = _spec()
    assert set(spec["paths"]) == {path for path, _ in EXPECTED_OPERATIONS}
    for (path, method), operation_id in EXPECTED_OPERATIONS.items():
        assert spec["paths"][path][method]["operationId"] == operation_id


def test_create_operations_are_202_resources_with_polling_headers():
    spec = _spec()
    for path in ("/v1/images", "/v1/videos", "/v1/music"):
        responses = spec["paths"][path]["post"]["responses"]
        assert "200" not in responses
        assert "409" in responses
        created = responses["202"]
        assert set(created["headers"]) == {
            "ETag",
            "Idempotency-Replayed",
            "Link",
            "Location",
            "Retry-After",
        }
        schema = created["content"]["application/json"]["schema"]
        assert schema["$ref"].startswith("#/components/schemas/")

        parameters = spec["paths"][path]["post"]["parameters"]
        assert any(parameter["name"] == "Idempotency-Key" for parameter in parameters)


def test_get_operations_support_conditional_polling():
    spec = _spec()
    for path in (
        "/v1/images/{image_id}",
        "/v1/videos/{video_id}",
        "/v1/music/{music_id}",
    ):
        operation = spec["paths"][path]["get"]
        assert "304" in operation["responses"]
        assert "ETag" in operation["responses"]["200"]["headers"]
        assert any(
            parameter["name"] == "If-None-Match"
            for parameter in operation["parameters"]
        )

    models = spec["paths"]["/v1/models"]["get"]
    assert "304" in models["responses"]
    assert "ETag" in models["responses"]["200"]["headers"]
    assert any(
        parameter["name"] == "If-None-Match"
        for parameter in models["parameters"]
    )


def test_validation_errors_use_shared_problem_details():
    spec = _spec()
    for path, method in EXPECTED_OPERATIONS:
        if path in {"/health", "/metrics"}:
            continue
        response = spec["paths"][path][method]["responses"]["422"]
        assert set(response["content"]) == {"application/problem+json"}
        schema = response["content"][
            "application/problem+json"
        ]["schema"]
        assert schema == {"$ref": "#/components/schemas/ProblemDetail"}


def test_public_parameter_schemas_are_strict_and_provider_neutral():
    schemas = _spec()["components"]["schemas"]
    provider_fields = {
        "n",
        "response_format",
        "output_format",
        "output_compression",
        "generate_audio",
        "camera_fixed",
        "prompt_extend",
        "return_last_frame",
        "motion_bucket_id",
        "cfg_scale",
        "service_tier",
        "audio_quality",
        "is_instrumental",
        "lyrics_optimizer",
        "voice_id",
        "style_weight",
        "weirdness_constraint",
        "audio_weight",
        "size",
        "aspect_ratio",
        "resolution",
    }
    for name in ("ImageParameters", "VideoParameters", "MusicParameters"):
        schema = schemas[name]
        assert schema["additionalProperties"] is False
        assert provider_fields.isdisjoint(schema["properties"])


def test_requests_are_strict_but_responses_allow_additive_fields():
    schemas = _spec()["components"]["schemas"]
    for name in (
        "ImageRequest",
        "VideoRequest",
        "MusicRequest",
        "ImageParameters",
        "VideoParameters",
        "MusicParameters",
        "Dimensions",
    ):
        assert schemas[name]["additionalProperties"] is False
    for name in (
        "ImageTaskResponse",
        "VideoTaskResponse",
        "MusicTaskResponse",
        "ProblemDetail",
    ):
        assert schemas[name]["additionalProperties"] is True


def test_inputs_have_one_canonical_non_empty_array_shape():
    schemas = _spec()["components"]["schemas"]
    for name in ("ImageRequest", "VideoRequest", "MusicRequest"):
        schema = schemas[name]
        input_schema = schema["properties"]["input"]
        assert input_schema["type"] == "array"
        assert input_schema["minItems"] == 1
        assert "anyOf" not in input_schema
        assert isinstance(schema["example"]["input"], list)


def test_fields_do_not_publish_alternate_non_null_shapes():
    allowed_one_of = {
        ("components", "schemas", name, "properties", "input", "items")
        for name in ("ImageRequest", "VideoRequest", "MusicRequest")
    }
    seen_one_of = set()

    for path, schema in _walk_schema(_spec()):
        if "anyOf" in schema:
            non_null = [
                item for item in schema["anyOf"] if item.get("type") != "null"
            ]
            assert len(non_null) <= 1, ".".join(map(str, path))
        if "oneOf" in schema:
            seen_one_of.add(path)

    assert seen_one_of == allowed_one_of


def test_media_parts_and_outputs_use_one_uri_shape():
    schemas = _spec()["components"]["schemas"]
    inputs = (
        "ImageInput",
        "VideoImageInput",
        "VideoAudioInput",
        "VideoInput",
        "MusicImageInput",
        "MusicAudioInput",
    )
    for name in inputs:
        schema = schemas[name]
        assert "uri" in schema["properties"]
        assert "uri" in schema["required"]
        assert {"url", "data", "mime_type"}.isdisjoint(schema["properties"])

    for name in ("ImageOutput", "VideoOutput", "MusicOutput"):
        schema = schemas[name]
        assert "uri" in schema["properties"]
        assert "uri" in schema["required"]
        assert {"url", "data", "cover_url"}.isdisjoint(schema["properties"])


def test_visual_parameters_use_one_dimensions_shape():
    schemas = _spec()["components"]["schemas"]
    dimensions = schemas["Dimensions"]
    assert dimensions["additionalProperties"] is False
    assert set(dimensions["properties"]) == {"width", "height"}
    assert set(dimensions["required"]) == {"width", "height"}

    legacy = {"size", "width", "height", "aspect_ratio", "resolution"}
    for name in ("ImageParameters", "VideoParameters"):
        properties = schemas[name]["properties"]
        assert "dimensions" in properties
        assert legacy.isdisjoint(properties)


def test_unused_fastapi_validation_envelopes_are_not_published():
    schemas = _spec()["components"]["schemas"]
    assert "HTTPValidationError" not in schemas
    assert "ValidationError" not in schemas


def test_model_catalogue_does_not_expose_backend_details():
    properties = _spec()["components"]["schemas"]["ModelEntry"]["properties"]
    assert set(properties) == {"id", "object", "modality"}


def test_routing_and_errors_do_not_expose_backend_identity():
    schemas = _spec()["components"]["schemas"]
    routing = schemas["RoutingDirective"]
    assert set(routing["properties"]) == {"profile"}
    assert routing["required"] == ["profile"]
    problem_properties = schemas["ProblemDetail"]["properties"]
    assert "provider" not in problem_properties
    assert {
        "type",
        "title",
        "status",
        "detail",
        "instance",
        "code",
        "request_id",
        "errors",
    } == set(problem_properties)


def test_duration_units_are_explicit_in_public_parameters():
    schemas = _spec()["components"]["schemas"]
    for name in ("VideoParameters", "MusicParameters"):
        properties = schemas[name]["properties"]
        assert "duration_seconds" in properties
        assert "duration" not in properties
