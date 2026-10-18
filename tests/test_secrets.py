"""Credential protection across startup, console writes and process restarts."""

import stat
import sys
from concurrent.futures import ThreadPoolExecutor

import pytest
import yaml
from cryptography.fernet import Fernet
from fastapi.testclient import TestClient

from mm_gateway.config import BackendConfig, KeyConfig, Settings
from mm_gateway.management import config_from_settings
from mm_gateway.registry import _PROVIDER_CLASSES
from mm_gateway.secrets import ENCRYPTED_PREFIX, SecretError, SecretStore
from mm_gateway.server.app import create_app
from tests.test_management import ADMIN, ROOT, USER, config


@pytest.fixture(autouse=True)
def isolated_secrets(monkeypatch):
    monkeypatch.delenv("MM_GATEWAY_SECRET_KEY_FILE", raising=False)
    monkeypatch.setitem(_PROVIDER_CLASSES, "fake", "FakeProvider")
    monkeypatch.setitem(sys.modules, "mm_gateway.providers.fake", sys.modules["tests.conftest"])


def test_startup_encrypts_yaml_credentials_and_restarts_without_changing_them(tmp_path, monkeypatch):
    path = tmp_path / "gateway.yaml"
    path.write_text("""
management:
  api_key: management-test
backends:
  - name: provider
    type: fake
    api_key: top-level-secret
    base_url: https://url-user:url-password@example.test/v1?api_key=url-query-secret
    extra:
      credentials_json:
        type: service_account
        client_email: example@example.test
        private_key: |
          -----BEGIN PRIVATE KEY-----
          private-key-material
          -----END PRIVATE KEY-----
      auth: {username: auth-user, password: auth-password}
      nested: {refresh_token: refresh-secret, client_secret: client-secret}
      headers: {X-Custom: custom-header-secret}
      image_model: fake-image-1
    credentials:
      - shorthand-secret
      - {id: named, key: named-account-secret, extra: {password: account-password}}
keys:
  - {id: user, key: generation-test}
proxies:
  - base_url: https://proxy.test
    headers: {X-Custom: static-header-secret}
    accounts:
      - {id: primary, headers: {Authorization: 'Bearer proxy-secret'}}
outbound_proxy: http://proxy-user:proxy-password@proxy.test:8080
""", encoding="utf-8")
    monkeypatch.setenv("MM_GATEWAY_CONFIG", str(path))
    app = create_app()
    backend = app.state.settings.backends[0]
    assert backend.api_key == "top-level-secret"
    assert backend.accounts()[0][1] == "shorthand-secret"
    assert backend.accounts()[1][1] == "named-account-secret"
    assert dict(app.state.registry.accounts_of("provider"))["account-0"].backend.api_key == "shorthand-secret"
    assert backend.extra["credentials_json"]["private_key"].endswith("-----END PRIVATE KEY-----\n")
    saved = path.read_bytes()
    for secret in ("management-test", "top-level-secret", "url-user", "url-password", "url-query-secret",
                   "private-key-material", "auth-user", "auth-password", "refresh-secret", "client-secret",
                   "custom-header-secret", "shorthand-secret", "named-account-secret", "account-password",
                   "generation-test", "static-header-secret", "proxy-secret", "proxy-user", "proxy-password"):
        assert secret.encode() not in saved
    assert ENCRYPTED_PREFIX.encode() in saved
    assert stat.S_IMODE(path.stat().st_mode) == 0o600
    key_file = tmp_path / ".mm-gateway.key"
    assert stat.S_IMODE(key_file.stat().st_mode) == 0o600
    key = key_file.read_bytes()
    restarted = create_app()
    assert restarted.state.settings == app.state.settings
    assert path.read_bytes() == saved
    assert key_file.read_bytes() == key
    # Mixing a newly pasted plaintext key into an encrypted config migrates it too.
    document = yaml.safe_load(saved)
    document["backends"][0]["api_key"] = "rotated-plaintext"
    path.write_text(yaml.safe_dump(document))
    assert Settings.from_file(path).backends[0].api_key == "rotated-plaintext"
    assert "rotated-plaintext" not in path.read_text()


def test_yaml_environment_references_remain_dynamic_and_defaults_are_encrypted(tmp_path, monkeypatch):
    path = tmp_path / "gateway.yaml"
    path.write_text("""
backends:
  - {name: dynamic, type: fake, api_key: ${DYNAMIC_PROVIDER_KEY}}
  - name: fallback
    type: fake
    api_key: ${FALLBACK_PROVIDER_KEY:inline-default-secret}
    extra:
      credentials_json: {private_key: '${DYNAMIC_PROVIDER_KEY}'}
proxies:
  - base_url: https://example.test
    accounts:
      - id: first
        headers:
          Authorization: Bearer ${DYNAMIC_PROVIDER_KEY}
""")
    monkeypatch.setenv("DYNAMIC_PROVIDER_KEY", "first-environment-secret")
    monkeypatch.delenv("FALLBACK_PROVIDER_KEY", raising=False)
    first = Settings.from_file(path)
    assert first.backends[0].api_key == "first-environment-secret"
    assert first.backends[1].api_key == "inline-default-secret"
    saved = path.read_text()
    assert "${DYNAMIC_PROVIDER_KEY}" in saved
    assert "inline-default-secret" not in saved
    assert "first-environment-secret" not in saved
    monkeypatch.setenv("DYNAMIC_PROVIDER_KEY", "second-environment-secret")
    monkeypatch.setenv("FALLBACK_PROVIDER_KEY", "override-environment-secret")
    second = Settings.from_file(path)
    assert second.backends[0].api_key == "second-environment-secret"
    assert second.backends[1].api_key == "override-environment-secret"
    assert second.backends[1].extra["credentials_json"]["private_key"] == "second-environment-secret"
    assert second.proxies[0].enumerate_accounts()[0][1]["Authorization"] == "Bearer second-environment-secret"
    assert path.read_text() == saved


def test_environment_only_yaml_needs_no_writable_file_or_key(tmp_path, monkeypatch):
    path = tmp_path / "gateway.yaml"
    text = "backends:\n- {name: test, type: fake, api_key: '${DYNAMIC_PROVIDER_KEY}'}\n"
    path.write_text(text)
    monkeypatch.setenv("DYNAMIC_PROVIDER_KEY", "environment-secret")
    def fail_write(*args):
        pytest.fail("An environment reference must not trigger a write")
    monkeypatch.setattr("mm_gateway.secrets.atomic_write", fail_write)
    assert Settings.from_file(path).backends[0].api_key == "environment-secret"
    assert path.read_text() == text
    assert not (tmp_path / ".mm-gateway.key").exists()


def test_yaml_migration_preserves_unrelated_comments_and_block_layout(tmp_path):
    path = tmp_path / "gateway.yaml"
    path.write_text("""# Keep the operator's notes.
server: {port: 8123}  # Keep the port.
backends:
  - name: provider
    type: fake
    api_key: |-
      multi-line
      secret-value
    tags: [keep]  # Keep the tags.
    extra:
      password: simple-secret  # Keep this comment.
      region: east
""")
    first = Settings.from_file(path)
    saved = path.read_text()
    assert "multi-line" not in saved and "simple-secret" not in saved
    for comment in ("# Keep the operator's notes.", "# Keep the port.", "# Keep the tags.", "# Keep this comment."):
        assert comment in saved
    assert "    tags: [keep]" in saved
    assert first == Settings.from_file(path)


def test_yaml_anchors_and_explicit_types_remain_usable_after_migration(tmp_path):
    path = tmp_path / "gateway.yaml"
    path.write_text("""
backends:
  - name: first
    type: fake
    api_key: &shared !!str 123456
    extra: &options {password: shared-password, retries: 3}
  - name: second
    type: fake
    api_key: *shared
    extra: *options
""")
    first = Settings.from_file(path)
    assert "shared-password" not in path.read_text() and "123456" not in path.read_text()
    assert first.backends[0].api_key == "123456"
    assert first.backends[1].extra == {"password": "shared-password", "retries": 3}
    assert first == Settings.from_file(path)


@pytest.mark.parametrize("shared", ["shared-secret-value", "literal-prefix-${SHARED_ENV_KEY}"])
def test_yaml_shared_credentials_are_also_encrypted_at_the_anchor_definition(tmp_path, monkeypatch, shared):
    path = tmp_path / "gateway.yaml"
    path.write_text("""
shared: &shared shared-secret-value
options: &options {value: &password shared-password}
backends:
  - name: first
    type: fake
    api_key: *shared
    extra:
      auth: *options
      other_option: *password
""".replace("shared-secret-value", shared))
    monkeypatch.setenv("SHARED_ENV_KEY", "environment-secret")
    first = Settings.from_file(path)
    assert "shared-secret-value" not in path.read_text() and "shared-password" not in path.read_text()
    assert "literal-prefix" not in path.read_text()
    assert first.backends[0].api_key == shared.replace("${SHARED_ENV_KEY}", "environment-secret")
    assert first.backends[0].extra["other_option"] == "shared-password"
    assert first == Settings.from_file(path)


def test_partially_encrypted_sensitive_objects_do_not_double_encrypt(tmp_path):
    path = tmp_path / "gateway.yaml"
    encrypted = SecretStore(path).encrypt({"password": "nested-secret"})["password"]
    path.write_text(yaml.safe_dump({"backends": [{"name": "first", "type": "fake", "api_key": "top-secret",
                                                 "extra": {"auth": {"password": encrypted}}}]}))
    first = Settings.from_file(path)
    assert first.backends[0].extra["auth"]["password"] == "nested-secret"
    assert first == Settings.from_file(path)


def test_url_query_auth_and_mixed_environment_templates_are_encrypted(tmp_path, monkeypatch):
    path = tmp_path / "gateway.yaml"
    path.write_text("""
backends:
  - name: provider
    type: fake
    api_key: literal-prefix-${DYNAMIC_PROVIDER_KEY}
    base_url: https://example.test?key=query-key-secret
    extra:
      outbound_proxy: http://${PROXY_USER}:literal-proxy-password@proxy.test
      signed_url: https://example.test?sig=signed-secret
      default_headers: {X-Custom: custom-header-secret}
""")
    monkeypatch.setenv("DYNAMIC_PROVIDER_KEY", "environment-secret")
    monkeypatch.setenv("PROXY_USER", "proxy-user")
    first = Settings.from_file(path)
    assert first.backends[0].api_key == "literal-prefix-environment-secret"
    for secret in ("literal-prefix", "query-key-secret", "literal-proxy-password", "signed-secret", "custom-header-secret"):
        assert secret not in path.read_text()
    assert first == Settings.from_file(path)


def test_existing_plaintext_management_overlay_is_migrated_at_startup(tmp_path):
    path = tmp_path / "management.json"
    settings = Settings(management_api_key="management-test", management_config_path=str(path),
                        keys=[KeyConfig(id="user", key="generation-test")],
                        backends=[BackendConfig(name="provider", type="fake", api_key="legacy-secret",
                                                extra={"password": "legacy-password"})])
    path.write_text(config_from_settings(settings).model_dump_json())
    app = create_app(settings)
    assert app.state.registry.get("provider").backend.api_key == "legacy-secret"
    saved = path.read_bytes()
    assert b"legacy-secret" not in saved and b"legacy-password" not in saved and b"generation-test" not in saved
    assert stat.S_IMODE(path.stat().st_mode) == 0o600
    assert create_app(settings).state.settings == app.state.settings
    assert path.read_bytes() == saved


def test_console_saves_encrypt_secrets_and_redacted_edits_survive_restart(tmp_path):
    path = tmp_path / "management.json"
    settings = Settings(management_api_key="management-test", management_config_path=str(path),
                        keys=[KeyConfig(id="user", key="generation-test")])
    app = create_app(settings)
    with TestClient(app) as client:
        _, headers = config(client)
        response = client.put(ROOT + "/backends/provider", headers=headers, json={
            "name": "provider", "type": "fake", "api_key": "web-top-secret",
            "extra": {"credentials_json": {"private_key": "web-private-key"},
                      "auth": {"username": "web-username", "password": "web-password"}},
            "credentials": [{"id": "first", "api_key": "web-first-secret"},
                            {"id": "second", "api_key": "web-second-secret"}],
        })
        assert response.status_code == 200, response.text
        saved = path.read_text()
        for secret in ("web-top-secret", "web-private-key", "web-username", "web-password",
                       "web-first-secret", "web-second-secret", "generation-test"):
            assert secret not in saved and secret not in response.text
        assert client.get("/v1/models", headers=USER).status_code == 200
        body, headers = config(client)
        body["config"]["backends"][0]["credentials"].reverse()
        body["config"]["backends"][0]["tags"] = ["edited"]
        assert client.put(ROOT + "/config", headers=headers, json=body["config"]).status_code == 200
    restarted = create_app(settings)
    stored = restarted.state.settings.backends[0]
    assert stored.api_key == "web-top-secret"
    assert stored.extra["auth"] == {"username": "web-username", "password": "web-password"}
    assert stored.credentials[0]["api_key"] == "web-second-secret"
    assert stored.credentials[1]["api_key"] == "web-first-secret"
    assert dict(restarted.state.registry.accounts_of("provider"))["second"].backend.api_key == "web-second-secret"


@pytest.mark.parametrize("damage", ["missing-key", "wrong-key", "invalid-key", "tampered", "unknown-version"])
@pytest.mark.parametrize("file_format", ["yaml", "json"])
def test_startup_rejects_unreadable_ciphertext_without_rewriting_it(tmp_path, damage, file_format):
    path = tmp_path / ("gateway.yaml" if file_format == "yaml" else "management.json")
    if file_format == "yaml":
        path.write_text("backends:\n- {name: test, type: fake, api_key: original-secret}\n")
        Settings.from_file(path)
        load = lambda: Settings.from_file(path)
    else:
        settings = Settings(management_config_path=str(path),
                            backends=[BackendConfig(name="test", type="fake", api_key="original-secret")])
        path.write_text(config_from_settings(settings).model_dump_json())
        create_app(settings)
        load = lambda: create_app(settings)
    key_file = tmp_path / ".mm-gateway.key"
    if damage == "missing-key":
        key_file.unlink()
    elif damage == "wrong-key":
        key_file.write_bytes(Fernet.generate_key())
    elif damage == "invalid-key":
        key_file.write_bytes(b"invalid-key")
    elif damage == "tampered":
        path.write_text(path.read_text().replace(ENCRYPTED_PREFIX, ENCRYPTED_PREFIX + "invalid"))
    else:
        path.write_text(path.read_text().replace(ENCRYPTED_PREFIX, "enc:v2:"))
    before = path.read_bytes()
    with pytest.raises(SecretError) as error:
        load()
    assert "original-secret" not in str(error.value)
    assert path.read_bytes() == before
    if damage == "missing-key":
        assert not key_file.exists()


def test_failed_yaml_migration_does_not_start_or_replace_plaintext(tmp_path, monkeypatch):
    path = tmp_path / "gateway.yaml"
    original = "backends:\n- {name: test, type: fake, api_key: original-secret}\n"
    path.write_text(original)
    def fail_replace(*args):
        raise OSError("read-only directory")
    monkeypatch.setattr("mm_gateway.secrets.os.replace", fail_replace)
    with pytest.raises(OSError):
        Settings.from_file(path)
    assert path.read_text() == original
    assert sorted(p.name for p in tmp_path.iterdir()) == [".mm-gateway.key", "gateway.yaml"]


@pytest.mark.parametrize("error", [OSError("private-key-material"), SecretError("invalid key file")])
def test_failed_encryption_during_web_save_leaves_disk_runtime_and_revision_unchanged(tmp_path, monkeypatch, error):
    path = tmp_path / "management.json"
    settings = Settings(management_api_key="management-test", management_config_path=str(path),
                        keys=[KeyConfig(id="user", key="generation-test")])
    app = create_app(settings)
    with TestClient(app) as client:
        body, headers = config(client)
        assert client.put(ROOT + "/config", headers=headers, json=body["config"]).status_code == 200
        body, headers = config(client)
        original = path.read_bytes()
        def fail_encrypt(*args, **kwargs):
            raise error
        monkeypatch.setattr(SecretStore, "_fernet", fail_encrypt)
        response = client.put(ROOT + "/backends/new", headers=headers,
                              json={"name": "new", "type": "fake", "api_key": "new-private-secret"})
        assert response.status_code == 503
        assert "private-key-material" not in response.text and "new-private-secret" not in response.text
        assert path.read_bytes() == original
        assert app.state.settings.backends == []
        assert config(client)[0]["revision"] == body["revision"]


def test_custom_persistent_key_location_and_concurrent_first_writes(tmp_path, monkeypatch):
    key_path = tmp_path / "keys" / "master.key"
    monkeypatch.setenv("MM_GATEWAY_SECRET_KEY_FILE", str(key_path))
    paths = [tmp_path / str(index) / "managed.json" for index in range(8)]
    def encrypt(path):
        return SecretStore(path).encrypt({"api_key": "shared-secret"})
    with ThreadPoolExecutor(max_workers=8) as workers:
        encrypted = list(workers.map(encrypt, paths))
    assert stat.S_IMODE(key_path.stat().st_mode) == 0o600
    for path, value in zip(paths, encrypted):
        assert SecretStore(path).decrypt(value) == {"api_key": "shared-secret"}
    assert list(key_path.parent.iterdir()) == [key_path]


def test_gateway_starts_and_web_management_works_without_gateway_api_key(tmp_path, monkeypatch):
    monkeypatch.delenv("GATEWAY_API_KEY", raising=False)
    monkeypatch.setenv("MANAGEMENT_API_KEY", "management-test")
    monkeypatch.setenv("MANAGEMENT_CONFIG_PATH", str(tmp_path / "management.json"))
    settings = Settings._from_legacy_env()
    assert settings.keys[0].key == ""
    app = create_app(settings)
    with TestClient(app) as client:
        assert client.get("/health").status_code == 200
        assert client.get("/v1/models").status_code == 200
        assert client.get(ROOT + "/config").status_code == 401
        _, headers = config(client)
        response = client.put(ROOT + "/backends/web-managed", headers=headers,
                              json={"name": "web-managed", "type": "fake", "api_key": "web-provider-secret"})
        assert response.status_code == 200, response.text
        assert any(model["id"] == "fake-image-1" for model in client.get("/v1/models").json()["data"])
        # Replace the initial open key with a generation key created through management.
        body, headers = config(client)
        body["config"]["keys"] = [{"id": "managed", "key": "web-generation-token"}]
        assert client.put(ROOT + "/config", headers=headers, json=body["config"]).status_code == 200
        assert client.get("/v1/models").status_code == 401
        assert client.get("/v1/models", headers={"Authorization": "Bearer web-generation-token"}).status_code == 200
    with TestClient(create_app(settings)) as client:
        assert client.get("/v1/models").status_code == 401
        assert client.get("/v1/models", headers={"Authorization": "Bearer web-generation-token"}).status_code == 200
