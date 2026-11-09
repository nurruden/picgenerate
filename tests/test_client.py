import base64
from pathlib import Path

import pytest
import requests

from generatepic.client import (
    ShyfaiClient,
    ShyfaiError,
    normalize_base_url,
    normalize_size,
)

PNG = b"\x89PNG\r\n\x1a\n" + b"\x00" * 16


class FakeResponse:
    def __init__(self, status_code, payload=None, content=b"", headers=None):
        self.status_code = status_code
        self._payload = payload
        self.content = content
        self.headers = headers or {}

    def json(self):
        if self._payload is None:
            raise ValueError("not json")
        return self._payload


class FakeSession:
    def __init__(self, responses):
        self.responses = list(responses)
        self.calls = []

    def request(self, method, url, **kwargs):
        self.calls.append({"method": method, "url": url, **kwargs})
        if not self.responses:
            raise AssertionError(f"unexpected request {method} {url}")
        response = self.responses.pop(0)
        if isinstance(response, Exception):
            raise response
        return response


def client(session, **kwargs):
    return ShyfaiClient(
        "https://api.example.com/v1/",
        "sk-test",
        upload_base_url="https://uploads.example.com",
        session=session,
        poll_interval=0,
        poll_deadline=5,
        **kwargs,
    )


def test_normalize_base_url_strips_v1():
    assert normalize_base_url("https://api.example.com/v1/") == "https://api.example.com"


def test_normalize_size_accepts_pixels_ratio_and_auto():
    assert normalize_size("1536×2048") == "1536x2048"
    assert normalize_size("3:4") == "3:4"
    assert normalize_size("auto") == "auto"
    with pytest.raises(ShyfaiError):
        normalize_size("5000x100")


def test_text_to_image_posts_generations_and_saves_url(tmp_path):
    session = FakeSession(
        [
            FakeResponse(
                200,
                {
                    "request_id": "req-1",
                    "status": "completed",
                    "data": [{"url": "https://cdn.example.com/cat"}],
                },
            ),
            FakeResponse(200, content=PNG, headers={"Content-Type": "image/png"}),
        ]
    )
    result = client(session).text_to_image("一只橘猫", output=tmp_path / "cat.png", idempotency_key="order-1")

    submit = session.calls[0]
    assert submit["method"] == "POST"
    assert submit["url"] == "https://api.example.com/v1/images/generations"
    assert submit["json"]["prompt"] == "一只橘猫"
    assert submit["json"]["model"] == "gpt-image-2"
    assert submit["json"]["size"] == "1024x1024"
    assert submit["headers"]["Authorization"] == "Bearer sk-test"
    assert submit["headers"]["Idempotency-Key"] == "order-1"
    assert result.request_id == "req-1"
    assert result.images[0].path.read_bytes() == PNG


def test_text_to_image_decodes_base64(tmp_path):
    encoded = base64.b64encode(PNG).decode()
    session = FakeSession(
        [FakeResponse(200, {"status": "completed", "data": [{"b64_json": encoded}]})]
    )
    result = client(session).text_to_image("海边", response_format="b64_json", output=tmp_path / "sea.png")
    assert result.images[0].path.read_bytes() == PNG
    assert session.calls[0]["json"]["response_format"] == "b64_json"


def test_connection_drop_retries_the_same_task(tmp_path):
    session = FakeSession(
        [
            requests.ConnectionError("reset"),
            FakeResponse(200, {"status": "completed", "request_id": "kept-1", "data": [{"b64_json": base64.b64encode(PNG).decode()}]}),
        ]
    )
    result = client(session).text_to_image("一只橘猫", output=tmp_path / "cat.png", response_format="b64_json")
    assert result.request_id == "kept-1"
    assert len(session.calls) == 2
    assert session.calls[0]["headers"]["Idempotency-Key"] == session.calls[1]["headers"]["Idempotency-Key"]


def test_image_to_image_uploads_then_edits(tmp_path):
    source = tmp_path / "source.png"
    source.write_bytes(PNG)
    session = FakeSession(
        [
            FakeResponse(200, {"state": "ready", "url": "https://uploads.example.com/v1/images/uploads/a"}),
            FakeResponse(
                200,
                {"request_id": "edit-1", "status": "completed", "data": [{"b64_json": base64.b64encode(PNG).decode()}]},
            ),
        ]
    )
    result = client(session).image_to_image(
        "保留人物，换成雪山背景",
        [source],
        output=tmp_path / "out.png",
        response_format="b64_json",
    )

    assert session.calls[0]["url"] == "https://uploads.example.com/v1/images/uploads"
    assert "Authorization" not in session.calls[0].get("headers", {})
    edit = session.calls[1]
    assert edit["url"] == "https://api.example.com/v1/images/edits"
    assert edit["json"]["images"] == [{"image_url": "https://uploads.example.com/v1/images/uploads/a"}]
    assert "保留人物" in edit["json"]["prompt"]
    assert result.images[0].path is not None


def test_image_to_image_falls_back_to_multipart_when_upload_missing(tmp_path):
    source = tmp_path / "source.png"
    source.write_bytes(PNG)
    session = FakeSession(
        [
            FakeResponse(404, {"error": {"message": "not found"}}),
            FakeResponse(200, {"status": "completed", "data": [{"b64_json": base64.b64encode(PNG).decode()}]}),
        ]
    )
    client(session).image_to_image("改成水彩", [source], output=tmp_path / "out.png", response_format="b64_json")

    edit = session.calls[1]
    assert edit["url"].endswith("/v1/images/edits")
    assert "image[]" in [item[0] for item in edit["files"]]
    assert edit["data"]["prompt"] == "改成水彩"
    assert "json" not in edit


def test_image_to_image_falls_back_to_multipart_when_upload_gateway_fails(tmp_path):
    source = tmp_path / "source.png"
    source.write_bytes(PNG)
    session = FakeSession(
        [
            FakeResponse(502, None, content=b"<html><h1>502 Bad Gateway</h1></html>"),
            FakeResponse(200, {"status": "completed", "data": [{"b64_json": base64.b64encode(PNG).decode()}]}),
        ]
    )
    client(session).image_to_image("改成水彩", [source], output=tmp_path / "out.png", response_format="b64_json")

    edit = session.calls[1]
    assert edit["url"].endswith("/v1/images/edits")
    assert "image[]" in [item[0] for item in edit["files"]]


def test_image_to_image_uses_remote_url_without_upload():
    session = FakeSession(
        [FakeResponse(200, {"status": "completed", "data": [{"url": "https://cdn.example.com/out.png"}]})]
    )
    result = client(session).image_to_image(
        "保留构图",
        ["https://uploads.example.com/source.png"],
        output=None,
        response_format="url",
    )
    assert len(session.calls) == 1
    assert session.calls[0]["json"]["images"][0]["image_url"].endswith("source.png")
    assert result.images[0].url.endswith("out.png")
    assert result.images[0].path is None


def test_async_acceptance_polls_status_without_api_key(tmp_path):
    session = FakeSession(
        [
            FakeResponse(
                202,
                {
                    "request_id": "task-9",
                    "status": "queued",
                    "status_url": "https://api.example.com/v1/images/tasks/task-9",
                },
            ),
            FakeResponse(200, {"request_id": "task-9", "status": "running"}),
            FakeResponse(
                200,
                {
                    "request_id": "task-9",
                    "status": "completed",
                    "data": [{"b64_json": base64.b64encode(PNG).decode()}],
                },
            ),
        ]
    )
    result = client(session).text_to_image(
        "白瓷花瓶",
        output=tmp_path / "vase.png",
        response_format="b64_json",
        callback_url="https://app.example.com/hook",
    )
    assert session.calls[1]["method"] == "GET"
    assert session.calls[1]["url"].endswith("/tasks/task-9")
    assert "headers" not in session.calls[1]
    assert session.calls[0]["json"]["callback_url"] == "https://app.example.com/hook"
    assert result.status == "completed"
    assert result.request_id == "task-9"


def test_api_error_includes_code_and_request_id():
    session = FakeSession(
        [
            FakeResponse(
                400,
                {"error": {"message": "图片格式无效", "code": "invalid_image"}, "request_id": "bad-1"},
            )
        ]
    )
    with pytest.raises(ShyfaiError) as raised:
        client(session).text_to_image("测试")
    assert raised.value.status_code == 400
    assert raised.value.code == "invalid_image"
    assert raised.value.request_id == "bad-1"
    assert "图片格式无效" in str(raised.value)


def test_failed_task_does_not_look_like_success():
    session = FakeSession(
        [FakeResponse(200, {"request_id": "nope", "status": "failed", "error": {"message": "内容策略拒绝", "code": "content_policy_violation"}})]
    )
    with pytest.raises(ShyfaiError) as raised:
        client(session).text_to_image("测试")
    assert raised.value.code == "content_policy_violation"


def test_from_env_reads_dotenv(tmp_path, monkeypatch):
    for key in ("SHYFAI_BASE_URL", "SHYFAI_API_KEY", "BASE_URL", "API_KEY", "SHYFAI_MODEL"):
        monkeypatch.delenv(key, raising=False)
    env_file = tmp_path / ".env"
    env_file.write_text(
        "SHYFAI_BASE_URL=https://gateway.example.com/v1\nSHYFAI_API_KEY=sk-from-file\nSHYFAI_MODEL=demo-model\n",
        encoding="utf-8",
    )
    created = ShyfaiClient.from_env(env_file)
    assert created.base_url == "https://gateway.example.com"
    assert created.api_key == "sk-from-file"
    assert created.model == "demo-model"


def test_cli_text_prints_saved_path(tmp_path, monkeypatch, capsys):
    from generatepic import cli

    destination = tmp_path / "out.png"

    class FakeClient:
        @classmethod
        def from_env(cls):
            return cls()

        def text_to_image(self, prompt, **kwargs):
            assert prompt == "一只猫"
            assert kwargs["size"] == "1024x1024"
            from generatepic.client import GeneratedImage, GenerationResult

            return GenerationResult("rid", "completed", [GeneratedImage(None, destination)])

    monkeypatch.setattr(cli, "ShyfaiClient", FakeClient)
    assert cli.main(["text", "一只猫", "-o", str(destination)]) == 0
    assert str(destination) in capsys.readouterr().out
