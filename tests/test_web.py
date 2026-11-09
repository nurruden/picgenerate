import time
from io import BytesIO
from pathlib import Path

import pytest

from generatepic.client import GeneratedImage, GenerationResult, ShyfaiError
from generatepic.web import client_from_dotenv, create_app

PNG = b"\x89PNG\r\n\x1a\n" + b"\x00" * 16


def _fake_images(kwargs):
    path = Path(kwargs["output"])
    count = kwargs.get("n") or 1
    images = []
    for index in range(count):
        target = path if count == 1 else path.with_name(f"{path.stem}-{index}{path.suffix}")
        target.write_bytes(PNG)
        images.append(GeneratedImage(None, target))
    return images


class FakeClient:
    def __init__(self):
        self.model = "gpt-image-2"
        self.calls = []

    def text_to_image(self, prompt, **kwargs):
        self.calls.append(("text", prompt, kwargs))
        return GenerationResult("text-1", "completed", _fake_images(kwargs))

    def image_to_image(self, prompt, images, **kwargs):
        self.calls.append(("edit", prompt, [Path(image).read_bytes() for image in images], kwargs))
        return GenerationResult("edit-1", "completed", _fake_images(kwargs))


@pytest.fixture
def app(tmp_path):
    fake = FakeClient()
    application = create_app(client_factory=lambda: fake, results_dir=tmp_path)
    application.config["FAKE"] = fake
    application.config["TESTING"] = True
    return application


def test_dev_data_stays_in_the_project():
    from generatepic.web import bundle_root, data_root

    project = Path(__file__).resolve().parents[1]
    assert data_root() == project
    assert (bundle_root() / "generatepic" / "templates" / "index.html").is_file()


def test_page_has_upload_control(app):
    response = app.test_client().get("/")
    assert response.status_code == 200
    html = response.get_data(as_text=True)
    assert 'type="file"' in html
    assert "图生图" in html
    assert "历史记录" in html
    assert "模型绘制" in html
    assert "已进行" in html
    assert "sk-" not in html


def wait_job(client, response):
    assert response.status_code == 202, response.get_data(as_text=True)
    payload = response.get_json()
    current = payload
    for _ in range(50):
        current = client.get("/api/jobs/" + payload["job_id"]).get_json()
        if current["status"] in {"completed", "failed"}:
            return current
        time.sleep(0.02)
    raise AssertionError(current)


def test_config_rejects_placeholder_env(tmp_path):
    env_file = tmp_path / ".env"
    env_file.write_text(
        "SHYFAI_BASE_URL=https://api.example.com\nSHYFAI_API_KEY=sk-your-key\n",
        encoding="utf-8",
    )
    application = create_app(results_dir=tmp_path / "results", env_file=env_file)
    payload = application.test_client().get("/api/config").get_json()
    assert payload["ready"] is False
    assert "示例" in payload["error"]


def test_config_reads_saved_env(tmp_path):
    env_file = tmp_path / ".env"
    env_file.write_text(
        "SHYFAI_BASE_URL=https://gateway.example.com/v1\nSHYFAI_API_KEY=sk-live\nSHYFAI_MODEL=demo\n",
        encoding="utf-8",
    )
    client = client_from_dotenv(env_file)
    assert client.base_url == "https://gateway.example.com"
    assert client.model == "demo"
    application = create_app(results_dir=tmp_path / "results", env_file=env_file)
    payload = application.test_client().get("/api/config").get_json()
    assert payload == {"ready": True, "model": "demo"}


def test_edit_requires_upload(app):
    response = app.test_client().post(
        "/api/generate",
        data={"mode": "edit", "prompt": "保留人物", "size": "1024x1024"},
    )
    assert response.status_code == 400
    assert "上传" in response.get_json()["error"]


def test_edit_uploads_image_and_returns_local_result(app):
    response = app.test_client().post(
        "/api/generate",
        data={
            "mode": "edit",
            "prompt": "保留人物，换成雪山",
            "size": "896x1152",
            "images": (BytesIO(PNG), "portrait.png"),
        },
        content_type="multipart/form-data",
    )
    payload = wait_job(app.test_client(), response)
    assert payload["request_id"] == "edit-1"
    assert payload["stage"] == "completed"
    assert payload["images"][0]["url"].startswith("/results/")
    kind, prompt, blobs, kwargs = app.config["FAKE"].calls[0]
    assert kind == "edit"
    assert prompt == "保留人物，换成雪山"
    assert blobs == [PNG]
    assert kwargs["size"] == "896x1152"
    image = app.test_client().get(payload["images"][0]["url"])
    assert image.status_code == 200
    assert image.data == PNG


def test_original_size_uses_the_uploaded_image(app):
    from PIL import Image

    source = BytesIO()
    Image.new("RGB", (640, 480), (20, 40, 60)).save(source, format="PNG")
    source.seek(0)
    response = app.test_client().post(
        "/api/generate",
        data={
            "mode": "edit",
            "prompt": "保留人物",
            "size": "original",
            "images": (source, "photo.png"),
        },
        content_type="multipart/form-data",
    )
    wait_job(app.test_client(), response)
    assert app.config["FAKE"].calls[0][3]["size"] == "640x480"


def test_each_reference_image_is_edited_on_its_own(app):
    from PIL import Image

    first = BytesIO()
    second = BytesIO()
    Image.new("RGB", (640, 480), (20, 40, 60)).save(first, format="PNG")
    Image.new("RGB", (800, 800), (60, 40, 20)).save(second, format="PNG")
    first.seek(0)
    second.seek(0)
    response = app.test_client().post(
        "/api/generate",
        data={
            "mode": "edit",
            "prompt": "换成新背景",
            "size": "original",
            "count": "2",
            "images": [(first, "one.png"), (second, "two.png")],
        },
        content_type="multipart/form-data",
    )
    job = wait_job(app.test_client(), response)
    calls = app.config["FAKE"].calls
    assert job["status"] == "completed"
    assert len(calls) == 2
    assert all(len(call[2]) == 1 for call in calls)
    assert {call[3]["size"] for call in calls} == {"640x480", "800x800"}
    assert all(call[3]["n"] == 2 for call in calls)
    assert len(job["images"]) == 4


def test_original_size_requires_image_to_image(app):
    response = app.test_client().post(
        "/api/generate",
        data={"mode": "text", "prompt": "一只橘猫", "size": "original"},
    )
    assert response.status_code == 400
    assert "图生图" in response.get_json()["error"]


def test_selected_model_is_forwarded(app):
    response = app.test_client().post(
        "/api/generate",
        data={"mode": "text", "prompt": "一只橘猫", "size": "1024x1024", "model": "gpt-image-2.5"},
    )
    wait_job(app.test_client(), response)
    assert app.config["FAKE"].calls[0][2]["model"] == "gpt-image-2.5"


def test_models_endpoint_lists_choices(app):
    class Listed:
        model = "gpt-image-2"

        def list_models(self):
            return [
                {"id": "gpt-image-2", "display_name": "GPT Image 2"},
                {"id": "gpt-image-2.5", "display_name": "gpt-image-2.5"},
            ]

    app.config["CLIENT_FACTORY"] = lambda: Listed()
    payload = app.test_client().get("/api/models").get_json()
    assert payload["default"] == "gpt-image-2"
    assert payload["models"][1] == {"id": "gpt-image-2.5", "name": "gpt-image-2.5"}


def test_five_images_are_requested_separately(app):
    response = app.test_client().post(
        "/api/generate",
        data={"mode": "text", "prompt": "一只橘猫", "size": "1024x1024", "count": "5"},
    )
    job = wait_job(app.test_client(), response)
    calls = app.config["FAKE"].calls
    assert len(calls) == 1
    assert calls[0][2]["n"] == 5
    assert "不要在同一张图里拼图" in calls[0][1]
    assert len(job["images"]) == 5


def test_history_is_kept_after_restart(app, tmp_path):
    response = app.test_client().post(
        "/api/generate",
        data={"mode": "text", "prompt": "一只橘猫", "size": "1024x1024"},
    )
    job = wait_job(app.test_client(), response)
    listed = app.test_client().get("/api/history").get_json()["history"]
    assert listed[0]["job_id"] == job["job_id"]
    assert listed[0]["prompt"] == "一只橘猫"
    assert listed[0]["images"][0]["url"].startswith("/results/")

    restarted = create_app(client_factory=lambda: FakeClient(), results_dir=tmp_path)
    again = restarted.test_client().get("/api/history").get_json()["history"]
    assert again[0]["job_id"] == job["job_id"]
    archive = restarted.test_client().get("/api/jobs/" + job["job_id"] + "/archive")
    assert archive.status_code == 200


def test_archive_contains_every_result(app):
    import io
    import zipfile

    response = app.test_client().post(
        "/api/generate",
        data={"mode": "text", "prompt": "一只橘猫", "size": "1024x1024", "count": "2"},
    )
    job = wait_job(app.test_client(), response)
    archive = app.test_client().get("/api/jobs/" + job["job_id"] + "/archive")
    assert archive.status_code == 200
    assert archive.mimetype == "application/zip"
    assert "attachment" in archive.headers["Content-Disposition"]
    assert archive.headers["Content-Disposition"].endswith('.zip"') or ".zip" in archive.headers["Content-Disposition"]
    with zipfile.ZipFile(io.BytesIO(archive.data)) as bundle:
        files = bundle.namelist()
        assert len(files) == 2
        assert bundle.read(files[0]) == PNG
        assert bundle.read(files[1]) == PNG


def test_text_mode_does_not_require_a_file(app):
    response = app.test_client().post(
        "/api/generate",
        data={"mode": "text", "prompt": "一只橘猫", "size": "1024x1024"},
    )
    payload = wait_job(app.test_client(), response)
    assert payload["status"] == "completed"
    assert app.config["FAKE"].calls[0][0] == "text"


def test_invalid_upload_is_rejected(app):
    response = app.test_client().post(
        "/api/generate",
        data={
            "mode": "edit",
            "prompt": "改一改",
            "size": "1024x1024",
            "images": (BytesIO(b"not-an-image"), "notes.png"),
        },
        content_type="multipart/form-data",
    )
    assert response.status_code == 400
    assert "无法识别" in response.get_json()["error"]


def test_result_downloads_with_attachment_name(app):
    target = Path(app.config["RESULTS_DIR"]) / "abc.png"
    target.write_bytes(PNG)
    response = app.test_client().get("/results/abc.png?download=1&filename=../../cat.png")
    assert response.status_code == 200
    assert response.data == PNG
    disposition = response.headers["Content-Disposition"]
    assert disposition.startswith("attachment;")
    assert "cat.png" in disposition
    inline = app.test_client().get("/results/abc.png")
    assert "attachment" not in inline.headers.get("Content-Disposition", "")


def test_results_reject_path_escape(app):
    response = app.test_client().get("/results/../../.env")
    assert response.status_code == 404


def test_generation_failure_stays_on_the_job(app):
    class Boom:
        def text_to_image(self, prompt, **kwargs):
            raise ShyfaiError("上游失败", status_code=502, request_id="up-1")

    app.config["CLIENT_FACTORY"] = lambda: Boom()
    response = app.test_client().post(
        "/api/generate",
        data={"mode": "text", "prompt": "一只橘猫", "size": "1024x1024"},
    )
    payload = wait_job(app.test_client(), response)
    assert payload["status"] == "failed"
    assert payload["request_id"] == "up-1"
    assert "上游失败" in payload["error"]


def test_upstream_error_is_json(app):
    def explode():
        raise ShyfaiError("密钥不正确", status_code=401, request_id="req-9")

    app.config["CLIENT_FACTORY"] = explode
    response = app.test_client().post(
        "/api/generate",
        data={"mode": "text", "prompt": "一只橘猫", "size": "1024x1024"},
    )
    assert response.status_code == 400
    assert response.get_json()["request_id"] == "req-9"
