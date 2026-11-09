"""本机页面：上传参考图，或按文字生成图片。"""

from __future__ import annotations

import io
import json
import os
import re
import shutil
import socket
import sys
import tempfile
import threading
import time
import uuid
import webbrowser
import zipfile
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

from flask import Flask, abort, jsonify, request, send_file, send_from_directory
from PIL import Image, ImageOps

from generatepic.client import ShyfaiClient, ShyfaiError, sniff_image_mime
from generatepic.panels import split_panels

def bundle_root() -> Path:
    """打包后的只读资源目录。开发时就是项目根目录。"""
    frozen = getattr(sys, "frozen", False)
    if frozen:
        return Path(getattr(sys, "_MEIPASS"))
    return Path(__file__).resolve().parents[1]


def data_root() -> Path:
    """可写的配置和图片目录。安装版放在用户本地数据目录，避免装到 Program Files 后无法保存。"""
    if getattr(sys, "frozen", False):
        base = Path(os.environ.get("LOCALAPPDATA") or Path.home())
        folder = base / "generatepic"
        folder.mkdir(parents=True, exist_ok=True)
        return folder
    return Path(__file__).resolve().parents[1]


ROOT = data_root()
INDEX_PATH = bundle_root() / "generatepic" / "templates" / "index.html"
PLACEHOLDER_HOSTS = {"https://api.example.com"}
PLACEHOLDER_KEYS = {"sk-your-key", "YOUR_API_KEY"}
MIME_SUFFIX = {
    "image/png": ".png",
    "image/jpeg": ".jpg",
    "image/webp": ".webp",
    "image/gif": ".gif",
}
MAX_IMAGES = 16
MAX_BYTES = 80 * 1024 * 1024
_JOB_ID = re.compile(r"^[a-f0-9]{32}$")


def create_app(
    *,
    client_factory=None,
    results_dir: Path,
    env_file: Path | None = None,
) -> Flask:
    results_dir.mkdir(parents=True, exist_ok=True)
    if client_factory is None:
        dotenv = env_file or (ROOT / ".env")

        def client_factory() -> ShyfaiClient:
            return client_from_dotenv(dotenv)

    app = Flask(__name__)
    app.config["MAX_CONTENT_LENGTH"] = MAX_BYTES
    app.config["CLIENT_FACTORY"] = client_factory
    app.config["RESULTS_DIR"] = results_dir.resolve()
    app.config["JOBS"] = {}
    app.config["JOBS_LOCK"] = threading.RLock()
    app.config["HISTORY_PATH"] = results_dir.resolve() / "history.json"

    @app.get("/")
    def index():
        return INDEX_PATH.read_text(encoding="utf-8")

    @app.get("/api/config")
    def config():
        try:
            client = app.config["CLIENT_FACTORY"]()
        except ShyfaiError as exc:
            return jsonify({"ready": False, "error": str(exc)})
        return jsonify({"ready": True, "model": client.model})

    @app.post("/api/generate")
    def generate():
        mode = (request.form.get("mode") or "").strip()
        prompt = request.form.get("prompt") or ""
        size = (request.form.get("size") or "1024x1024").strip()
        model = _clean_model(request.form.get("model"))
        try:
            count = _clean_count(request.form.get("count"))
        except ShyfaiError as exc:
            return _error_response(exc)
        if mode not in {"text", "edit"}:
            return jsonify({"error": "请选择文生图或图生图"}), 400
        if not prompt.strip():
            return jsonify({"error": "请填写描述"}), 400
        upload_dir: Path | None = None
        try:
            app.config["CLIENT_FACTORY"]()
            paths: list[Path] = []
            if mode == "edit":
                uploads = [item for item in request.files.getlist("images") if item and item.filename]
                if not uploads:
                    return jsonify({"error": "请先上传至少一张参考图"}), 400
                if len(uploads) > MAX_IMAGES:
                    return jsonify({"error": "参考图最多 16 张"}), 400
                upload_dir = Path(tempfile.mkdtemp(prefix="generatepic-"))
                paths = [_store_upload(item, upload_dir) for item in uploads]
            if size == "original" and mode != "edit":
                raise ShyfaiError("与原图一致只能用于图生图，请先上传参考图")
        except ShyfaiError as exc:
            if upload_dir is not None:
                shutil.rmtree(upload_dir, ignore_errors=True)
            return _error_response(exc)
        job = _new_job(mode, prompt, model, size, count)
        _save_job(app, job)
        thread = threading.Thread(
            target=_run_generation,
            args=(app, job["id"], mode, prompt, size, model, count, paths, upload_dir),
            daemon=True,
        )
        thread.start()
        return jsonify(_public_job(job)), 202

    @app.get("/api/models")
    def models():
        try:
            client = app.config["CLIENT_FACTORY"]()
            listed = client.list_models()
        except (ShyfaiError, AttributeError) as exc:
            return jsonify({"models": [], "error": str(exc)})
        choices = []
        for item in listed:
            if not isinstance(item, dict):
                continue
            model_id = _clean_model(str(item.get("id") or ""))
            if not model_id:
                continue
            choices.append({"id": model_id, "name": str(item.get("display_name") or model_id)})
        return jsonify({"default": client.model, "models": choices})

    @app.get("/api/jobs/<job_id>")
    def job_status(job_id: str):
        if not _JOB_ID.fullmatch(job_id):
            abort(404)
        job = _get_job(app, job_id)
        if job is None:
            return jsonify({"error": "找不到这个生成任务"}), 404
        return jsonify(_public_job(job))

    @app.get("/api/history")
    def history():
        return jsonify({"history": [_history_public(item) for item in _read_history(app)]})

    @app.get("/api/jobs/<job_id>/archive")
    def archive(job_id: str):
        if not _JOB_ID.fullmatch(job_id):
            abort(404)
        job = _get_job(app, job_id)
        if job is None:
            return jsonify({"error": "找不到这个生成任务"}), 404
        folder = app.config["RESULTS_DIR"]
        names = []
        for image in job.get("images") or []:
            url = str(image.get("url") or "")
            name = url.rsplit("/", 1)[-1] if url.startswith("/results/") else ""
            target = (folder / name).resolve() if name else None
            if target is not None and target.parent == folder and target.is_file():
                names.append(target)
        if not names:
            return jsonify({"error": "还没有可以下载的图片"}), 404
        buffer = io.BytesIO()
        with zipfile.ZipFile(buffer, "w", compression=zipfile.ZIP_STORED) as bundle:
            for index, path in enumerate(names, start=1):
                bundle.write(path, arcname=f"{index:02d}-{path.name}")
        buffer.seek(0)
        stamp = time.strftime("%Y%m%d-%H%M%S")
        return send_file(buffer, mimetype="application/zip", as_attachment=True, download_name=f"generatepic-{stamp}.zip")

    @app.get("/results/<path:name>")
    def results(name: str):
        if "/" in name or "\\" in name or name.startswith("."):
            abort(404)
        folder = app.config["RESULTS_DIR"]
        target = (folder / name).resolve()
        if target.parent != folder or not target.is_file():
            abort(404)
        as_attachment = request.args.get("download") == "1"
        return send_from_directory(
            folder,
            name,
            as_attachment=as_attachment,
            download_name=_download_name(request.args.get("filename"), name) if as_attachment else None,
        )

    @app.errorhandler(413)
    def too_large(_exc):
        return jsonify({"error": "上传文件过大，单次请控制在 80MB 以内"}), 413

    return app


def _run_generation(
    app,
    job_id: str,
    mode: str,
    prompt: str,
    size: str,
    model: str | None,
    count: int,
    paths: list[Path],
    upload_dir: Path | None,
) -> None:
    separate_prompt = _single_image_prompt(prompt, count)
    try:
        if mode == "edit" and len(paths) > 1:
            _run_each_source(app, job_id, separate_prompt, size, model, count, paths)
        else:
            image_size = _size_like_original(paths[0]) if size == "original" else size
            result = _request_images(
                app.config["CLIENT_FACTORY"](),
                mode,
                separate_prompt,
                paths,
                model,
                count,
                image_size,
                _new_result_path(app.config["RESULTS_DIR"]),
                _progress(app, job_id, count),
            )
            _finish_result(app, job_id, result)
    except ShyfaiError as exc:
        _update_job(
            app,
            job_id,
            status="failed",
            stage="failed",
            message=str(exc),
            error=str(exc),
            request_id=exc.request_id,
        )
    finally:
        if upload_dir is not None:
            shutil.rmtree(upload_dir, ignore_errors=True)


def _progress(app, job_id: str, count: int):
    def on_progress(stage: str, message: str) -> None:
        if stage == "generating" and count > 1:
            message = f"正在一次绘制 {count} 张独立图片"
        _update_job(app, job_id, stage=stage, message=message)

    return on_progress


def _request_images(client, mode, prompt, paths, model, count, size, output, on_progress):
    if mode == "text":
        return client.text_to_image(prompt, model=model, n=count, size=size, output=output, on_progress=on_progress)
    return client.image_to_image(prompt, paths, model=model, n=count, size=size, output=output, on_progress=on_progress)


def _finish_result(app, job_id: str, result) -> None:
    folder = app.config["RESULTS_DIR"]
    images = [_public_image(image, folder) for image in result.images]
    _update_job(app, job_id, stage="saving", message="正在整理每一张图片")
    images = _expand_panels(images, folder)
    _update_job(
        app,
        job_id,
        status="completed",
        stage="completed",
        message="生成完成",
        request_id=result.request_id,
        images=images,
    )
    _remember(app, job_id)


def _run_each_source(app, job_id: str, prompt: str, size: str, model: str | None, count: int, paths: list[Path]) -> None:
    total = len(paths)
    folder = app.config["RESULTS_DIR"]
    _update_job(app, job_id, stage="generating", message=f"正在同时处理 {total} 张原图，每张生成 {count} 张")
    errors: list[str] = []
    finished = 0
    done_lock = threading.Lock()

    def work(path: Path) -> None:
        nonlocal finished
        client = app.config["CLIENT_FACTORY"]()
        image_size = _size_like_original(path) if size == "original" else size
        result = client.image_to_image(
            prompt,
            [path],
            model=model,
            n=count,
            size=image_size,
            output=_new_result_path(folder),
        )
        images = _expand_panels([_public_image(image, folder) for image in result.images], folder)
        with done_lock:
            finished += 1
            _publish_source(app, job_id, images, result.request_id, finished, total)

    workers = min(4, total)
    with ThreadPoolExecutor(max_workers=workers) as pool:
        futures = [pool.submit(work, path) for path in paths]
        for future in as_completed(futures):
            try:
                future.result()
            except ShyfaiError as exc:
                errors.append(str(exc))
    job = _get_job(app, job_id) or {}
    if not job.get("images"):
        raise ShyfaiError(errors[0] if errors else "没有生成图片")
    message = "生成完成" if not errors else f"部分原图失败：{errors[0]}"
    _update_job(app, job_id, status="completed", stage="completed", message=message)
    _remember(app, job_id)


def _publish_source(app, job_id: str, images: list[dict[str, str]], request_id, finished: int, total: int) -> None:
    with app.config["JOBS_LOCK"]:
        job = app.config["JOBS"].get(job_id)
        if job is None:
            return
        current = list(job.get("images") or [])
        current.extend(images)
        job["images"] = current
        if request_id:
            previous = job.get("request_id")
            job["request_id"] = f"{previous}, {request_id}" if previous else str(request_id)
        job["stage"] = "generating"
        job["message"] = f"已完成 {finished}/{total} 张原图，其余还在绘制"
        _remember_locked(app, job)


def _new_job(mode: str, prompt: str, model: str | None, size: str, count: int) -> dict:
    now = time.time()
    return {
        "id": uuid.uuid4().hex,
        "status": "running",
        "stage": "accepted",
        "message": "请求已接收",
        "started": now,
        "request_id": None,
        "images": [],
        "error": None,
        "mode": mode,
        "prompt": prompt.strip(),
        "model": model or "",
        "size": size,
        "count": count,
    }


def _save_job(app, job: dict) -> None:
    with app.config["JOBS_LOCK"]:
        app.config["JOBS"][job["id"]] = job


def _get_job(app, job_id: str) -> dict | None:
    with app.config["JOBS_LOCK"]:
        job = app.config["JOBS"].get(job_id)
        if job is not None:
            return dict(job)
    for record in _read_history(app):
        if record.get("id") == job_id:
            return record
    return None


def _update_job(app, job_id: str, **changes) -> None:
    with app.config["JOBS_LOCK"]:
        job = app.config["JOBS"].get(job_id)
        if job is not None:
            job.update(changes)


def _history_record(job: dict) -> dict:
    return {
        "id": job["id"],
        "started": job["started"],
        "status": job["status"],
        "stage": job["stage"],
        "message": job["message"],
        "request_id": job.get("request_id"),
        "images": list(job.get("images") or []),
        "mode": job.get("mode") or "",
        "prompt": job.get("prompt") or "",
        "model": job.get("model") or "",
        "size": job.get("size") or "",
        "count": job.get("count") or 1,
    }


def _remember(app, job_id: str) -> None:
    with app.config["JOBS_LOCK"]:
        job = app.config["JOBS"].get(job_id)
        if job is not None and job.get("images"):
            _remember_locked(app, job)


def _remember_locked(app, job: dict) -> None:
    if not job.get("images"):
        return
    record = _history_record(job)
    path = app.config["HISTORY_PATH"]
    jobs = [item for item in _read_history_file(path) if item.get("id") != record["id"]]
    jobs.insert(0, record)
    temporary = path.with_suffix(".json.tmp")
    temporary.write_text(json.dumps({"jobs": jobs[:50]}, ensure_ascii=False), encoding="utf-8")
    temporary.replace(path)


def _read_history(app) -> list[dict]:
    return _read_history_file(app.config["HISTORY_PATH"])


def _read_history_file(path: Path) -> list[dict]:
    if not path.is_file():
        return []
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return []
    jobs = payload.get("jobs") if isinstance(payload, dict) else None
    if not isinstance(jobs, list):
        return []
    return [item for item in jobs if isinstance(item, dict) and item.get("id")]


def _history_public(record: dict) -> dict:
    started = record.get("started") or time.time()
    public = _public_job({**record, "id": record["id"], "started": started})
    public["prompt"] = record.get("prompt") or ""
    public["mode"] = record.get("mode") or ""
    public["model"] = record.get("model") or ""
    public["created"] = time.strftime("%m-%d %H:%M", time.localtime(started))
    return public


def _public_job(job: dict) -> dict:
    return {
        "job_id": job["id"],
        "status": job["status"],
        "stage": job["stage"],
        "message": job["message"],
        "elapsed_seconds": max(0, int(time.time() - job["started"])),
        "request_id": job.get("request_id"),
        "images": list(job.get("images") or []),
        "error": job.get("error"),
    }


def _error_response(exc: ShyfaiError):
    body = {"error": str(exc)}
    if exc.request_id:
        body["request_id"] = exc.request_id
    if exc.idempotency_key:
        body["idempotency_key"] = exc.idempotency_key
    status = 400 if exc.status_code in {None, 400, 401, 404, 409, 413, 415} else 502
    return jsonify(body), status


def client_from_dotenv(path: Path) -> ShyfaiClient:
    values = _parse_env(path)
    base_url = _pick(values, "SHYFAI_BASE_URL", "BASE_URL")
    api_key = _pick(values, "SHYFAI_API_KEY", "API_KEY")
    upload_base_url = _pick(values, "SHYFAI_UPLOAD_BASE_URL", "UPLOAD_BASE_URL") or None
    model = _pick(values, "SHYFAI_MODEL", "MODEL") or "gpt-image-2"
    if not base_url or not api_key:
        raise ShyfaiError(f"请在 {path} 中填写 SHYFAI_BASE_URL 和 SHYFAI_API_KEY，并保存文件")
    if base_url.rstrip("/") in PLACEHOLDER_HOSTS or api_key in PLACEHOLDER_KEYS:
        raise ShyfaiError("请把 .env 里的示例地址和示例密钥换成控制台中的真实值，并保存文件")
    return ShyfaiClient(base_url, api_key, upload_base_url=upload_base_url, model=model)


def _port_is_open(port: int) -> bool:
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as sock:
        sock.settimeout(0.2)
        return sock.connect_ex(("127.0.0.1", port)) == 0


def _alert(text: str) -> None:
    print(text, flush=True)
    if sys.platform == "win32":
        import ctypes

        ctypes.windll.user32.MessageBoxW(None, text, "generatepic", 0x40)


def main() -> None:
    url = "http://127.0.0.1:8765"
    if _port_is_open(8765):
        webbrowser.open(url)
        return
    env_file = data_root() / ".env"
    example = bundle_root() / ".env.example"
    if not env_file.is_file() and example.is_file():
        env_file.write_text(example.read_text(encoding="utf-8"), encoding="utf-8")
    app = create_app(results_dir=data_root() / "outputs" / "web", env_file=env_file)
    print(f"打开 {url}", flush=True)
    print(f"配置文件 {env_file}", flush=True)
    threading.Timer(0.6, lambda: webbrowser.open(url)).start()
    try:
        app.run(host="127.0.0.1", port=8765, threaded=True)
    except OSError as exc:
        _alert(f"无法启动本地页面：{exc}")


def _parse_env(path: Path) -> dict[str, str]:
    values: dict[str, str] = {}
    if not path.is_file():
        return values
    for raw_line in path.read_text(encoding="utf-8").splitlines():
        line = raw_line.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        key, value = line.split("=", 1)
        values[key.strip()] = value.strip().strip('"').strip("'")
    return values


def _pick(values: dict[str, str], *keys: str) -> str:
    for key in keys:
        if values.get(key):
            return values[key]
        if os.environ.get(key):
            return os.environ[key]
    return ""


def _expand_panels(images: list[dict[str, str]], folder: Path) -> list[dict[str, str]]:
    expanded: list[dict[str, str]] = []
    for image in images:
        url = image.get("url") or ""
        name = url.rsplit("/", 1)[-1] if url.startswith("/results/") else ""
        path = folder / name if name else None
        if path is None or not path.is_file():
            expanded.append(image)
            continue
        try:
            panels = split_panels(path)
        except OSError:
            expanded.append(image)
            continue
        if len(panels) < 2:
            expanded.append(image)
            continue
        expanded.extend({"url": f"/results/{panel.name}"} for panel in panels)
    return expanded


def _size_like_original(path: Path) -> str:
    try:
        with Image.open(path) as image:
            oriented = ImageOps.exif_transpose(image)
            width, height = oriented.size
    except OSError as exc:
        raise ShyfaiError("无法读取原图尺寸") from exc
    width = _snap_side(width)
    height = _snap_side(height)
    if width > height * 3:
        width = height * 3
    elif height > width * 3:
        height = width * 3
    return f"{width}x{height}"


def _snap_side(value: int) -> int:
    snapped = int(round(value / 16) * 16)
    return min(4096, max(16, snapped))


def _clean_count(value: str | None) -> int:
    text = (value or "1").strip()
    if not text.isdigit():
        raise ShyfaiError("生成张数必须是 1 到 5 的整数")
    count = int(text)
    if not 1 <= count <= 5:
        raise ShyfaiError("生成张数必须是 1 到 5 的整数")
    return count


def _single_image_prompt(prompt: str, count: int) -> str:
    text = prompt.strip()
    if count == 1:
        return text
    return text + "\n每一张都是独立的完整图片，不要在同一张图里拼图、分格或并排多张照片。"


def _clean_model(value: str | None) -> str | None:
    text = (value or "").strip()
    if not text or len(text) > 80 or not re.fullmatch(r"[A-Za-z0-9][A-Za-z0-9._:-]*", text):
        return None
    return text


def _download_name(requested: str | None, fallback: str) -> str:
    raw = Path(requested or fallback).name
    cleaned = re.sub(r"[^A-Za-z0-9._-]", "", raw)
    if not cleaned or cleaned.startswith(".") or "." not in cleaned:
        return fallback
    return cleaned[:120]


def _new_result_path(folder: Path) -> Path:
    return folder / f"{uuid.uuid4().hex}.png"


def _store_upload(storage, folder: Path) -> Path:
    raw_name = Path(storage.filename or "").name
    suffix = Path(raw_name).suffix.lower()
    if suffix not in {".png", ".jpg", ".jpeg", ".webp", ".gif"}:
        suffix = ".img"
    destination = folder / f"{uuid.uuid4().hex}{suffix}"
    storage.save(destination)
    try:
        mime = sniff_image_mime(destination)
    except ShyfaiError:
        destination.unlink(missing_ok=True)
        raise ShyfaiError(f"无法识别图片格式: {raw_name or '上传文件'}") from None
    corrected = destination.with_suffix(MIME_SUFFIX[mime])
    if corrected != destination:
        destination.replace(corrected)
    return corrected


def _public_image(image, folder: Path) -> dict[str, str]:
    if image.path is not None and image.path.is_file():
        path = image.path.resolve()
        if path.parent != folder or not path.is_file():
            raise ShyfaiError("生成结果不在预期目录中")
        return {"url": f"/results/{path.name}"}
    if image.url:
        return {"url": image.url}
    raise ShyfaiError("生成结果里没有可显示的图片")


if __name__ == "__main__":
    main()
