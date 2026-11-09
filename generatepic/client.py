"""magic·art 图片接口客户端。

文档：https://shyfai.cn/#start
文生图 POST /v1/images/generations
图生图 POST /v1/images/edits
"""

from __future__ import annotations

import base64
import os
import time
import uuid
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any
from urllib.parse import urlparse

import requests

CONNECT_TIMEOUT = 10
SYNC_READ_TIMEOUT = 420
UPLOAD_READ_TIMEOUT = 120
POLL_READ_TIMEOUT = 30
POLL_INTERVAL_SECONDS = 3
POLL_DEADLINE_SECONDS = 900
BUSY_STATUS_CODES = {429, 466, 477, 488, 502, 503, 504}
PENDING_STATUSES = {"queued", "running"}
TERMINAL_FAILURES = {"failed", "cancelled"}
OUTPUT_FORMATS = {"png", "jpeg", "jpg", "webp"}
MIME_BY_EXT = {
    "png": "image/png",
    "jpg": "image/jpeg",
    "jpeg": "image/jpeg",
    "webp": "image/webp",
    "gif": "image/gif",
}


class ShyfaiError(Exception):
    """接口或本地参数错误。"""

    def __init__(
        self,
        message: str,
        *,
        status_code: int | None = None,
        code: str | None = None,
        request_id: str | None = None,
        idempotency_key: str | None = None,
    ) -> None:
        super().__init__(message)
        self.status_code = status_code
        self.code = code
        self.request_id = request_id
        self.idempotency_key = idempotency_key


class UploadNotSupported(ShyfaiError):
    """当前上传入口没有公开上传接口。"""


@dataclass
class GeneratedImage:
    url: str | None
    path: Path | None


@dataclass
class GenerationResult:
    request_id: str | None
    status: str
    images: list[GeneratedImage]
    raw: dict[str, Any] = field(default_factory=dict, repr=False)


def load_env_file(path: Path | None = None) -> None:
    """读取 .env。已存在的环境变量优先，不覆盖。"""
    env_path = path or Path.cwd() / ".env"
    if not env_path.is_file():
        return
    for raw_line in env_path.read_text(encoding="utf-8").splitlines():
        line = raw_line.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        key, value = line.split("=", 1)
        key = key.strip()
        value = value.strip().strip('"').strip("'")
        if key:
            os.environ.setdefault(key, value)


def normalize_base_url(url: str) -> str:
    """去掉末尾斜杠和 requests 示例不需要的 /v1。"""
    normalized = url.strip().rstrip("/")
    if normalized.endswith("/v1"):
        normalized = normalized[: -len("/v1")]
    parsed = urlparse(normalized)
    if parsed.scheme not in {"http", "https"} or not parsed.netloc:
        raise ShyfaiError(f"API 地址无效: {url}")
    return normalized


def normalize_size(size: str) -> str:
    text = size.strip().replace("×", "x")
    if text.lower() == "auto":
        return "auto"
    if "x" in text and ":" not in text:
        width_text, height_text = text.split("x", 1)
        if not width_text.isdigit() or not height_text.isdigit():
            raise ShyfaiError("像素尺寸请写成 1024x1024，使用小写字母 x")
        width, height = int(width_text), int(height_text)
        if width < 1 or height < 1 or width > 4096 or height > 4096:
            raise ShyfaiError("宽和高必须在 1 到 4096 像素之间")
        return f"{width}x{height}"
    if ":" in text:
        left, right = text.split(":", 1)
        if _is_ratio_part(left) and _is_ratio_part(right):
            return text
    raise ShyfaiError("size 应为 1024x1024、3:4 或 auto")


def _is_ratio_part(value: str) -> bool:
    if not value or value.count(".") > 1:
        return False
    number = value.replace(".", "", 1)
    return number.isdigit() and float(value) > 0


def sniff_image_mime(path: Path) -> str:
    head = path.read_bytes()[:16]
    if head.startswith(b"\x89PNG\r\n\x1a\n"):
        return "image/png"
    if head.startswith(b"\xff\xd8\xff"):
        return "image/jpeg"
    if head.startswith(b"RIFF") and head[8:12] == b"WEBP":
        return "image/webp"
    if head.startswith((b"GIF87a", b"GIF89a")):
        return "image/gif"
    raise ShyfaiError(f"无法识别图片格式: {path}")


def extension_for(content_type: str | None, output_format: str | None, url: str | None) -> str:
    if content_type:
        mime = content_type.split(";", 1)[0].strip().lower()
        for ext, known in MIME_BY_EXT.items():
            if mime == known:
                return "jpg" if ext == "jpeg" else ext
    if output_format:
        return "jpg" if output_format == "jpeg" else output_format
    if url:
        suffix = Path(urlparse(url).path).suffix.lower().lstrip(".")
        if suffix in {"png", "jpg", "jpeg", "webp"}:
            return "jpg" if suffix == "jpeg" else suffix
    return "png"


class ShyfaiClient:
    """同步文生图、图生图。出现 202 或任务未完成时轮询 status_url。"""

    def __init__(
        self,
        base_url: str,
        api_key: str,
        *,
        upload_base_url: str | None = None,
        model: str = "gpt-image-2",
        session: requests.Session | None = None,
        poll_interval: float = POLL_INTERVAL_SECONDS,
        poll_deadline: float = POLL_DEADLINE_SECONDS,
    ) -> None:
        if not api_key.strip():
            raise ShyfaiError("API Key 不能为空")
        self.base_url = normalize_base_url(base_url)
        self.api_key = api_key.strip()
        self.upload_base_url = normalize_base_url(upload_base_url or base_url)
        self.model = model.strip() or "gpt-image-2"
        self.session = session or requests.Session()
        self.poll_interval = poll_interval
        self.poll_deadline = poll_deadline

    @classmethod
    def from_env(cls, env_file: Path | None = None) -> ShyfaiClient:
        load_env_file(env_file)
        base_url = os.environ.get("SHYFAI_BASE_URL") or os.environ.get("BASE_URL")
        api_key = os.environ.get("SHYFAI_API_KEY") or os.environ.get("API_KEY")
        upload_base_url = os.environ.get("SHYFAI_UPLOAD_BASE_URL") or os.environ.get("UPLOAD_BASE_URL")
        model = os.environ.get("SHYFAI_MODEL") or os.environ.get("MODEL") or "gpt-image-2"
        if not base_url or not api_key:
            raise ShyfaiError(
                "请在环境变量或 .env 中设置 SHYFAI_BASE_URL 和 SHYFAI_API_KEY"
            )
        return cls(base_url, api_key, upload_base_url=upload_base_url or None, model=model)

    def list_models(self) -> list[dict[str, Any]]:
        response = self._send(
            "GET",
            f"{self.base_url}/v1/models",
            headers=self._auth_headers(),
            timeout=(CONNECT_TIMEOUT, POLL_READ_TIMEOUT),
        )
        body = self._require_json(response)
        data = body.get("data")
        if not isinstance(data, list):
            raise ShyfaiError("模型列表响应缺少 data 数组", status_code=response.status_code)
        return data

    def text_to_image(
        self,
        prompt: str,
        *,
        model: str | None = None,
        n: int = 1,
        size: str | None = "1024x1024",
        resolution: str | None = None,
        aspect_ratio: str | None = None,
        quality: str | None = None,
        output_format: str = "png",
        response_format: str = "url",
        output: str | Path | None = None,
        callback_url: str | None = None,
        client_reference: str | None = None,
        idempotency_key: str | None = None,
        on_progress=None,
    ) -> GenerationResult:
        fields = self._image_fields(
            prompt,
            model=model,
            n=n,
            size=size,
            resolution=resolution,
            aspect_ratio=aspect_ratio,
            quality=quality,
            output_format=output_format,
            response_format=response_format,
            callback_url=callback_url,
            client_reference=client_reference,
        )
        key = idempotency_key or str(uuid.uuid4())
        _report(on_progress, "generating", "模型正在绘制")
        response = self._post_with_retry(
            on_progress,
            method="POST",
            url=f"{self.base_url}/v1/images/generations",
            headers={**self._auth_headers(), "Idempotency-Key": key, "Content-Type": "application/json"},
            json=fields,
            timeout=(CONNECT_TIMEOUT, SYNC_READ_TIMEOUT),
            idempotency_key=key,
        )
        return self._finish(response, key, output, fields["output_format"], on_progress)

    def image_to_image(
        self,
        prompt: str,
        images: list[str | Path],
        *,
        model: str | None = None,
        n: int = 1,
        size: str | None = "1024x1024",
        resolution: str | None = None,
        aspect_ratio: str | None = None,
        quality: str | None = None,
        output_format: str = "png",
        response_format: str = "url",
        output: str | Path | None = None,
        mask: str | Path | None = None,
        callback_url: str | None = None,
        client_reference: str | None = None,
        idempotency_key: str | None = None,
        on_progress=None,
    ) -> GenerationResult:
        if not images:
            raise ShyfaiError("图生图至少需要一张参考图")
        if len(images) > 16:
            raise ShyfaiError("参考图最多 16 张")
        fields = self._image_fields(
            prompt,
            model=model,
            n=n,
            size=size,
            resolution=resolution,
            aspect_ratio=aspect_ratio,
            quality=quality,
            output_format=output_format,
            response_format=response_format,
            callback_url=callback_url,
            client_reference=client_reference,
        )
        key = idempotency_key or str(uuid.uuid4())
        remote_urls, local_paths = _split_images(images)
        if local_paths:
            _report(on_progress, "upload", "正在上传参考图")
            try:
                uploaded = {path.resolve(): self.upload(path)["url"] for path in local_paths}
            except UploadNotSupported:
                if remote_urls:
                    raise ShyfaiError(
                        "上传接口不可用，无法混用本地文件和远程图片 URL。请先把本地图片放到可公开访问的地址。"
                    ) from None
                _report(on_progress, "generating", "模型正在绘制")
                return self._edit_multipart(fields, local_paths, mask, key, output, on_progress)
            remote_urls = [
                str(image) if _is_remote(str(image)) else uploaded[Path(image).resolve()]
                for image in images
            ]
        mask_url = self._resolve_mask(mask) if mask is not None else None
        _report(on_progress, "generating", "模型正在绘制")
        return self._edit_json(fields, remote_urls, mask_url, key, output, on_progress)

    def upload(self, path: str | Path) -> dict[str, Any]:
        file_path = Path(path)
        if not file_path.is_file():
            raise ShyfaiError(f"找不到参考图: {file_path}")
        mime = sniff_image_mime(file_path)
        with file_path.open("rb") as handle:
            response = self._send(
                "POST",
                f"{self.upload_base_url}/v1/images/uploads",
                files={"file": (file_path.name, handle, mime)},
                timeout=(CONNECT_TIMEOUT, UPLOAD_READ_TIMEOUT),
            )
        if response.status_code in {404, 405, 501, 502, 503, 504}:
            raise UploadNotSupported(
                "当前入口的 /v1/images/uploads 不可用，改为随编辑请求直接提交图片",
                status_code=response.status_code,
            )
        body = self._require_json(response)
        if body.get("state") not in {None, "ready"}:
            raise ShyfaiError(f"参考图尚未就绪: {body.get('state')}")
        if not body.get("url"):
            raise ShyfaiError("上传响应缺少 url")
        return body

    def _edit_json(
        self,
        fields: dict[str, Any],
        image_urls: list[str],
        mask_url: str | None,
        idempotency_key: str,
        output: str | Path | None,
        on_progress=None,
    ) -> GenerationResult:
        payload = dict(fields)
        payload["images"] = [{"image_url": url} for url in image_urls]
        if mask_url:
            payload["mask"] = mask_url
        response = self._post_with_retry(
            on_progress,
            method="POST",
            url=f"{self.base_url}/v1/images/edits",
            headers={**self._auth_headers(), "Idempotency-Key": idempotency_key, "Content-Type": "application/json"},
            json=payload,
            timeout=(CONNECT_TIMEOUT, SYNC_READ_TIMEOUT),
            idempotency_key=idempotency_key,
        )
        return self._finish(response, idempotency_key, output, fields["output_format"], on_progress)

    def _edit_multipart(
        self,
        fields: dict[str, Any],
        paths: list[Path],
        mask: str | Path | None,
        idempotency_key: str,
        output: str | Path | None,
        on_progress=None,
    ) -> GenerationResult:
        handles = []
        files: list[tuple[str, tuple[str, Any, str]]] = []
        try:
            for path in paths:
                mime = sniff_image_mime(path)
                handle = path.open("rb")
                handles.append(handle)
                files.append(("image[]", (path.name, handle, mime)))
            if mask is not None:
                mask_path = Path(mask)
                if _is_remote(str(mask)):
                    raise ShyfaiError("直接上传编辑时，蒙版也需要是本地图片文件")
                if not mask_path.is_file():
                    raise ShyfaiError(f"找不到蒙版: {mask_path}")
                mime = sniff_image_mime(mask_path)
                handle = mask_path.open("rb")
                handles.append(handle)
                files.append(("mask", (mask_path.name, handle, mime)))
            response = self._post_with_retry(
                on_progress,
                method="POST",
                url=f"{self.base_url}/v1/images/edits",
                headers={**self._auth_headers(), "Idempotency-Key": idempotency_key},
                data=_form_fields(fields),
                files=files,
                timeout=(CONNECT_TIMEOUT, SYNC_READ_TIMEOUT),
                idempotency_key=idempotency_key,
            )
        finally:
            for handle in handles:
                handle.close()
        return self._finish(response, idempotency_key, output, fields["output_format"], on_progress)

    def _resolve_mask(self, mask: str | Path) -> str:
        if _is_remote(str(mask)):
            return str(mask)
        return self.upload(mask)["url"]

    def _finish(
        self,
        response: requests.Response,
        idempotency_key: str,
        output: str | Path | None,
        output_format: str,
        on_progress=None,
    ) -> GenerationResult:
        if response.status_code in {408, 504}:
            body = _safe_json(response)
            request_id = _request_id(body, response)
            if request_id:
                body = self._poll(_status_url(body, self.base_url, request_id))
            else:
                raise self._error_from_response(response, idempotency_key)
        elif response.status_code >= 400:
            raise self._error_from_response(response, idempotency_key)
        else:
            body = self._require_json(response)
            status = body.get("status")
            if response.status_code == 202 or status in PENDING_STATUSES:
                request_id = _request_id(body, response)
                body = self._poll(_status_url(body, self.base_url, request_id))
        if body.get("status") in TERMINAL_FAILURES:
            raise _error_from_body(body, idempotency_key=idempotency_key)
        _report(on_progress, "saving", "正在保存图片")
        images = self._materialize(body, output, output_format)
        return GenerationResult(
            request_id=body.get("request_id") or body.get("id"),
            status=str(body.get("status") or "completed"),
            images=images,
            raw=body,
        )

    def _poll(self, status_url: str) -> dict[str, Any]:
        deadline = time.monotonic() + self.poll_deadline
        while True:
            try:
                response = self._send(
                    "GET",
                    status_url,
                    timeout=(CONNECT_TIMEOUT, POLL_READ_TIMEOUT),
                )
            except ShyfaiError as exc:
                if exc.status_code not in BUSY_STATUS_CODES and "网络" not in str(exc):
                    raise
                response = None
            if response is not None and response.status_code in BUSY_STATUS_CODES:
                response = None
            if response is not None:
                body = self._require_json(response)
                status = body.get("status")
                if status == "completed" or (status is None and body.get("data")):
                    return body
                if status in TERMINAL_FAILURES:
                    raise _error_from_body(body)
                if status not in PENDING_STATUSES:
                    raise ShyfaiError(f"未知任务状态: {status}", request_id=body.get("request_id"))
            if time.monotonic() >= deadline:
                raise ShyfaiError(
                    "本地等待结束，任务未必失败。请稍后用 status_url 继续查询，不要立刻换新任务重画。",
                    request_id=_request_id_from_url(status_url),
                )
            time.sleep(self.poll_interval)

    def _materialize(
        self,
        body: dict[str, Any],
        output: str | Path | None,
        output_format: str,
    ) -> list[GeneratedImage]:
        data = body.get("data")
        if not isinstance(data, list) or not data:
            raise ShyfaiError("响应里没有图片数据", request_id=body.get("request_id"))
        def save(index: int, item: Any) -> GeneratedImage:
            if not isinstance(item, dict):
                raise ShyfaiError("图片结果格式不正确", request_id=body.get("request_id"))
            url = item.get("url") if isinstance(item.get("url"), str) else None
            encoded = item.get("b64_json") if isinstance(item.get("b64_json"), str) else None
            path: Path | None = None
            if encoded:
                target = _output_path(output, index, len(data), output_format, None)
                target.write_bytes(base64.b64decode(encoded))
                path = target
            elif url and output is not None:
                target = _output_path(output, index, len(data), output_format, url)
                downloaded, content_type = self._download(url)
                suffix = extension_for(content_type, output_format, url)
                if target.suffix.lower().lstrip(".") not in {suffix, "jpeg"}:
                    target = target.with_suffix(f".{suffix}")
                target.write_bytes(downloaded)
                path = target
            elif not url:
                raise ShyfaiError("图片结果既没有 url 也没有 b64_json", request_id=body.get("request_id"))
            return GeneratedImage(url=url, path=path)

        if len(data) == 1:
            return [save(0, data[0])]
        with ThreadPoolExecutor(max_workers=min(4, len(data))) as pool:
            return list(pool.map(lambda pair: save(*pair), enumerate(data)))

    def _download(self, url: str) -> tuple[bytes, str | None]:
        response = self._send("GET", url, timeout=(CONNECT_TIMEOUT, UPLOAD_READ_TIMEOUT))
        if response.status_code >= 400:
            raise ShyfaiError(f"下载图片失败: HTTP {response.status_code}", status_code=response.status_code)
        return response.content, response.headers.get("Content-Type")

    def _image_fields(
        self,
        prompt: str,
        *,
        model: str | None,
        n: int,
        size: str | None,
        resolution: str | None,
        aspect_ratio: str | None,
        quality: str | None,
        output_format: str,
        response_format: str,
        callback_url: str | None,
        client_reference: str | None,
    ) -> dict[str, Any]:
        text = prompt.strip()
        if not text:
            raise ShyfaiError("prompt 不能为空")
        if isinstance(n, bool) or not isinstance(n, int) or not 1 <= n <= 12:
            raise ShyfaiError("n 必须是 1 到 12 的整数")
        fmt = output_format.strip().lower()
        if fmt not in OUTPUT_FORMATS:
            raise ShyfaiError("output_format 仅支持 png、jpeg、jpg、webp")
        if response_format not in {"url", "b64_json"}:
            raise ShyfaiError("response_format 仅支持 url 或 b64_json")
        fields: dict[str, Any] = {
            "model": (model or self.model).strip(),
            "prompt": text,
            "n": n,
            "output_format": fmt,
            "response_format": response_format,
        }
        if not fields["model"]:
            raise ShyfaiError("model 不能为空")
        if size is not None:
            fields["size"] = normalize_size(size)
        if resolution is not None:
            fields["resolution"] = resolution.strip().lower()
        if aspect_ratio is not None:
            fields["aspect_ratio"] = aspect_ratio.strip()
        if quality is not None:
            fields["quality"] = quality.strip()
        if callback_url is not None:
            fields["callback_url"] = callback_url.strip()
        if client_reference is not None:
            fields["client_reference"] = client_reference
        return fields

    def _auth_headers(self) -> dict[str, str]:
        return {"Authorization": f"Bearer {self.api_key}"}

    def _post_with_retry(self, on_progress, **kwargs: Any) -> requests.Response:
        try:
            return self._send(**kwargs)
        except ShyfaiError as exc:
            if "网络" not in str(exc):
                raise
            _rewind_files(kwargs.get("files"))
            _report(on_progress, "generating", "连接中断，正在用同一个任务重新连接")
            time.sleep(self.poll_interval)
            try:
                return self._send(**kwargs)
            except ShyfaiError as retry_exc:
                if "网络" not in str(retry_exc):
                    raise
                raise ShyfaiError(
                    "连接中断，同一个任务重新连接后仍然失败。请过一会儿再试。",
                    idempotency_key=retry_exc.idempotency_key,
                ) from retry_exc

    def _send(self, method: str, url: str, *, idempotency_key: str | None = None, **kwargs: Any) -> requests.Response:
        try:
            return self.session.request(method, url, **kwargs)
        except (requests.Timeout, requests.ConnectionError) as exc:
            raise ShyfaiError(
                f"网络请求失败: {exc.__class__.__name__}。如果请求可能已经受理，请用同一个 Idempotency-Key 重试，不要立刻换新任务。",
                idempotency_key=idempotency_key,
            ) from exc

    def _require_json(self, response: requests.Response) -> dict[str, Any]:
        if response.status_code >= 400:
            raise self._error_from_response(response, None)
        body = _safe_json(response)
        if body is None:
            raise ShyfaiError("响应不是 JSON", status_code=response.status_code)
        return body

    def _error_from_response(self, response: requests.Response, idempotency_key: str | None) -> ShyfaiError:
        body = _safe_json(response) or {}
        return _error_from_body(body, status_code=response.status_code, idempotency_key=idempotency_key, response=response)


def _report(callback, stage: str, message: str) -> None:
    if callback is not None:
        callback(stage, message)


def _split_images(images: list[str | Path]) -> tuple[list[str], list[Path]]:
    remote_urls: list[str] = []
    local_paths: list[Path] = []
    seen: set[Path] = set()
    for image in images:
        text = str(image)
        if _is_remote(text):
            remote_urls.append(text)
            continue
        path = Path(image)
        if not path.is_file():
            raise ShyfaiError(f"找不到参考图: {path}")
        resolved = path.resolve()
        if resolved not in seen:
            local_paths.append(path)
            seen.add(resolved)
    return remote_urls, local_paths


def _is_remote(value: str) -> bool:
    return value.startswith(("http://", "https://", "data:"))


def _rewind_files(files: Any) -> None:
    if not files:
        return
    items = files.values() if isinstance(files, dict) else files
    for item in items:
        handle = item[1] if isinstance(item, tuple) and len(item) > 1 else item
        if isinstance(handle, tuple) and len(handle) > 1:
            handle = handle[1]
        if hasattr(handle, "seek"):
            handle.seek(0)


def _form_fields(fields: dict[str, Any]) -> dict[str, str]:
    return {key: str(value) for key, value in fields.items() if value is not None}


def _safe_json(response: requests.Response) -> dict[str, Any] | None:
    try:
        body = response.json()
    except ValueError:
        return None
    if not isinstance(body, dict):
        return None
    return body


def _request_id(body: dict[str, Any] | None, response: requests.Response | None = None) -> str | None:
    if body:
        request_id = body.get("request_id") or body.get("id")
        if isinstance(request_id, str) and request_id:
            return request_id
    if response is not None:
        header = response.headers.get("X-Request-Id")
        if header:
            return header
    return None


def _status_url(body: dict[str, Any] | None, base_url: str, request_id: str | None) -> str:
    if body and isinstance(body.get("status_url"), str) and body["status_url"]:
        return body["status_url"]
    if not request_id:
        raise ShyfaiError("任务已受理，但响应里没有 request_id 或 status_url")
    return f"{base_url}/v1/images/tasks/{request_id}"


def _request_id_from_url(status_url: str) -> str | None:
    parts = urlparse(status_url).path.rstrip("/").split("/")
    if len(parts) >= 2 and parts[-2] == "tasks":
        return parts[-1]
    return None


def _error_from_body(
    body: dict[str, Any],
    *,
    status_code: int | None = None,
    idempotency_key: str | None = None,
    response: requests.Response | None = None,
) -> ShyfaiError:
    error = body.get("error")
    code = None
    if isinstance(error, dict):
        message = str(error.get("message") or "请求失败")
        code = error.get("code")
        code = str(code) if code else None
    elif isinstance(error, str) and error:
        message = error
    else:
        message = f"请求失败: HTTP {status_code}" if status_code else str(body.get("status") or "任务失败")
    request_id = _request_id(body, response)
    if request_id:
        message = f"{message} (request_id={request_id})"
    return ShyfaiError(
        message,
        status_code=status_code,
        code=code,
        request_id=request_id,
        idempotency_key=idempotency_key,
    )


def _output_path(
    output: str | Path | None,
    index: int,
    count: int,
    output_format: str,
    url: str | None,
) -> Path:
    ext = "jpg" if output_format == "jpeg" else output_format
    stamp = time.strftime("%Y%m%d-%H%M%S")
    if output is None:
        directory = Path("outputs")
        directory.mkdir(parents=True, exist_ok=True)
        return directory / f"{stamp}-{index}.{ext}"
    path = Path(output)
    suffix = path.suffix.lower()
    if suffix in {".png", ".jpg", ".jpeg", ".webp"}:
        path.parent.mkdir(parents=True, exist_ok=True)
        if count == 1:
            return path
        return path.with_name(f"{path.stem}-{index}{path.suffix}")
    path.mkdir(parents=True, exist_ok=True)
    guessed = extension_for(None, output_format, url)
    return path / f"{stamp}-{index}.{guessed}"
