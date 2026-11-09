"""命令行：文生图与图生图。"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

from generatepic.client import ShyfaiClient, ShyfaiError


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        prog="generatepic",
        description="调用 magic·art 图片接口，生成或按参考图改画。",
    )
    subparsers = parser.add_subparsers(dest="command", required=True)

    text = subparsers.add_parser("text", help="文生图")
    _add_common(text)
    text.add_argument("prompt", help="画面描述")

    edit = subparsers.add_parser("edit", help="图生图 / 按参考图编辑")
    _add_common(edit)
    edit.add_argument("prompt", help="要修改的内容和需要保留的内容")
    edit.add_argument(
        "-i",
        "--image",
        dest="images",
        action="append",
        required=True,
        help="参考图本地路径或 http(s) URL，可重复，最多 16 张",
    )
    edit.add_argument("--mask", help="可选蒙版，本地路径或 URL")

    subparsers.add_parser("models", help="列出当前 Key 可用的模型")

    args = parser.parse_args(argv)
    try:
        client = ShyfaiClient.from_env()
        if args.command == "models":
            for item in client.list_models():
                actions = ",".join(item.get("actions") or [])
                suffix = f"  [{actions}]" if actions else ""
                print(f"{item.get('id', '')}{suffix}")
            return 0
        kwargs = _generation_kwargs(args)
        if args.command == "text":
            result = client.text_to_image(args.prompt, **kwargs)
        else:
            result = client.image_to_image(args.prompt, args.images, mask=args.mask, **kwargs)
    except ShyfaiError as exc:
        print(str(exc), file=sys.stderr)
        if exc.request_id and exc.request_id not in str(exc):
            print(f"request_id: {exc.request_id}", file=sys.stderr)
        if exc.idempotency_key:
            print(f"idempotency_key: {exc.idempotency_key}", file=sys.stderr)
        return 1
    if result.request_id:
        print(f"request_id: {result.request_id}")
    for image in result.images:
        if image.path is not None:
            print(image.path)
        elif image.url:
            print(image.url)
    return 0


def _add_common(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("-o", "--output", default="outputs", help="保存文件或目录，默认 outputs/")
    parser.add_argument("--model", help="模型 ID，默认读取 SHYFAI_MODEL 或 gpt-image-2")
    parser.add_argument("--size", default="1024x1024", help="1024x1024、3:4 或 auto")
    parser.add_argument("--resolution", help="与比例配合，例如 1k、2k、4k")
    parser.add_argument("--n", type=int, default=1, help="生成张数，1 到 12")
    parser.add_argument("--quality", help="质量档位，例如 auto")
    parser.add_argument("--format", default="png", dest="output_format", choices=["png", "jpeg", "jpg", "webp"])
    parser.add_argument("--b64", action="store_true", help="要求接口返回 Base64，而不是图片 URL")
    parser.add_argument("--callback", dest="callback_url", help="真实回调地址。提供后接口会异步受理")
    parser.add_argument("--reference", dest="client_reference", help="业务侧关联 ID")
    parser.add_argument(
        "--idempotency-key",
        help="同一笔请求重试时必须复用此键和相同参数；留空则自动生成",
    )


def _generation_kwargs(args: argparse.Namespace) -> dict[str, object]:
    output_format = args.output_format
    output = args.output
    if output and Path(output).suffix.lower() in {".png", ".jpg", ".jpeg", ".webp"} and output_format == "png":
        suffix = Path(output).suffix.lower().lstrip(".")
        if suffix in {"jpg", "jpeg"}:
            output_format = "jpeg"
        elif suffix == "webp":
            output_format = "webp"
    return {
        "model": args.model,
        "n": args.n,
        "size": args.size,
        "resolution": args.resolution,
        "quality": args.quality,
        "output_format": output_format,
        "response_format": "b64_json" if args.b64 else "url",
        "output": output,
        "callback_url": args.callback_url,
        "client_reference": args.client_reference,
        "idempotency_key": args.idempotency_key,
    }


if __name__ == "__main__":
    raise SystemExit(main())
