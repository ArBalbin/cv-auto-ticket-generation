from pathlib import Path

from core.config import (
    OBJECT_STORAGE_ACCESS_KEY_ID,
    OBJECT_STORAGE_ADDRESSING_STYLE,
    OBJECT_STORAGE_BUCKET,
    OBJECT_STORAGE_ENABLED,
    OBJECT_STORAGE_ENDPOINT_URL,
    OBJECT_STORAGE_PREFIX,
    OBJECT_STORAGE_PUBLIC_BASE_URL,
    OBJECT_STORAGE_REGION,
    OBJECT_STORAGE_SECRET_ACCESS_KEY,
)

try:
    import boto3
    from botocore.client import Config
    from botocore.exceptions import BotoCoreError, ClientError
except ModuleNotFoundError:
    boto3 = None

    class BotoCoreError(Exception):
        pass

    class ClientError(Exception):
        pass


_client = None
_warned_unavailable = False


def is_configured() -> bool:
    return bool(OBJECT_STORAGE_ENABLED and OBJECT_STORAGE_BUCKET)


def _client_config():
    if OBJECT_STORAGE_ADDRESSING_STYLE in {"path", "virtual"}:
        return Config(s3={"addressing_style": OBJECT_STORAGE_ADDRESSING_STYLE})
    return None


def client():
    global _client, _warned_unavailable

    if not is_configured():
        return None
    if boto3 is None:
        if not _warned_unavailable:
            print("[ObjectStorage] boto3 is not installed; ticket upload disabled")
            _warned_unavailable = True
        return None
    if _client is not None:
        return _client

    kwargs = {
        "service_name": "s3",
        "endpoint_url": OBJECT_STORAGE_ENDPOINT_URL or None,
        "region_name": OBJECT_STORAGE_REGION,
        "aws_access_key_id": OBJECT_STORAGE_ACCESS_KEY_ID,
        "aws_secret_access_key": OBJECT_STORAGE_SECRET_ACCESS_KEY,
    }
    config = _client_config()
    if config is not None:
        kwargs["config"] = config

    _client = boto3.client(**kwargs)
    return _client


def _object_url(key: str) -> str:
    if OBJECT_STORAGE_PUBLIC_BASE_URL:
        return f"{OBJECT_STORAGE_PUBLIC_BASE_URL}/{key}"
    if OBJECT_STORAGE_ENDPOINT_URL:
        return f"{OBJECT_STORAGE_ENDPOINT_URL.rstrip('/')}/{OBJECT_STORAGE_BUCKET}/{key}"
    return f"s3://{OBJECT_STORAGE_BUCKET}/{key}"


def upload_ticket_pdf(pdf_path: str, queue_number: int) -> dict | None:
    c = client()
    if c is None:
        return None

    path = Path(pdf_path)
    if not path.exists():
        print(f"[ObjectStorage] Ticket PDF does not exist: {pdf_path}")
        return None

    key_parts = []
    if OBJECT_STORAGE_PREFIX:
        key_parts.append(OBJECT_STORAGE_PREFIX)
    key_parts.append(f"Q{queue_number:03d}")
    key_parts.append(path.name)
    key = "/".join(key_parts)

    try:
        c.upload_file(
            Filename=str(path),
            Bucket=OBJECT_STORAGE_BUCKET,
            Key=key,
            ExtraArgs={"ContentType": "application/pdf"},
        )
    except (BotoCoreError, ClientError) as exc:
        print(f"[ObjectStorage] Ticket upload failed: {exc}")
        return None

    url = _object_url(key)
    print(f"[ObjectStorage] Ticket uploaded: {url}")
    return {"storage_key": key, "storage_url": url}
