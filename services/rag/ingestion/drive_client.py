"""Read-only Google Drive access for legal PDF discovery and download."""

from __future__ import annotations

from contextlib import contextmanager
import os
from pathlib import Path
import tempfile
from typing import Iterator

from google.oauth2 import service_account
from google_auth_httplib2 import AuthorizedHttp
from googleapiclient.discovery import build
from googleapiclient.http import MediaIoBaseDownload
import httplib2


DRIVE_READONLY_SCOPE = "https://www.googleapis.com/auth/drive.readonly"
FOLDER_MIME_TYPE = "application/vnd.google-apps.folder"
PDF_MIME_TYPE = "application/pdf"


def load_env_file(path: str | Path, *, override: bool = False) -> None:
    """Load simple KEY=VALUE entries without echoing credentials."""
    env_path = Path(path)
    if not env_path.exists():
        return
    for raw in env_path.read_text(encoding="utf-8-sig").splitlines():
        line = raw.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        key, value = line.split("=", 1)
        key, value = key.strip(), value.strip()
        if len(value) > 1 and value[0] == value[-1] and value[0] in "\"'":
            value = value[1:-1]
        if override or key not in os.environ:
            os.environ[key] = value


class DriveClient:
    """Small injectable wrapper around Drive v3; only read operations are exposed."""

    def __init__(self, service):
        self.service = service

    @classmethod
    def from_env(cls, *, env_file: str | Path | None = None, timeout: int = 120):
        if env_file:
            load_env_file(env_file)
        credential_file = os.getenv("GOOGLE_APPLICATION_CREDENTIALS")
        if not credential_file or not Path(credential_file).is_file():
            raise FileNotFoundError(
                "Set GOOGLE_APPLICATION_CREDENTIALS to an accessible service-account JSON file."
            )
        credentials = service_account.Credentials.from_service_account_file(
            credential_file, scopes=[DRIVE_READONLY_SCOPE]
        )
        http = AuthorizedHttp(credentials, http=httplib2.Http(timeout=timeout))
        return cls(build("drive", "v3", http=http, cache_discovery=False))

    def list_children(self, folder_id: str) -> list[dict]:
        result: list[dict] = []
        token = None
        while True:
            response = self.service.files().list(
                q=f"'{folder_id}' in parents and trashed = false",
                pageSize=1000,
                pageToken=token,
                fields="nextPageToken,files(id,name,mimeType,size,modifiedTime,md5Checksum,parents,webViewLink)",
                supportsAllDrives=True,
                includeItemsFromAllDrives=True,
            ).execute()
            result.extend(response.get("files", []))
            token = response.get("nextPageToken")
            if not token:
                return result

    def discover_pdfs(self, folder_id: str) -> tuple[list[dict], list[dict]]:
        queue, seen = [folder_id], set()
        pdfs, errors = [], []
        while queue:
            current = queue.pop(0)
            if current in seen:
                continue
            seen.add(current)
            try:
                children = self.list_children(current)
            except Exception as exc:
                errors.append({"folder_id": current, "error": f"{type(exc).__name__}: {exc}"})
                continue
            for entry in children:
                if entry.get("mimeType") == FOLDER_MIME_TYPE:
                    queue.append(entry["id"])
                elif entry.get("mimeType") == PDF_MIME_TYPE or entry.get("name", "").lower().endswith(".pdf"):
                    pdfs.append(entry)
        pdfs.sort(key=lambda item: item.get("name", "").casefold())
        return pdfs, errors

    def download(self, file_id: str, destination: str | Path) -> Path:
        path = Path(destination)
        request = self.service.files().get_media(fileId=file_id, supportsAllDrives=True)
        with path.open("wb") as handle:
            downloader = MediaIoBaseDownload(handle, request, chunksize=1024 * 1024)
            done = False
            while not done:
                _, done = downloader.next_chunk()
        return path

    @contextmanager
    def temporary_pdf(self, file_id: str) -> Iterator[Path]:
        with tempfile.TemporaryDirectory(prefix="legal_pdf_") as directory:
            path = Path(directory) / "source.pdf"
            yield self.download(file_id, path)
