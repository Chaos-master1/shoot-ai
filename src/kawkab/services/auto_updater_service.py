from __future__ import annotations

import hashlib
import json
import os
import platform
import re
import subprocess
import tempfile
from dataclasses import dataclass
from pathlib import Path
from urllib.parse import urlparse

import httpx

from kawkab.core.logging import get_logger

logger = get_logger(__name__)

# Only these hosts are ever trusted as a download source or redirect target.
# GitHub release assets are requested from github.com and typically 302 to
# objects.githubusercontent.com for the actual binary.
_ALLOWED_DOWNLOAD_HOSTS = {"github.com", "objects.githubusercontent.com"}


@dataclass
class ReleaseInfo:
    version: str
    download_url: str
    release_notes: str
    published_at: str
    is_prerelease: bool = False


class AutoUpdaterService:
    def __init__(self, repo: str = "wizard404error/shoot-ai", current_version: str = "0.13.0"):
        self.repo = repo
        self.current_version = current_version
        self._check_url = f"https://api.github.com/repos/{repo}/releases/latest"
        # Set by download_update() only after the source host (and
        # checksum, when GitHub's API provides one) has been verified.
        # apply_update() refuses to launch anything else -- the bridge
        # layer passes both the url and path straight through from JS
        # with no validation of its own (the QWebChannel bridge object is
        # globally reachable from the page's JS context), so this service
        # cannot assume its callers already checked anything.
        self._verified_download: Path | None = None

    def check_for_update(self) -> str:
        try:
            resp = httpx.get(
                self._check_url, timeout=10.0, headers={"Accept": "application/vnd.github.v3+json"}
            )
            if resp.status_code != 200:
                return json.dumps(
                    {"error": f"GitHub API returned {resp.status_code}", "has_update": False}
                )
            data = resp.json()
            latest_tag = data.get("tag_name", "").lstrip("v")
            if self._compare_versions(latest_tag, self.current_version) > 0:
                assets = data.get("assets", [])
                download_url = ""
                digest = ""
                system = platform.system().lower()
                for asset in assets:
                    name = asset.get("name", "").lower()
                    if (
                        system == "windows"
                        and (name.endswith(".exe") or "setup" in name)
                        or system == "darwin"
                        and name.endswith(".dmg")
                    ):
                        download_url = asset["browser_download_url"]
                        digest = asset.get("digest") or ""
                        break
                return json.dumps(
                    {
                        "has_update": True,
                        "version": latest_tag,
                        "download_url": download_url,
                        # GitHub's release-assets API includes a "sha256:<hex>"
                        # digest for each asset. Pass it back so the caller can
                        # hand it to download_update() for real verification;
                        # empty if GitHub didn't provide one for this asset.
                        "digest": digest,
                        "release_notes": data.get("body", ""),
                        "published_at": data.get("published_at", ""),
                    }
                )
            return json.dumps({"has_update": False, "version": self.current_version})
        except Exception as e:
            logger.error(f"check_for_update failed: {e}")
            return json.dumps({"error": str(e), "has_update": False})

    def download_update(self, download_url: str, expected_digest: str = "") -> str:
        """Download an update installer.

        Verifies the source host (and every redirect hop) against a fixed
        allowlist, and the downloaded file's SHA-256 against
        expected_digest when the caller has one (from check_for_update()'s
        "digest" field). Previously this followed arbitrary redirects with
        no host check and never verified the download at all -- a
        compromised/spoofed download_url (this method's only input,
        passed straight through from JS with no validation at the bridge
        layer) could make the app fetch and, via apply_update(), execute
        anything.
        """
        try:
            host = (urlparse(download_url).hostname or "").lower()
            if host not in _ALLOWED_DOWNLOAD_HOSTS:
                logger.error(f"Refusing to download update from untrusted host: {host or '(none)'}")
                return json.dumps(
                    {"error": f"Refusing to download from untrusted host: {host or '(none)'}"}
                )

            raw_name = download_url.rsplit("/", 1)[-1] or "update"
            # Keep only a safe basename regardless of what the URL
            # contains -- no path separators or traversal sequences can
            # reach the filesystem call below.
            safe_name = re.sub(r"[^A-Za-z0-9._-]", "_", os.path.basename(raw_name)) or "update"
            if safe_name in (".", ".."):
                safe_name = "update"

            dest = Path(tempfile.gettempdir()) / safe_name
            logger.info(f"Downloading update from {download_url} to {dest}")

            hasher = hashlib.sha256()
            downloaded = 0
            current_url = download_url
            with httpx.Client(follow_redirects=False, timeout=300.0) as http_client:
                for _ in range(5):  # bounded chain; each hop re-validated below
                    resp = http_client.get(current_url)
                    if resp.is_redirect:
                        location = resp.headers.get("location", "")
                        next_host = (urlparse(location).hostname or "").lower()
                        if next_host not in _ALLOWED_DOWNLOAD_HOSTS:
                            logger.error(
                                f"Refusing to follow update redirect to untrusted host: {next_host or '(none)'}"
                            )
                            return json.dumps(
                                {
                                    "error": f"Refusing to follow redirect to untrusted host: {next_host or '(none)'}"
                                }
                            )
                        current_url = location
                        continue
                    resp.raise_for_status()
                    with open(dest, "wb") as f:
                        for chunk in resp.iter_bytes(chunk_size=8192):
                            f.write(chunk)
                            hasher.update(chunk)
                            downloaded += len(chunk)
                    break
                else:
                    return json.dumps({"error": "Too many redirects while downloading update"})

            actual_digest = f"sha256:{hasher.hexdigest()}"
            if expected_digest:
                if actual_digest != expected_digest:
                    dest.unlink(missing_ok=True)
                    logger.error(
                        f"Update checksum mismatch: expected {expected_digest}, got {actual_digest}"
                    )
                    return json.dumps(
                        {"error": "Checksum verification failed -- download discarded"}
                    )
            else:
                logger.warning(
                    "download_update: no expected checksum supplied -- integrity "
                    "was verified by source host only, not by content hash."
                )

            self._verified_download = dest
            return json.dumps(
                {"ok": True, "path": str(dest), "size": downloaded, "digest": actual_digest}
            )
        except Exception as e:
            logger.error(f"download_update failed: {e}")
            return json.dumps({"error": str(e)})

    def apply_update(self, installer_path: str) -> str:
        """Launch a downloaded installer.

        Only ever launches the exact path this same service instance just
        produced via download_update() -- installer_path is compared
        against that, not trusted on its own. Previously this launched
        whatever path the caller supplied with no relationship to an
        actual verified download required at all, via
        `subprocess.Popen([...], shell=True)` (Windows/Linux branches);
        shell=True is unneeded for a list-form Popen and was dropped.
        """
        try:
            if self._verified_download is None or Path(installer_path) != self._verified_download:
                logger.error(
                    f"apply_update refused: {installer_path!r} does not match the "
                    f"last verified download_update() result ({self._verified_download!r})"
                )
                return json.dumps(
                    {"error": "This path was not produced by a verified download_update() call"}
                )

            target = str(self._verified_download)
            system = platform.system().lower()
            if system == "windows":
                subprocess.Popen([target, "/SILENT", "/SUPPRESSMSGBOXES"])
            elif system == "darwin":
                subprocess.Popen(["open", target])
            else:
                subprocess.Popen(["xdg-open", target])
            return json.dumps({"ok": True, "message": "Installer launched"})
        except Exception as e:
            logger.error(f"apply_update failed: {e}")
            return json.dumps({"error": str(e)})

    def get_current_version(self) -> str:
        return json.dumps(
            {
                "version": self.current_version,
                "platform": platform.system(),
                "arch": platform.machine(),
            }
        )

    @staticmethod
    def _compare_versions(v1: str, v2: str) -> int:
        parts1 = [int(x) for x in v1.split(".") if x.isdigit()]
        parts2 = [int(x) for x in v2.split(".") if x.isdigit()]
        for a, b in zip(parts1, parts2, strict=False):
            if a > b:
                return 1
            if a < b:
                return -1
        return len(parts1) - len(parts2)
