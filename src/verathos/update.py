"""Self-update: a miner or validator follows the releases of the repository its code was installed from.

A release is a commit tagged ``v<major>.<minor>.<patch>`` (``verathos.release``) whose tag is signed by a release key:
one that ``release/allowed_signers`` of the running code lists (``SIGNERS``), as ``git verify-tag`` checks SSH
signatures. A release that changes the keys is therefore signed by a key of the release before it. Once a few minutes
after the start, then every ``CHECK_S``, the updater lists the tags of the checkout's ``origin``; the newest release
above the running version whose tag verifies (``fetch``) is planned at a random time within ``STAGGER_S``, so a fleet
does not restart at once. A tag that does not verify (unsigned, signed by another key or with GPG, or the tag object
of another tag) is logged and not taken until the next start; an older release that verifies is taken instead. The
process applies the update where nothing is in flight (``at_safe_point``: a validator between epochs, a miner outside
throughput tests and measurements): it fetches the tag, checks that its commit is the one listed, that the tag
verifies and that its code is the version it names, checks the commit out, installs it into the running environment
as the installer does (``EXTRAS``), and exits with ``EXIT_UPDATED``; the process manager (systemd, supervisord or
pm2, which restart a process that exits) starts the new release. Nothing changes, and the reason is logged once,
while the checkout has local changes, lists no release key (only its placeholder), or the origin cannot be read. A
failed install puts the previous commit back, and that release is not tried again until the next start. Engines and
references follow the catalog's pins, not this updater.
"""

from __future__ import annotations

import base64
import logging
import os
import random
import re
import shutil
import subprocess
import sys
import threading
import time
from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path

import verathos
from verathos import release

log = logging.getLogger("verathos.update")  # the processes log under their role's logger (``start``)
CODE = Path(verathos.__file__).resolve().parents[2]  # the checkout: <code>/src/verathos/__init__.py
EXTRAS = "server,chain"  # as scripts/install.sh installs the code
CHECK_S = 3600.0
FIRST_CHECK_S = (60.0, 600.0)  # the first check, a random time after the start
STAGGER_S = 3600.0
EXIT_UPDATED = 75
GIT_TIMEOUT_S = 120.0
INSTALL_TIMEOUT_S = 1800.0
VERSION_LINE = re.compile(r'^__version__\s*=\s*"([^"]+)"', re.M)
SIGNERS = Path("release") / "allowed_signers"  # the release keys, in the checkout (ssh-keygen's ALLOWED SIGNERS)
KEY = re.compile(r"(?:^|\s)(ssh-ed25519|ssh-rsa|ecdsa-sha2-nistp(?:256|384|521)|sk-ssh-ed25519@openssh\.com|"
                 r"sk-ecdsa-sha2-nistp256@openssh\.com)\s+([A-Za-z0-9+/]+={0,2})(?=\s|$)")
SIGNATURES = {"-----BEGIN PGP SIGNATURE-----": "GPG", "-----BEGIN PGP MESSAGE-----": "GPG",
              "-----BEGIN SIGNED MESSAGE-----": "X.509", "-----BEGIN SSH SIGNATURE-----": "SSH"}  # as git tells them
GOOD = 'Good "git" signature for '  # git verify-tag: a valid signature by a key the allowed signers list
GIT_SIGNING = (2, 34)  # the first git that verifies SSH signatures


class Refused(Exception):
    """Why the code is not updated now."""


class Untrusted(Refused):
    """A release tag that does not verify: not taken until the next start."""


class Moved(Refused):
    """A release tag that points elsewhere than when it was listed."""


@dataclass(frozen=True)
class Release:
    tag: str
    version: tuple[int, int, int]
    commit: str


def releases(ls_remote: str) -> list[Release]:
    """The release tags in ``git ls-remote --tags`` output, oldest first, each with its commit (an annotated tag's
    peeled one). Other tags (pre-releases, other names) are not releases."""
    direct: dict[str, str] = {}
    peeled: dict[str, str] = {}
    for line in ls_remote.splitlines():
        sha, _, ref = line.strip().partition("\t")
        if not ref.startswith("refs/tags/"):
            continue
        name = ref.removeprefix("refs/tags/")
        if name.endswith("^{}"):
            peeled[name.removesuffix("^{}")] = sha
        else:
            direct[name] = sha
    found = []
    for name, sha in direct.items():
        try:
            found.append(Release(name, release.parse(name, release.TAG), peeled.get(name, sha)))
        except ValueError:
            continue
    return sorted(found, key=lambda r: r.version)


def newer(found: list[Release], running: tuple[int, int, int]) -> list[Release]:
    """The releases above ``running``, newest first."""
    return sorted((r for r in found if r.version > running), key=lambda r: r.version, reverse=True)


def signing_keys(allowed_signers: str) -> list[str]:
    """The public keys (``<type> <base64>``) an allowed signers file lists. A line without a valid key, such as the
    placeholder, lists none."""
    keys = []
    for line in allowed_signers.splitlines():
        if line.lstrip().startswith("#"):
            continue
        for kind, blob in KEY.findall(line):
            try:
                raw = base64.b64decode(blob, validate=True)
            except ValueError:
                continue
            size = int.from_bytes(raw[:4], "big")
            if raw[4:4 + size] == kind.encode() and len(raw) > 4 + size:  # the blob names its type, as ssh reads it
                keys.append(f"{kind} {blob}")
    return keys


def signature(tag_object: str) -> str | None:
    """The kind of signature (SSH, GPG or X.509) git verifies on a tag object: its last one; None if unsigned."""
    kind = None
    for line in tag_object.splitlines():
        kind = next((name for start, name in SIGNATURES.items() if line.startswith(start)), kind)
    return kind


def version_at(init_py: str) -> tuple[int, int, int]:
    """The version ``src/verathos/__init__.py`` names. Raises Refused."""
    match = VERSION_LINE.search(init_py)
    try:
        return release.parse(match[1] if match else "")
    except ValueError:
        raise Refused("its src/verathos/__init__.py names no release version") from None


def dotted(version: tuple[int, int, int]) -> str:
    return ".".join(map(str, version))


def git(code: Path, *args: str) -> str:
    """``git -C code args``' output, never asking for credentials. Raises Refused."""
    try:
        result = subprocess.run(["git", "-C", str(code), *args], capture_output=True, text=True,
                                timeout=GIT_TIMEOUT_S, env={**os.environ, "GIT_TERMINAL_PROMPT": "0"})
    except (OSError, subprocess.TimeoutExpired) as exc:
        raise Refused(f"git {args[0]}: {exc}") from None
    if result.returncode != 0:
        lines = result.stderr.strip().splitlines()
        raise Refused(f"git {args[0]} failed: {lines[-1] if lines else result.returncode}")
    return result.stdout


def verify_tag(code: Path, tag_object: str, signers: Path) -> str:
    """Who signed a tag object (``<principal> with <key>``), as ``git verify-tag`` finds it with ``signers`` as the
    allowed signers file; GPG and X.509 signatures never verify. Raises Untrusted with git's last line (or Refused:
    git did not run)."""
    command = ["git", "-C", str(code), "-c", f"gpg.ssh.allowedSignersFile={signers}",
               "-c", "gpg.ssh.program=ssh-keygen", "-c", "gpg.program=false", "-c", "gpg.x509.program=false",
               "verify-tag", tag_object]
    try:
        result = subprocess.run(command, capture_output=True, text=True, timeout=GIT_TIMEOUT_S,
                                env={**os.environ, "GIT_TERMINAL_PROMPT": "0"})
    except (OSError, subprocess.TimeoutExpired) as exc:
        raise Refused(f"git verify-tag: {exc}") from None
    lines = (result.stdout + result.stderr).strip().splitlines()
    good = [line.removeprefix(GOOD) for line in lines if line.startswith(GOOD)]
    if result.returncode != 0 or not good:
        raise Untrusted(lines[-1] if lines else f"git verify-tag exited with {result.returncode}")
    return good[0]


def install(code: Path) -> None:
    """The checked-out code into the running environment, with the installer's extras. Raises Refused."""
    from verathos.install.engines import uv

    try:
        command = [uv(), "pip", "install", "--quiet", "--python", sys.executable, "-e", f"{code}[{EXTRAS}]"]
        result = subprocess.run(command, capture_output=True, text=True, timeout=INSTALL_TIMEOUT_S)
    except (SystemExit, OSError, subprocess.TimeoutExpired) as exc:
        raise Refused(f"uv pip install: {exc}") from None
    if result.returncode != 0:
        lines = result.stderr.strip().splitlines()
        raise Refused(f"uv pip install failed: {lines[-1] if lines else result.returncode}")


def exit_for_restart(code: int) -> None:
    for handler in logging.getLogger().handlers:
        handler.flush()
    sys.stdout.flush()
    sys.stderr.flush()
    os._exit(code)


class Updater:
    """Plans (``check``, in the background) and applies (``at_safe_point``) the update to the newest release; see the
    module. ``git``, ``install``, ``exit``, ``clock`` and ``rng`` are the outside world, replaced in tests."""

    def __init__(self, *, code: Path = CODE, running: str = verathos.__version__,
                 git: Callable[..., str] = git, install: Callable[[Path], None] = install,
                 exit: Callable[[int], None] = exit_for_restart, clock: Callable[[], float] = time.monotonic,
                 rng: random.Random | None = None, logger: logging.Logger = log) -> None:
        self.code, self.running, self.log = code, release.parse(running), logger
        self.git, self.install, self.exit, self.clock = git, install, exit, clock
        self.rng = rng or random.Random()
        self.pending: Release | None = None
        self.due_at = 0.0
        self.failed: set[str] = set()  # tags that did not verify or install: not tried again by this process
        self._said: str | None = None
        self._lock = threading.Lock()

    def say(self, why: str) -> None:
        """Log why the code is not updated, once until the reason changes."""
        if why != self._said:
            self.log.warning("self-update: %s", why)
            self._said = why

    def clean(self) -> None:
        """Raises Refused if a tracked file of the checkout has local changes."""
        if self.git(self.code, "status", "--porcelain", "--untracked-files=no").strip():
            raise Refused(f"local changes in {self.code}: not updating (see git -C {self.code} status)")

    def signers(self) -> Path:
        """The checkout's release keys (``SIGNERS``), once they list a key and git can verify with them. Raises
        Refused."""
        path = self.code / SIGNERS
        try:
            keys = signing_keys(path.read_text())
        except OSError:
            raise Refused(f"{path} is missing: no release can be verified, so none is installed; update with the "
                          "installer") from None
        if not keys:
            raise Refused(f"{path} lists no release signing key (only a placeholder): no release can be verified, so "
                          "none is installed by itself; run the installer again to update")
        if shutil.which("ssh-keygen") is None:
            raise Refused("ssh-keygen, which verifies release signatures, is missing: install openssh-client")
        found = re.search(r"(\d+)\.(\d+)", self.git(self.code, "--version"))
        if found is None or (int(found[1]), int(found[2])) < GIT_SIGNING:
            raise Refused(f"git {found[0] if found else '?'} does not verify SSH signatures: release tags need git "
                          f"{GIT_SIGNING[0]}.{GIT_SIGNING[1]} or newer")
        return path

    def fetch(self, target: Release, signers: Path) -> tuple[str, str]:
        """Fetch ``target``'s tag: its commit and signer, once the commit is the one listed and the tag verifies, the
        tag object of ``target.tag`` signed by a key in ``signers``. Its objects stay inert until checked out. Raises
        Untrusted, Moved or Refused."""
        self.git(self.code, "fetch", "--quiet", "--no-tags", "origin", f"refs/tags/{target.tag}")
        tag = self.git(self.code, "rev-parse", "FETCH_HEAD").strip()
        commit = self.git(self.code, "rev-parse", f"{tag}^{{commit}}").strip()
        if commit != target.commit:
            raise Moved(f"{target.tag} is at {commit[:12]} now, not {target.commit[:12]} as listed")
        if self.git(self.code, "cat-file", "-t", tag).strip() != "tag":
            raise Untrusted(f"{target.tag} is not signed (a lightweight tag)")
        content = self.git(self.code, "cat-file", "tag", tag)
        if f"tag {target.tag}" not in content.partition("\n\n")[0].splitlines():
            raise Untrusted(f"{target.tag} names the tag object of another tag")
        kind = signature(content)
        if kind != "SSH":
            raise Untrusted(f"{target.tag} is not signed" if kind is None else
                            f"{target.tag} is signed with {kind}, not with a release key (SSH)")
        try:
            return commit, verify_tag(self.code, tag, signers)
        except Untrusted as exc:
            raise Untrusted(f"{target.tag} is not signed by a key {SIGNERS} lists ({exc})") from None

    def check(self) -> Release | None:
        """The newest release above the running version whose tag verifies, planned within ``STAGGER_S``; None if
        there is none. A tag that does not verify is logged and not taken again by this process. Raises Refused."""
        if not (self.code / ".git").exists():
            raise Refused(f"{self.code} is not a git checkout: update it with the installer")
        signers = self.signers()
        with self._lock:
            candidates = [r for r in newer(releases(self.git(self.code, "ls-remote", "--tags", "origin")),
                                           self.running) if r.tag not in self.failed]
            if candidates:
                self.clean()
            target, signer = None, ""
            for candidate in candidates:
                if candidate == self.pending:  # verified when it was planned
                    target = candidate
                    break
                try:
                    signer = self.fetch(candidate, signers)[1]
                except Untrusted as exc:
                    self.failed.add(candidate.tag)
                    self.log.warning("self-update: %s; not updating to it", exc)
                    continue
                target = candidate
                break
            if target != self.pending:
                self.pending = target
                if target is not None:
                    self.due_at = self.clock() + self.rng.uniform(0.0, STAGGER_S)
                    self.log.info("self-update: release %s is out (running %s, signed by %s); updating in about %.0f "
                                  "minutes, when nothing is in flight", target.tag, dotted(self.running), signer,
                                  (self.due_at - self.clock()) / 60)
            return target

    def due(self) -> bool:
        return self.pending is not None and self.clock() >= self.due_at

    def apply(self) -> None:
        """Update to the pending release and exit for the process manager to start it. Raises Refused: then the code
        is as it was."""
        with self._lock:
            target = self.pending
            if target is None:
                return
            self.clean()
            try:
                commit = self.fetch(target, self.signers())[0]
            except Moved:
                self.pending = None  # listed again at the next check
                raise
            except Untrusted:
                self.failed.add(target.tag)
                self.pending = None
                raise
            named = version_at(self.git(self.code, "show", f"{commit}:src/verathos/__init__.py"))
            if named != target.version:
                self.failed.add(target.tag)
                self.pending = None
                raise Refused(f"{target.tag} holds verathos {dotted(named)}, not the version it names; not updating "
                              "to it")
            previous = self.git(self.code, "rev-parse", "HEAD").strip()
            self.git(self.code, "checkout", "--quiet", "--detach", commit)
            try:
                self.install(self.code)
            except Exception as exc:  # the previous code back, whatever stopped the install
                self.failed.add(target.tag)
                self.pending = None
                try:
                    self.git(self.code, "checkout", "--quiet", "--detach", previous)
                    self.install(self.code)
                except Exception as again:
                    self.log.critical("self-update: putting %s back failed too (%s): reinstall with the installer",
                                      previous[:12], again)
                raise Refused(f"installing {target.tag} failed ({exc}); back on {previous[:12]}") from None
            self.log.warning("self-update: installed %s (%s, was %s at %s); exiting for the process manager to start "
                             "it", target.tag, commit[:12], dotted(self.running), previous[:12])
        self.exit(EXIT_UPDATED)

    def at_safe_point(self) -> None:
        """Called where nothing is in flight: apply the update once it is due (the process exits)."""
        if not self.due():
            return
        try:
            self.apply()
        except Refused as exc:
            self.say(str(exc))
        except Exception:
            self.log.exception("self-update failed")

    def poll(self) -> None:
        """One check; a refusal is logged once until it changes."""
        try:
            self.check()
        except Refused as exc:
            self.say(str(exc))
        except Exception:
            self.log.exception("self-update check failed; checking again in an hour")
        else:
            self._said = None

    def start(self) -> Updater:
        """Check in the background: a random time after the start, then every ``CHECK_S``."""
        def run() -> None:
            time.sleep(self.rng.uniform(*FIRST_CHECK_S))
            while True:
                self.poll()
                time.sleep(CHECK_S)

        threading.Thread(target=run, name="self-update", daemon=True).start()
        return self


def start(enabled: bool, role: str) -> Updater | None:
    """The updater of a ``role`` (miner, validator) whose configuration has ``auto_update``, checking in the
    background and logging under the role's logger; None (and why, logged) otherwise."""
    role_log = logging.getLogger(f"verathos.{role}")
    if not enabled:
        role_log.info("self-update off (auto_update): update the %s with the installer", role)
        return None
    if not (CODE / ".git").exists():
        role_log.warning("self-update: %s is not a git checkout; update the %s with the installer", CODE, role)
        return None
    role_log.info("self-update on: releases (v<version> tags signed by a key in %s) of the origin of %s, checked "
                  "hourly; running %s", SIGNERS, CODE, verathos.__version__)
    return Updater(logger=role_log).start()
