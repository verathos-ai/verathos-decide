"""The self-update: which release it takes, when it waits or refuses, and how it updates a checkout (a real git
repository as the origin, release tags signed with throwaway SSH keys; the install and the exit replaced); the
installer's check of the tags it installs."""

from __future__ import annotations

import argparse
import base64
import logging
import os
import random
import re
import shutil
import subprocess
from pathlib import Path
from types import SimpleNamespace

import pytest

from verathos import update
from verathos.install import common
from verathos.install.miner import miner_config
from verathos.install.validator import validator_config
from verathos.miner.__main__ import Publisher
from verathos.miner.config import MinerConfig
from verathos.miner.timing import Deferral
from verathos.validator import light
from verathos.validator.config import ValidatorConfig

ROOT = Path(__file__).resolve().parents[1]


@pytest.fixture(autouse=True)
def updater_log(caplog) -> None:
    caplog.set_level(logging.INFO, logger="verathos.update")


@pytest.fixture(autouse=True)
def plain_git(monkeypatch) -> None:
    """git without the user's configuration, with an identity for the commits the tests make."""
    for key, value in {"GIT_CONFIG_GLOBAL": "/dev/null", "GIT_CONFIG_NOSYSTEM": "1", "GIT_AUTHOR_NAME": "t",
                       "GIT_AUTHOR_EMAIL": "t@example.org", "GIT_COMMITTER_NAME": "t",
                       "GIT_COMMITTER_EMAIL": "t@example.org"}.items():
        monkeypatch.setenv(key, value)


SHIPPED_SIGNERS = (ROOT / update.SIGNERS).read_text()


def git(cwd: Path, *args: str, stdin: str | None = None) -> str:
    return subprocess.run(["git", *args], cwd=cwd, check=True, capture_output=True, text=True,
                          input=stdin).stdout.strip()


@pytest.fixture(scope="module")
def keys(tmp_path_factory) -> Path:
    """Throwaway SSH keys: ``release``, ``other`` and ``next`` (and their ``.pub``)."""
    found = re.search(r"(\d+)\.(\d+)", subprocess.run(["git", "--version"], capture_output=True, text=True).stdout)
    if shutil.which("ssh-keygen") is None or found is None or (int(found[1]), int(found[2])) < update.GIT_SIGNING:
        pytest.skip("git here cannot sign or verify SSH signatures (git 2.34 and ssh-keygen needed)")
    path = tmp_path_factory.mktemp("keys")
    for name in ("release", "other", "next"):
        subprocess.run(["ssh-keygen", "-q", "-t", "ed25519", "-N", "", "-C", name, "-f", str(path / name)], check=True)
    return path


def allowed(keys: Path, *names: str) -> str:
    """An allowed signers file listing the keys ``names`` for release tags."""
    return "".join(f'verathos-release namespaces="git" {" ".join((keys / f"{n}.pub").read_text().split()[:2])}\n'
                   for n in names)


def commit(origin: Path, version: str, tag: str | None = None, *, key: Path | None = None, annotated: bool = False,
           names: str | None = None, signers: str | None = None) -> str:
    """A commit of code naming ``names`` (default ``version``) with the allowed signers ``signers`` (default: as
    before), tagged ``tag``: signed with ``key``, else annotated or lightweight."""
    init = origin / "src" / "verathos" / "__init__.py"
    init.parent.mkdir(parents=True, exist_ok=True)
    init.write_text(f'"""The package, release {version}."""\n\n__version__ = "{names or version}"\n')
    if signers is not None:
        (origin / update.SIGNERS).parent.mkdir(exist_ok=True)
        (origin / update.SIGNERS).write_text(signers)
    git(origin, "add", "-A")
    git(origin, "commit", "-q", "-m", f"release {version}")
    if key:
        git(origin, "-c", "gpg.format=ssh", "-c", f"user.signingkey={key}", "tag", "-s", "-m", tag, tag)
    elif tag:
        git(origin, "tag", *(["-a", "-m", tag] if annotated else []), tag)
    return git(origin, "rev-parse", "HEAD")


def forged(origin: Path, tag: str, like: str, *, version: str | None = None, signature: str | None = None) -> str:
    """Tag ``tag``: a copy of the tag object of ``like`` (its signature with it), on a new commit of code naming
    ``version`` and naming ``tag`` if given, else ``like``'s commit and name; ``signature`` replaces the signature."""
    content = git(origin, "cat-file", "tag", like) + "\n"
    if version:
        target = commit(origin, version)
        content = re.sub(r"^object \w+$", f"object {target}", content, count=1, flags=re.M)
        content = content.replace(f"\ntag {like}\n", f"\ntag {tag}\n", 1)
    if signature:
        content = content[:content.index("-----BEGIN ")] + signature
    obj = git(origin, "hash-object", "-t", "tag", "-w", "--stdin", stdin=content)
    git(origin, "update-ref", f"refs/tags/{tag}", obj)
    return obj


class World:
    """An origin with releases 1.0.0 and 1.0.1, both signed by the release key that the allowed signers of their code
    list (or ``signers``), and a checkout of 1.0.0 with an updater."""

    def __init__(self, tmp_path: Path, keys: Path, install_fails: int = 0, signers: str | None = None) -> None:
        self.origin, self.keys = tmp_path / "origin", keys
        self.origin.mkdir(parents=True)
        git(self.origin, "init", "-q", "-b", "main")
        signers = allowed(keys, "release") if signers is None else signers
        self.v100 = commit(self.origin, "1.0.0", "v1.0.0", key=keys / "release", signers=signers)
        self.v101 = commit(self.origin, "1.0.1", "v1.0.1", key=keys / "release")
        git(self.origin.parent, "clone", "-q", str(self.origin), "code")
        self.code = self.origin.parent / "code"
        git(self.code, "checkout", "-q", "--detach", "v1.0.0")
        self.now, self.installs, self.exits, self.install_fails = 0.0, [], [], install_fails
        self.updater = update.Updater(code=self.code, running="1.0.0", install=self.install, exit=self.exits.append,
                                      clock=lambda: self.now, rng=random.Random(7))

    def install(self, code: Path) -> None:
        self.installs.append(git(code, "rev-parse", "HEAD"))
        if self.install_fails:
            self.install_fails -= 1
            raise update.Refused("uv pip install failed: no matching distribution")

    def head(self) -> str:
        return git(self.code, "rev-parse", "HEAD")


def test_releases_are_the_version_tags_and_the_newest_above_the_running_version_is_taken() -> None:
    listed = "\n".join(["a1\trefs/tags/v1.0.0", "a2\trefs/tags/v1.0.0^{}", "b1\trefs/tags/v1.9.0",
                        "c1\trefs/tags/v1.10.0", "d1\trefs/tags/v1.11.0-rc1", "e1\trefs/tags/latest",
                        "f1\trefs/heads/main", "g1\trefs/tags/1.12.0"])
    found = update.releases(listed)
    assert [(r.tag, r.commit) for r in found] == [("v1.0.0", "a2"), ("v1.9.0", "b1"), ("v1.10.0", "c1")]
    assert [r.tag for r in update.newer(found, (1, 0, 0))] == ["v1.10.0", "v1.9.0"]  # by number, not by name
    assert [r.tag for r in update.newer(found, (1, 9, 5))] == ["v1.10.0"]
    assert update.newer(found, (1, 10, 0)) == [] and update.newer(found, (2, 0, 0)) == []


def test_an_update_waits_for_its_time_then_takes_the_release_installs_it_and_exits(tmp_path, keys, caplog) -> None:
    w = World(tmp_path, keys)
    target = w.updater.check()
    assert (target.tag, target.commit) == ("v1.0.1", w.v101)
    assert 0 < w.updater.due_at <= update.STAGGER_S and not w.updater.due()  # a random time, not at once
    assert any("release v1.0.1 is out (running 1.0.0, signed by verathos-release with ED25519 key SHA256:" in m
               for m in caplog.messages)
    w.updater.at_safe_point()
    assert w.head() == w.v100 and w.installs == [] and w.exits == []
    w.now = w.updater.due_at
    w.updater.check()  # listed again: the time stays
    assert w.updater.due()
    w.updater.at_safe_point()
    assert w.head() == w.v101 and w.installs == [w.v101] and w.exits == [update.EXIT_UPDATED]
    restarted = update.Updater(code=w.code, running="1.0.1", git=update.git)
    assert restarted.check() is None and not restarted.due()  # the newest release runs


def test_local_changes_keep_the_code_as_it_is_and_are_logged_once(tmp_path, keys, caplog) -> None:
    w = World(tmp_path, keys)
    (w.code / "src" / "verathos" / "__init__.py").write_text('__version__ = "1.0.0"\n# edited\n')
    for _ in range(3):
        w.updater.poll()
    refusals = [m for m in caplog.messages if "local changes in" in m]
    assert len(refusals) == 1 and w.updater.pending is None and w.head() == w.v100
    git(w.code, "checkout", "--", ".")
    (w.code / "notes.txt").write_text("an untracked file")  # not a change of the code
    w.updater.poll()
    assert w.updater.pending.tag == "v1.0.1"
    (w.code / "src" / "verathos" / "__init__.py").write_text('__version__ = "1.0.0"\n# edited again\n')
    w.now = update.STAGGER_S
    w.updater.at_safe_point()  # changed after the check: still refused
    assert w.head() == w.v100 and w.exits == [] and "local changes in" in caplog.messages[-1]


def test_a_tag_whose_code_names_another_version_is_never_taken(tmp_path, keys, caplog) -> None:
    w = World(tmp_path, keys)
    commit(w.origin, "1.0.2", "v1.0.2", key=keys / "release", names="1.0.1")  # it would restart into an update again
    assert w.updater.check().tag == "v1.0.2"
    w.now = update.STAGGER_S
    w.updater.at_safe_point()
    assert w.head() == w.v100 and w.exits == [] and w.installs == []
    assert "v1.0.2 holds verathos 1.0.1, not the version it names" in caplog.messages[-1]
    assert w.updater.check().tag == "v1.0.1"  # the newest release that is what it says


def test_a_failed_install_puts_the_previous_commit_back(tmp_path, keys, caplog) -> None:
    w = World(tmp_path, keys, install_fails=1)
    w.updater.check()
    w.now = update.STAGGER_S
    w.updater.at_safe_point()
    assert w.head() == w.v100 and w.installs == [w.v101, w.v100] and w.exits == []
    assert f"installing v1.0.1 failed (uv pip install failed: no matching distribution); back on {w.v100[:12]}" in \
        caplog.messages[-1]
    assert w.updater.check() is None  # not tried again until the next start


def test_a_checkout_without_git_release_keys_or_an_origin_is_not_updated(tmp_path, keys) -> None:
    (tmp_path / "plain").mkdir()
    with pytest.raises(update.Refused, match="not a git checkout"):
        update.Updater(code=tmp_path / "plain", running="1.0.0").check()
    git(tmp_path, "init", "-q", "lone")
    with pytest.raises(update.Refused, match="release/allowed_signers is missing: no release can be verified"):
        update.Updater(code=tmp_path / "lone", running="1.0.0").check()
    (tmp_path / "lone" / "release").mkdir()
    (tmp_path / "lone" / update.SIGNERS).write_text(allowed(keys, "release"))
    with pytest.raises(update.Refused, match="git ls-remote failed"):
        update.Updater(code=tmp_path / "lone", running="1.0.0").check()
    assert update.start(False, "validator") is None


PGP = "-----BEGIN PGP SIGNATURE-----\n\niQEzBAABCAAdFiEE\n-----END PGP SIGNATURE-----\n"
REFUSED = "; not updating to it"


def test_the_release_keys_are_the_valid_public_keys_the_allowed_signers_list() -> None:
    def blob(kind: str, size: int = 32) -> str:
        raw = len(kind).to_bytes(4, "big") + kind.encode() + size.to_bytes(4, "big") + bytes(range(size))
        return base64.b64encode(raw).decode()

    ed, sk = blob("ssh-ed25519"), blob("sk-ssh-ed25519@openssh.com")
    lines = [f"# a comment: release ssh-ed25519 {ed}",
             f'release namespaces="git",valid-after="20260101" ssh-ed25519 {ed} owner@laptop',
             f"hardware sk-ssh-ed25519@openssh.com {sk}",
             f"mislabelled ssh-rsa {ed}",  # the blob is another type's
             "broken ssh-ed25519 AAAAnot+base64!",
             f"truncated ssh-ed25519 {base64.b64encode(blob('ssh-ed25519').encode()[:15]).decode()}",
             "release ssh-ed25519 REPLACE_WITH_THE_RELEASE_SIGNING_PUBLIC_KEY"]
    assert update.signing_keys("\n".join(lines)) == [f"ssh-ed25519 {ed}", f"sk-ssh-ed25519@openssh.com {sk}"]
    assert update.signing_keys(SHIPPED_SIGNERS) == [] and "PLACEHOLDER" in SHIPPED_SIGNERS
    tag = "object 1f\ntype commit\ntag v1.0.1\n\nv1.0.1\n"
    assert update.signature(tag) is None
    assert update.signature(tag + "-----BEGIN SSH SIGNATURE-----\nU1NI\n-----END SSH SIGNATURE-----\n") == "SSH"
    assert update.signature(tag + "-----BEGIN SSH SIGNATURE-----\n" + PGP) == "GPG"  # the last one, as git reads it


def test_only_a_tag_signed_by_a_listed_key_is_taken_and_each_refusal_is_logged_once(tmp_path, keys, caplog) -> None:
    w = World(tmp_path, keys)
    commit(w.origin, "1.0.2", "v1.0.2")
    commit(w.origin, "1.0.3", "v1.0.3", annotated=True)
    commit(w.origin, "1.0.4", "v1.0.4", key=keys / "other")
    forged(w.origin, "v1.0.5", "v1.0.1")  # v1.0.1's signed tag object under another name
    forged(w.origin, "v1.0.6", "v1.0.1", version="1.0.6")  # its signature on another commit and name: tampered
    forged(w.origin, "v1.0.7", "v1.0.1", version="1.0.7", signature=PGP)
    assert w.updater.check().tag == "v1.0.1"  # the newest release whose tag verifies
    refusals = [m.removeprefix("self-update: ").removesuffix(REFUSED) for m in caplog.messages if REFUSED in m]
    assert refusals[0] == "v1.0.7 is signed with GPG, not with a release key (SSH)"
    assert refusals[1].startswith("v1.0.6 is not signed by a key release/allowed_signers lists (")
    assert refusals[2:] == ["v1.0.5 names the tag object of another tag",
                            "v1.0.4 is not signed by a key release/allowed_signers lists (No principal matched.)",
                            "v1.0.3 is not signed", "v1.0.2 is not signed (a lightweight tag)"]
    assert w.updater.check().tag == "v1.0.1"
    assert sum(REFUSED in m for m in caplog.messages) == 6  # not tried again until the next start
    w.now = update.STAGGER_S
    w.updater.at_safe_point()
    assert w.head() == w.v101 and w.installs == [w.v101] and w.exits == [update.EXIT_UPDATED]


def test_a_tag_replaced_after_it_was_planned_is_verified_again_before_it_is_installed(tmp_path, keys, caplog) -> None:
    w = World(tmp_path, keys)
    assert w.updater.check().tag == "v1.0.1"
    git(w.origin, "tag", "-f", "-a", "-m", "v1.0.1", "v1.0.1", w.v101)  # unsigned, on the same commit
    w.now = update.STAGGER_S
    w.updater.at_safe_point()
    assert w.head() == w.v100 and w.installs == [] and w.exits == []
    assert caplog.messages[-1] == "self-update: v1.0.1 is not signed"
    assert w.updater.check() is None  # not taken until the next start


def test_a_release_rotating_the_key_is_signed_by_the_old_one_and_nodes_follow_it_to_the_new_one(tmp_path, keys,
                                                                                                 caplog) -> None:
    w = World(tmp_path, keys)
    v102 = commit(w.origin, "1.0.2", "v1.0.2", key=keys / "release", signers=allowed(keys, "next"))
    v103 = commit(w.origin, "1.0.3", "v1.0.3", key=keys / "next")
    assert w.updater.check().tag == "v1.0.2"  # 1.0.0 lists only the old key: v1.0.3 waits for the rotation
    assert any(m.startswith("self-update: v1.0.3 is not signed by a key") for m in caplog.messages)
    w.now = update.STAGGER_S
    w.updater.at_safe_point()
    assert w.head() == v102 and w.exits == [update.EXIT_UPDATED]

    def restarted(running: str) -> update.Updater:
        return update.Updater(code=w.code, running=running, install=w.install, exit=w.exits.append,
                              clock=lambda: w.now, rng=random.Random(7))

    rotated = restarted("1.0.2")
    assert rotated.check().tag == "v1.0.3"
    w.now += update.STAGGER_S
    rotated.at_safe_point()
    assert w.head() == v103 and w.exits == [update.EXIT_UPDATED] * 2
    commit(w.origin, "1.0.4", "v1.0.4", key=keys / "release")  # the old key, no longer listed
    assert restarted("1.0.3").check() is None
    assert caplog.messages[-1].startswith("self-update: v1.0.4 is not signed by a key release/allowed_signers lists")


def test_the_shipped_placeholder_lists_no_key_and_nothing_is_installed_until_a_key_is_added(tmp_path, keys,
                                                                                            caplog) -> None:
    w = World(tmp_path, keys, signers=SHIPPED_SIGNERS)
    for _ in range(3):
        w.updater.poll()
    refusals = [m for m in caplog.messages if "lists no release signing key" in m]
    assert len(refusals) == 1 and w.updater.pending is None
    assert refusals[0] == (f"self-update: {w.code / update.SIGNERS} lists no release signing key (only a "
                           "placeholder): no release can be verified, so none is installed by itself; run the "
                           "installer again to update")
    w.now = update.STAGGER_S
    w.updater.at_safe_point()
    assert w.head() == w.v100 and w.installs == [] and w.exits == []
    added = World(tmp_path / "added", keys, signers=SHIPPED_SIGNERS + allowed(keys, "release"))  # placeholder kept
    assert added.updater.check().tag == "v1.0.1"


def test_the_installer_holds_a_copy_of_the_release_keys() -> None:
    assert f"cat <<'EOF'\n{SHIPPED_SIGNERS}EOF\n" in (ROOT / "scripts" / "install.sh").read_text()


def fetch_code(home: Path, signers: str, *args: str) -> subprocess.CompletedProcess:
    """The installer's fetch_code (``<dir> <source> <ref>``), holding the release keys ``signers``."""
    script = (ROOT / "scripts" / "install.sh").read_text()
    assert script.count(SHIPPED_SIGNERS) == 1 and script.endswith('\nmain "$@"\n')
    script = script.replace(SHIPPED_SIGNERS, signers).removesuffix('main "$@"\n') + 'fetch_code "$@"\n'
    return subprocess.run(["bash", "-s", "--", *args], input=script, capture_output=True, text=True,
                          env={**os.environ, "HOME": str(home)}, timeout=60)


def test_the_installer_installs_a_tag_only_signed_by_a_key_it_holds(tmp_path, keys) -> None:
    w = World(tmp_path, keys)
    commit(w.origin, "1.0.2", "v1.0.2", annotated=True)
    commit(w.origin, "1.0.3", "v1.0.3", key=keys / "other")
    forged(w.origin, "v1.0.4", "v1.0.1")
    forged(w.origin, "v1.0.5", "v1.0.1", version="1.0.5")
    main = commit(w.origin, "1.0.6", "v1.0.6")
    holds, new = allowed(keys, "release"), tmp_path / "new"
    for tag, why in [("v1.0.2", "v1.0.2 is not signed with a release key (SSH)"),
                     ("v1.0.3", "v1.0.3 is not signed by a release key this installer holds (No principal matched.)"),
                     ("v1.0.4", "v1.0.4 names the tag object of another tag"),
                     ("v1.0.5", "v1.0.5 is not signed by a release key this installer holds ("),
                     ("v1.0.6", "v1.0.6 is not signed (a lightweight tag)")]:
        refused = fetch_code(tmp_path, holds, str(new), str(w.origin), tag)
        assert refused.returncode != 0 and why in refused.stderr and not new.exists()
    installed = fetch_code(tmp_path, holds, str(new), str(w.origin), "v1.0.1")
    assert installed.returncode == 0, installed.stderr
    assert 'v1.0.1: Good "git" signature for verathos-release with ED25519 key' in installed.stdout
    assert git(new, "rev-parse", "HEAD") == w.v101
    refused = fetch_code(tmp_path, holds, str(new), "", "v1.0.3")  # an update of that checkout
    assert refused.returncode != 0 and git(new, "rev-parse", "HEAD") == w.v101
    branch = fetch_code(tmp_path, holds, str(new), "", "main")
    assert branch.returncode == 0 and "note: main is not a release tag; it is installed unverified" in branch.stdout
    assert git(new, "rev-parse", "HEAD") == main



def test_the_installer_takes_a_releases_name_only_as_its_signed_tag(tmp_path, keys) -> None:
    """A branch named as a release (pushed by mistake, or what --ref names before the tag exists) is not installed
    unverified, and a source whose tags cannot be listed stops the install, where both installed the ref as given."""
    w = World(tmp_path, keys)
    commit(w.origin, "1.0.7")
    git(w.origin, "branch", "v1.0.7")
    holds, new = allowed(keys, "release"), tmp_path / "new"
    refused = fetch_code(tmp_path, holds, str(new), str(w.origin), "v1.0.7")
    assert refused.returncode != 0 and not new.exists()
    assert "v1.0.7 is not a tag of" in refused.stderr and "installed only from its signed tag" in refused.stderr
    unread = fetch_code(tmp_path, holds, str(w.code), str(tmp_path / "nothing-there"), "v1.0.1")
    assert unread.returncode != 0 and w.head() == w.v100

def test_the_installer_defaults_to_the_newest_release_once_it_holds_a_key(tmp_path, keys) -> None:
    w = World(tmp_path, keys)
    main = commit(w.origin, "1.0.2")  # the default branch, ahead of the newest release
    holds = allowed(keys, "release")
    installed = fetch_code(tmp_path, holds, str(tmp_path / "new"), str(w.origin), "")
    assert installed.returncode == 0 and f"({'v1.0.1'})" in installed.stdout
    assert git(tmp_path / "new", "rev-parse", "HEAD") == w.v101
    updated = fetch_code(tmp_path, holds, str(w.code), "", "")  # the checkout of 1.0.0, from its origin
    assert updated.returncode == 0 and w.head() == w.v101
    unverified = fetch_code(tmp_path, SHIPPED_SIGNERS, str(tmp_path / "placeholder"), str(w.origin), "")
    assert unverified.returncode == 0 and git(tmp_path / "placeholder", "rev-parse", "HEAD") == main
    assert "note: HEAD is not a release tag; it is installed unverified" in unverified.stdout
    refused = fetch_code(tmp_path, SHIPPED_SIGNERS, str(tmp_path / "asked"), str(w.origin), "v1.0.1")
    assert refused.returncode != 0
    assert "cannot verify v1.0.1: this installer holds no release signing key yet" in refused.stderr


def test_the_update_installs_the_code_as_the_installer_does() -> None:
    assert update.CODE == ROOT
    assert f'uv pip install --quiet -e "$dir[{update.EXTRAS}]"' in (ROOT / "scripts" / "install.sh").read_text()


def test_a_miner_restarts_for_an_update_only_outside_throughput_tests_and_waiting_measurements() -> None:
    def miner(testing=False, interrupted=False, deferral=None, window=True):
        return SimpleNamespace(testing=lambda: testing, interrupted=interrupted,
                               window=SimpleNamespace(deferral=lambda: deferral) if window else None)

    assert Publisher.update_held(miner()) is None
    assert Publisher.update_held(miner(window=False)) is None  # --no-publish: no schedule followed
    assert Publisher.update_held(miner(testing=True)) == "a throughput test is running"
    assert Publisher.update_held(miner(interrupted=True)) == "a measurement waits for a throughput test"
    selected = Deferral(45410, "the throughput test of epoch 45410 is selected, its traffic from block 1: measuring "
                               "after it")
    assert Publisher.update_held(miner(deferral=selected)) == selected.line


def test_a_light_validator_updates_only_between_steps(monkeypatch) -> None:
    order: list[str] = []

    class Light:
        def __init__(self, *args, **kwargs) -> None:
            pass

        def permit(self) -> None:
            return None

        def step(self) -> float:
            order.append("step")
            return 30.0

    class Stop(Exception):
        pass

    def sleep(seconds: float) -> None:
        order.append("sleep")
        if len(order) > 4:
            raise Stop

    monkeypatch.setattr(light, "LightValidator", Light)
    monkeypatch.setattr(light.time, "sleep", sleep)
    st = SimpleNamespace(get_subnet_owner_hotkey=lambda netuid: "owner")
    cfg = ValidatorConfig(netuid=405, wallet_name="v", hotkey_name="h", mode="light")
    with pytest.raises(Stop):
        light.run(cfg, st, SimpleNamespace(hotkey=SimpleNamespace(ss58_address="own")), None,
                  set_weights=lambda w, m: True, between=lambda: order.append("update"))
    assert order == ["step", "update", "sleep", "step", "update", "sleep"]


def test_the_installer_turns_the_self_update_on_except_for_the_auditor_and_keeps_the_choice(tmp_path) -> None:
    from verathos.install.validator import parser

    def ask(*argv: str) -> argparse.Namespace:
        return parser().parse_args(list(argv))

    assert common.auto_update(ask(), {}) and not common.auto_update(ask(), {}, default=False)  # new; the auditor's
    assert common.auto_update(ask(), {"wallet_name": "v"})  # set up by an earlier installer
    assert not common.auto_update(ask(), {"auto_update": False}) and not common.auto_update(ask("--no-auto-update"), {})
    assert common.auto_update(ask("--auto-update"), {"auto_update": False})
    base = dict(network="finney", netuid=96, wallet_name="v", hotkey_name="h", dry_run=False, managed_before=set(),
                home=tmp_path, mode="light", ports={})
    assert ValidatorConfig.model_validate(validator_config({}, **base, auto_update=True)).auto_update
    assert not ValidatorConfig.model_validate(validator_config({"auto_update": False}, **base)).auto_update  # kept
    config = miner_config({}, network="finney", netuid=96, wallet_name="m", hotkey_name="h", public_ip="203.0.113.7",
                          port=8091, endpoints={"gpu0": ("kev-4b", 8008)}, managed_before=set(), home=tmp_path,
                          auto_update=True)
    assert MinerConfig.model_validate(config).auto_update
    assert not ValidatorConfig(netuid=96, wallet_name="v", hotkey_name="h").auto_update  # a manual installation
    assert "self-update on" in common.update_line(True) and "--auto-update turns it on" in common.update_line(False)
