"""The self-update: which release it takes, when it waits or refuses, and how it updates a checkout (a real git
repository as the origin; the install and the exit replaced)."""

from __future__ import annotations

import argparse
import logging
import random
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


def git(cwd: Path, *args: str) -> str:
    return subprocess.run(["git", *args], cwd=cwd, check=True, capture_output=True, text=True).stdout.strip()


def commit(origin: Path, version: str, tag: str | None = None, *, annotated: bool = False,
           names: str | None = None) -> str:
    """A commit of code naming ``names`` (default ``version``), tagged ``tag``."""
    init = origin / "src" / "verathos" / "__init__.py"
    init.parent.mkdir(parents=True, exist_ok=True)
    init.write_text(f'"""The package, release {version}."""\n\n__version__ = "{names or version}"\n')
    git(origin, "add", "-A")
    git(origin, "commit", "-q", "-m", f"release {version}")
    if tag:
        git(origin, "tag", *(["-a", "-m", tag] if annotated else []), tag)
    return git(origin, "rev-parse", "HEAD")


class World:
    """An origin with releases 1.0.0 (an annotated tag) and 1.0.1, and a checkout of 1.0.0 with an updater."""

    def __init__(self, tmp_path: Path, install_fails: int = 0) -> None:
        self.origin = tmp_path / "origin"
        self.origin.mkdir()
        git(self.origin, "init", "-q", "-b", "main")
        self.v100 = commit(self.origin, "1.0.0", "v1.0.0", annotated=True)
        self.v101 = commit(self.origin, "1.0.1", "v1.0.1")
        git(tmp_path, "clone", "-q", str(self.origin), "code")
        self.code = tmp_path / "code"
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
    assert update.newest(found, (1, 0, 0)).tag == "v1.10.0"  # by number, not by name
    assert update.newest(found, (1, 9, 5)).tag == "v1.10.0"
    assert update.newest(found, (1, 10, 0)) is None and update.newest(found, (2, 0, 0)) is None


def test_an_update_waits_for_its_time_then_takes_the_release_installs_it_and_exits(tmp_path, caplog) -> None:
    w = World(tmp_path)
    target = w.updater.check()
    assert (target.tag, target.commit) == ("v1.0.1", w.v101)
    assert 0 < w.updater.due_at <= update.STAGGER_S and not w.updater.due()  # a random time, not at once
    assert any("release v1.0.1 is out (running 1.0.0)" in m for m in caplog.messages)
    w.updater.at_safe_point()
    assert w.head() == w.v100 and w.installs == [] and w.exits == []
    w.now = w.updater.due_at
    w.updater.check()  # listed again: the time stays
    assert w.updater.due()
    w.updater.at_safe_point()
    assert w.head() == w.v101 and w.installs == [w.v101] and w.exits == [update.EXIT_UPDATED]
    restarted = update.Updater(code=w.code, running="1.0.1", git=update.git)
    assert restarted.check() is None and not restarted.due()  # the newest release runs


def test_local_changes_keep_the_code_as_it_is_and_are_logged_once(tmp_path, caplog) -> None:
    w = World(tmp_path)
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


def test_a_tag_whose_code_names_another_version_is_never_taken(tmp_path, caplog) -> None:
    w = World(tmp_path)
    commit(w.origin, "1.0.2", "v1.0.2", names="1.0.1")  # it would restart into an update again
    assert w.updater.check().tag == "v1.0.2"
    w.now = update.STAGGER_S
    w.updater.at_safe_point()
    assert w.head() == w.v100 and w.exits == [] and w.installs == []
    assert "v1.0.2 holds verathos 1.0.1, not the version it names" in caplog.messages[-1]
    assert w.updater.check().tag == "v1.0.1"  # the newest release that is what it says


def test_a_failed_install_puts_the_previous_commit_back(tmp_path, caplog) -> None:
    w = World(tmp_path, install_fails=1)
    w.updater.check()
    w.now = update.STAGGER_S
    w.updater.at_safe_point()
    assert w.head() == w.v100 and w.installs == [w.v101, w.v100] and w.exits == []
    assert f"installing v1.0.1 failed (uv pip install failed: no matching distribution); back on {w.v100[:12]}" in \
        caplog.messages[-1]
    assert w.updater.check() is None  # not tried again until the next start


def test_a_checkout_without_git_or_an_origin_is_not_updated(tmp_path) -> None:
    (tmp_path / "plain").mkdir()
    with pytest.raises(update.Refused, match="not a git checkout"):
        update.Updater(code=tmp_path / "plain", running="1.0.0").check()
    git(tmp_path, "init", "-q", "lone")
    with pytest.raises(update.Refused, match="git ls-remote failed"):
        update.Updater(code=tmp_path / "lone", running="1.0.0").check()
    assert update.start(False, "validator") is None


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
