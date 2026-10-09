"""A reference machine: its front over real TLS (the callers' addresses and key, the pin, routing to the engines), the
placement of its engines, its setup, the file it hands its callers, the validator setup that takes the file, and the
validator's warning while such an engine does not serve."""

from __future__ import annotations

import asyncio
import contextlib
import json
import logging
import ssl
from collections.abc import Iterator
from pathlib import Path
from types import SimpleNamespace

import httpx
import pytest
from fastapi import FastAPI, Request

from tests.lineups import two_active_and_a_candidate
from tests.mock_engine import MockEngine
from tests.test_install import CLEF, KEV, ServiceManager, gpu
from tests.tlsfake import free_port, running, serving
from verathos import catalog
from verathos.engine import code_version
from verathos.install import common, engines, host, plan, services
from verathos.install import reference as setup
from verathos.install import validator as validator_setup
from verathos.install.host import GiB
from verathos.miner.engine import EngineError, HttpEngine
from verathos.protocol import tls
from verathos.reference import front
from verathos.reference.config import ReferenceConfig, api_key, engines_document
from verathos.validator import __main__ as validator_main
from verathos.validator.config import ValidatorConfig

PARAMETERS = {**KEV, "clef-flash": CLEF}
FAMILIES = {"kev-4b": "kev", "kev-9b": "kev", "clef-flash": "clef"}
REQUEST = {"model": "kev-4b", "state": "Hi team, we were billed twice for March. Also, the export button fails.",
           "questions": {"churn": {"type": "noul", "instructions": "Does the customer threaten to cancel?"},
                         "department": {"type": "choice", "instructions": "Which team should handle this?",
                                        "criteria": {"billing": "Charges", "technical": "Bugs"}}}}


def engine_app(model: str, salt: str, asked: list[str], codes: dict[str, str] | None = None) -> FastAPI:
    """An engine as it serves on its machine's loopback: deterministic answers, its revision and the engine code it
    runs (``codes[salt]``, if any), every request noted."""
    app, engine = FastAPI(), MockEngine(model, salt)

    @app.get("/v1/models")
    async def models():
        asked.append(f"{salt} models")
        code = {"code": codes[salt]} if codes and salt in codes else {}
        return {"data": [{"id": model, "revision": f"{salt}-revision", **code}]}

    @app.post("/v1/systemone")
    async def systemone(request: Request):
        asked.append(f"{salt} {request.headers.get('authorization', 'no key')}")
        return await engine.answer(await request.json())

    return app


@contextlib.contextmanager
def machine(tmp_path: Path, allow: list[str], listen: str = "127.0.0.1",
            codes: dict[str, str] | None = None) -> Iterator[SimpleNamespace]:
    """A reference machine on this host: kev-4b's reference and screen and kev-9b's reference, each an engine over
    plain http on loopback (``codes``: the engine code each names, by salt), behind the front over real TLS on a
    loopback port, its socket filtered to ``allow``."""
    asked: list[str] = []
    with contextlib.ExitStack() as stack:
        urls = {key: stack.enter_context(serving(engine_app(key[1], salt, asked, codes), None))
                for key, salt in ((("reference", "kev-4b"), "fp32"), (("screen", "kev-4b"), "bf16"),
                                  (("reference", "kev-9b"), "fp32-9b"))}
        served: dict[str, dict[str, str]] = {}
        for (kind, model), url in urls.items():
            served.setdefault(model, {})[kind] = url
        config = ReferenceConfig.model_validate({"public_host": "127.0.0.1", "port": free_port(), "listen_host": listen,
                                                 "allow": allow, "engines": served,
                                                 "state_dir": str(tmp_path / "front")})
        identity, key = tls.identity(config.state_dir), api_key(config.state_dir)
        server, listening = front.bind(config, identity, key)
        stack.enter_context(running(server, [listening]))
        yield SimpleNamespace(base=f"https://127.0.0.1:{config.port}", pin=identity.pin, key=key, asked=asked,
                              config=config, port=config.port)


def answer(url: str, key: str | None, pin: str | None, request: dict = REQUEST) -> dict:
    async def run() -> dict:
        engine = HttpEngine(url, key, timeout_s=5.0, tls_pin=pin)
        try:
            return await engine.answer(request)
        finally:
            await engine.close()

    return asyncio.run(run())


def problem(url: str, key: str | None, pin: str) -> str | None:
    async def run() -> str | None:
        engine = HttpEngine(url, key, timeout_s=2.0, tls_pin=pin)
        try:
            return await engine.problem()
        finally:
            await engine.close()

    return asyncio.run(run())


# ---- the front ----

def test_the_front_serves_each_engine_by_path_to_a_pinned_caller_with_the_key(tmp_path) -> None:
    """The answers through the front are the engine's own, the key stays at the front, and each path reaches its
    engine: the model's reference, its screen, another model's reference; nothing else."""
    with machine(tmp_path, ["198.51.100.9"]) as box:
        through = {kind: answer(f"{box.base}/{kind}/kev-4b", box.key, box.pin) for kind in ("reference", "screen")}
        kev9 = answer(f"{box.base}/reference/kev-9b", box.key, box.pin, {**REQUEST, "model": "kev-9b"})
        assert through["reference"] == asyncio.run(MockEngine("kev-4b", "fp32").answer(REQUEST))
        assert through["screen"] == asyncio.run(MockEngine("kev-4b", "bf16").answer(REQUEST))
        assert through["reference"] != through["screen"] and kev9["model"] == "kev-9b"
        assert box.asked == ["fp32 no key", "bf16 no key", "fp32-9b no key"]  # the key stays at the front

        async def revisions() -> list:
            out = []
            for kind, model in (("reference", "kev-4b"), ("screen", "kev-4b")):
                engine = HttpEngine(f"{box.base}/{kind}/{model}", box.key, tls_pin=box.pin)
                out.append(await engine.info())
                await engine.close()
            return out

        assert [doc["data"][0]["revision"] for doc in asyncio.run(revisions())] == ["fp32-revision", "bf16-revision"]
        assert problem(f"{box.base}/reference/kev-4b", box.key, box.pin) is None
        for path in ("screen/kev-9b", "reference/clef-flash", "e/gpu0", "reference/kev-4b/extra"):
            assert problem(f"{box.base}/{path}", box.key, box.pin) == "serves no such engine (HTTP 404)"
        with pytest.raises(EngineError, match="engine returned 404"):
            answer(f"{box.base}/screen/kev-9b", box.key, box.pin)


def test_the_front_refuses_a_missing_or_wrong_key_before_reading_the_request(tmp_path) -> None:
    with machine(tmp_path, ["198.51.100.9"]) as box:
        before = list(box.asked)
        with httpx.Client(verify=tls.context(box.pin), trust_env=False) as client:
            for headers in ({}, {"Authorization": "Bearer " + "x" * 43}, {"Authorization": box.key},
                            {"Authorization": f"Bearer {box.key}x"}):
                r = client.post(f"{box.base}/reference/kev-4b/v1/systemone", json=REQUEST, headers=headers)
                assert (r.status_code, r.json()) == (401, {"error": "the key is missing or wrong"})
            assert client.get(f"{box.base}/reference/kev-4b/v1/models").status_code == 401
        with pytest.raises(EngineError, match="engine returned 401"):
            answer(f"{box.base}/reference/kev-4b", "wrong", box.pin)
        assert problem(f"{box.base}/reference/kev-4b", None, box.pin) == "refuses the key (HTTP 401)"
        assert box.asked == before  # nothing reached an engine


def test_only_a_client_pinning_the_fronts_certificate_connects(tmp_path) -> None:
    """Another pin is refused before any byte of the request is sent; a client checking public CAs refuses the
    self-signed certificate; an engine at an IP address without a pin is not configured at all."""
    other = tls.identity(tmp_path / "other").pin
    with machine(tmp_path, ["198.51.100.9"]) as box:
        before = list(box.asked)
        with pytest.raises(EngineError, match=f"is not the pinned {other}"):
            answer(f"{box.base}/reference/kev-4b", box.key, other)
        assert problem(f"{box.base}/reference/kev-4b", box.key, other).startswith("unreachable (ConnectError: ")
        with httpx.Client(trust_env=False) as client, pytest.raises(httpx.ConnectError, match="CERTIFICATE_VERIFY"):
            client.post(f"{box.base}/reference/kev-4b/v1/systemone", json=REQUEST,
                        headers={"Authorization": f"Bearer {box.key}"})
        with pytest.raises(ValueError, match="needs its TLS pin"):
            HttpEngine(f"{box.base}/reference/kev-4b", box.key)
        with pytest.raises(ValueError, match="needs its TLS pin"):
            ValidatorConfig.model_validate({"netuid": 405, "wallet_name": "w", "hotkey_name": "h", "references": {
                "kev-4b": {"url": f"{box.base}/reference/kev-4b", "api_key": box.key}}})
        assert box.asked == before
        assert answer(f"{box.base}/reference/kev-4b", box.key, box.pin)["model"] == "kev-4b"


def test_the_front_drops_every_address_but_its_callers_before_a_handshake(tmp_path) -> None:
    """With real sockets: the listening socket drops a connection from an address not on the list at its first
    packet; this machine's loopback always passes."""
    with machine(tmp_path, ["127.0.0.2"], listen="0.0.0.0") as box:
        def connects(source: str) -> bool:
            transport = httpx.HTTPTransport(verify=tls.context(box.pin), local_address=source)
            with httpx.Client(transport=transport, timeout=httpx.Timeout(5.0, connect=1.0), trust_env=False) as c:
                try:
                    r = c.get(f"{box.base}/reference/kev-4b/v1/models",
                              headers={"Authorization": f"Bearer {box.key}"})
                except (httpx.ConnectError, httpx.ConnectTimeout, ssl.SSLError):
                    return False
                return r.status_code == 200

        assert connects("127.0.0.2") and connects("127.0.0.1")
        assert not connects("127.0.0.3")


def test_the_app_refuses_an_address_not_allowed_that_reaches_it() -> None:
    """Where the socket is not filtered, the app refuses other addresses (403), before the key and the body."""
    asked: list[str] = []
    app = front.create_app({("reference", "kev-4b"): "http://127.0.0.1:1"}, "k" * 43,
                           front.allows(frozenset({"198.51.100.9", "2001:db8::9"})),
                           upstream=httpx.AsyncClient(transport=httpx.ASGITransport(
                               app=engine_app("kev-4b", "fp32", asked))))

    async def post(client: tuple[str, int], key: str) -> httpx.Response:
        transport = httpx.ASGITransport(app=app, client=client)
        async with httpx.AsyncClient(transport=transport, base_url="https://front") as c:
            return await c.post("/reference/kev-4b/v1/systemone", json=REQUEST,
                                headers={"Authorization": f"Bearer {key}"})

    assert asyncio.run(post(("203.0.113.9", 4000), "k" * 43)).status_code == 403
    assert asyncio.run(post(("203.0.113.9", 4000), "wrong")).status_code == 403
    assert asyncio.run(post(("198.51.100.9", 4000), "wrong")).status_code == 401
    assert asyncio.run(post(("::ffff:198.51.100.9", 4000), "k" * 43)).status_code == 200
    assert asyncio.run(post(("2001:db8::9", 4000), "k" * 43)).status_code == 200
    assert asyncio.run(post(("127.0.0.1", 4000), "k" * 43)).status_code == 200  # this machine
    assert asked == ["fp32 no key"] * 3


def test_an_engine_that_does_not_answer_is_a_gateway_error_and_not_up() -> None:
    async def run() -> tuple[int, int, bool, str | None]:
        app = front.create_app({("reference", "kev-4b"): f"http://127.0.0.1:{free_port()}"}, "k" * 43,
                               front.allows(frozenset()))
        async with httpx.AsyncClient(transport=httpx.ASGITransport(app=app), base_url="https://front") as c:
            headers = {"Authorization": "Bearer " + "k" * 43}
            down = await c.post("/reference/kev-4b/v1/systemone", json=REQUEST, headers=headers)
            large = await c.post("/reference/kev-4b/v1/systemone", content=b"x" * (front.MAX_BODY_BYTES + 1),
                                 headers=headers)
        engine = HttpEngine(f"http://127.0.0.1:{free_port()}")  # nothing listens
        try:
            return down.status_code, large.status_code, await engine.reachable(), await engine.problem()
        finally:
            await engine.close()

    down, large, reachable, why = asyncio.run(run())
    assert (down, large, reachable) == (502, 413, False) and why.startswith("unreachable (ConnectError")


# ---- placement ----

def test_engines_go_smallest_first_on_the_gpus_and_references_that_do_not_fit_into_ram() -> None:
    def where(screens=PARAMETERS, gpus=None, ram: float = 210, device: str = "auto", keep=None,
              models=PARAMETERS) -> dict[str, str]:
        return plan.place_engines({m: PARAMETERS[m] for m in models}, FAMILIES, [m for m in screens if m in models],
                                  {0: 80 * GiB} if gpus is None else gpus, int(ram * GiB), device, keep or {})

    # one A100 80 GB: the three screens fit on it (about 62 GiB), the references run in RAM (about 104 GiB)
    assert where() == {"screen/kev-4b": "cuda:0", "screen/clef-flash": "cuda:0", "screen/kev-9b": "cuda:0",
                       "reference/kev-4b": "cpu", "reference/clef-flash": "cpu", "reference/kev-9b": "cpu"}
    # kev-4b alone: its screen and reference both on the GPU
    assert where(models=["kev-4b"]) == {"screen/kev-4b": "cuda:0", "reference/kev-4b": "cuda:0"}
    # references only: placed as a validator places its references
    assert where(screens=()) == {"reference/kev-4b": "cuda:0", "reference/clef-flash": "cuda:0",
                                 "reference/kev-9b": "cpu"}
    assert plan.place_references(PARAMETERS, {0: 80 * GiB}, int(210 * GiB), "auto", {}) == {
        "kev-4b": "cuda:0", "clef-flash": "cuda:0", "kev-9b": "cpu"}
    # references on GPUs first, each on the GPU with the most memory left; a screen left out where no GPU has room
    assert where(device="gpu", gpus={0: 80 * GiB, 1: 80 * GiB}, models=["kev-4b", "clef-flash"]) == {
        "reference/kev-4b": "cuda:0", "reference/clef-flash": "cuda:1", "screen/kev-4b": "cuda:0",
        "screen/clef-flash": "cuda:0"}
    assert where(device="gpu", models=["kev-4b", "clef-flash"]) == {"reference/kev-4b": "cuda:0",
                                                                    "reference/clef-flash": "cuda:0"}
    with pytest.raises(plan.PlanError, match="the kev-9b reference needs about 48 GiB on a GPU; not enough"):
        where(device="gpu")
    with pytest.raises(plan.PlanError, match="the kev-9b reference needs about 42 GiB of RAM"):
        where(device="cpu", ram=64)
    # no GPU: references in RAM, no screens
    assert where(gpus={}) == {"reference/kev-4b": "cpu", "reference/clef-flash": "cpu", "reference/kev-9b": "cpu"}
    # engines placed before keep their place (their memory is taken already)
    kept = {"reference/kev-4b": "cuda:0", "screen/kev-4b": "cuda:0"}
    assert where(models=["kev-4b"], gpus={0: 1 * GiB}, keep=kept) == kept
    assert where(models=["kev-4b"], gpus={0: 1 * GiB}, keep=kept, device="cpu") == {"screen/kev-4b": "cuda:0",
                                                                                     "reference/kev-4b": "cpu"}


# ---- the setup ----

class Boxes(ServiceManager):
    """A service manager recording what the reference setup applies, removes and asks."""

    def __init__(self) -> None:
        super().__init__([])
        self.running: set[str] = set()

    def apply(self, applied, restart):
        self.running |= {s.name for s in applied}
        return super().apply(applied, restart)

    def remove(self, names):
        self.running -= set(names)
        super().remove(names)

    def active(self, name: str) -> bool:
        return name in self.running

    def log_command(self, name: str) -> str:
        return f"journalctl -u {name} -f"


@pytest.fixture()
def reference_machine(tmp_path, monkeypatch):
    """The reference setup on a machine with one A100 80 GB and 216 GiB of RAM, its chain, downloads and services
    stood in for."""
    managers: list[Boxes] = [Boxes()]
    installed: list[tuple[str, str]] = []
    monkeypatch.setattr(common, "manager_for", lambda args, record: managers[-1])
    monkeypatch.setattr(common, "connect", lambda network: SimpleNamespace())
    monkeypatch.setattr(setup, "current_catalog", lambda *args: catalog.load())
    monkeypatch.setattr(host, "gpus", lambda: [gpu(0, 80)])
    monkeypatch.setattr(host, "memory_available", lambda: (216 * GiB, 210 * GiB))
    monkeypatch.setattr(host, "port_free", lambda port, address="0.0.0.0": True)
    monkeypatch.setattr(host, "detect_public_ip", lambda: "203.0.113.7")
    monkeypatch.setattr(engines, "parameters", lambda m: PARAMETERS[m.model_id])
    for family in (engines.KevEngine, engines.ClefEngine):
        monkeypatch.setattr(family, "install", lambda self, purpose: installed.append((self.engine.name, purpose)))
    monkeypatch.setattr(engines, "download", lambda python, m, hf_home: None)
    monkeypatch.setattr(common, "wait_engines", lambda manager, ports, timeout_s: None)

    async def through_front(config, pin, key, cat, timeout_s):
        return {plan.engine_key(kind, m): "serves the pinned revision" for kind, m in config.served()}

    monkeypatch.setattr(setup, "through_front", through_front)

    def run(*extra: str) -> SimpleNamespace:
        setup.init(["--yes", "--network", "test", "--home", str(tmp_path), *extra])
        record = common.Record.load(tmp_path, "reference")
        return SimpleNamespace(manager=managers[-1], record=record,
                               config=ReferenceConfig.load(tmp_path / "reference.json"),
                               document=json.loads((tmp_path / common.ENGINES_FILE).read_text()))

    return SimpleNamespace(run=run, managers=managers, installed=installed, home=tmp_path)


def test_the_reference_setup_runs_each_engine_and_the_front_and_writes_the_file_validators_take(
        reference_machine, capsys) -> None:
    box = reference_machine
    with pytest.raises(SystemExit, match="--allow"):
        box.run()  # the callers are required the first time
    done = box.run("--allow", "198.51.100.9", "--port", "20004")
    names = [s.name for s in done.manager.applied]
    assert names == ["verathos-reference-clef-flash", "verathos-reference-kev-4b", "verathos-reference-kev-9b",
                     "verathos-screen-clef-flash", "verathos-screen-kev-4b", "verathos-screen-kev-9b",
                     "verathos-reference"]
    applied = {s.name: s for s in done.manager.applied}
    fp32, bf16, front_service = (applied["verathos-reference-kev-4b"], applied["verathos-screen-kev-4b"],
                                 applied["verathos-reference"])
    assert fp32.environment["KEV_DTYPE"] == "fp32" and fp32.environment["CUDA_VISIBLE_DEVICES"] == ""  # in RAM
    assert "KEV_DTYPE" not in bf16.environment and bf16.environment["CUDA_VISIBLE_DEVICES"] == "0"
    assert fp32.command[0].endswith(".venv-reference/bin/python") and bf16.command[0].endswith(".venv/bin/python")
    assert applied["verathos-screen-clef-flash"].command[-4:] == ["--device", "cuda", "--dtype", "bfloat16"]
    # the code each engine runs: verathos-engine this release's, the Kev engine the revision the catalog pins
    assert applied["verathos-screen-clef-flash"].environment["VERATHOS_ENGINE_CODE"] == code_version()
    assert done.record.engines["screen/clef-flash"]["code"] == code_version()
    assert done.record.engines["reference/kev-4b"]["code"] == f"kev {catalog.load().get('kev-4b').engine.revision}"
    assert front_service.command[-2:] == ["--config", str(box.home / "reference.json")]
    assert set(front_service.after) == set(names[:-1])
    assert sorted(box.installed) == [("clef", "miner"), ("clef", "reference"), ("kev", "miner"), ("kev", "reference")]

    config, document = done.config, done.document
    assert (config.public_host, config.port, config.allow) == ("203.0.113.7", 20004, ["198.51.100.9"])
    assert set(config.engines) == {"kev-4b", "kev-9b", "clef-flash"}
    assert all(url.startswith("http://127.0.0.1:") for kinds in config.engines.values() for url in kinds.values())
    pin, key = tls.stored_pin(config.state_dir), api_key(config.state_dir)
    assert document["references"]["kev-4b"] == {"url": "https://203.0.113.7:20004/reference/kev-4b", "tls_pin": pin,
                                                "api_key": key}
    assert document["audit_screens"]["clef-flash"]["url"] == "https://203.0.113.7:20004/screen/clef-flash"
    assert (box.home / common.ENGINES_FILE).stat().st_mode & 0o777 == 0o600
    for section in ("references", "audit_screens"):
        ValidatorConfig.model_validate({"netuid": 405, "wallet_name": "w", "hotkey_name": "h",
                                        "references": document["references"], section: document[section]})
    out = capsys.readouterr().out
    assert f"TLS pin {pin}, key {key}" in out and "--remote-engines reference-engines.json" in out
    assert "kev-4b: reference https://203.0.113.7:20004/reference/kev-4b, screen " in out

    # again: the same certificate, key and ports; --allow replaces the callers; --no-screens stops the screens
    ports = {k: e["port"] for k, e in done.record.engines.items()}
    box.managers.append(Boxes())
    box.managers[-1].running = set(names)
    again = box.run("--allow", "203.0.113.50,2001:db8::5", "--no-screens")
    assert again.document["references"]["kev-4b"] == document["references"]["kev-4b"]  # the same pin and key
    assert again.document["audit_screens"] == {}
    assert again.config.allow == ["203.0.113.50", "2001:db8::5"] and again.config.listen_host == "::"
    assert again.manager.removed == ["verathos-screen-clef-flash", "verathos-screen-kev-4b", "verathos-screen-kev-9b"]
    assert {k: e["port"] for k, e in again.record.engines.items()} == {k: p for k, p in ports.items()
                                                                       if k.startswith("reference/")}
    # the references now have the GPU: placed before in RAM, they stay there
    assert {e["device"] for e in again.record.engines.values()} == {"cpu"}
    box.managers.append(Boxes())
    kept = box.run("--model", "kev-4b")
    assert kept.config.allow == ["203.0.113.50", "2001:db8::5"] and list(kept.config.engines) == ["kev-4b"]
    assert kept.record.options == {"network": "test", "netuid": 405, "screens": False, "models": ["kev-4b"]}
    assert list(box.run().config.engines) == ["kev-4b"]  # kept
    assert set(box.run("--model", "auto").config.engines) == {"kev-4b", "kev-9b", "clef-flash"}


def test_an_update_restarts_the_engines_whose_code_changed(reference_machine, monkeypatch) -> None:
    """verathos-engine runs this release's code: a release that changes it changes its services, which the service
    manager restarts; the Kev engines run the revision the catalog pins and stay as they are."""
    box = reference_machine
    before = {s.name: services.Systemd.render(s) for s in box.run("--allow", "198.51.100.9").manager.applied}
    monkeypatch.setattr(engines, "code_version", lambda: "0123456789ab")
    box.managers.append(Boxes())
    again = box.run()
    after = {s.name: services.Systemd.render(s) for s in again.manager.applied}
    assert sorted(n for n in after if after[n] != before[n]) == ["verathos-reference-clef-flash",
                                                                 "verathos-screen-clef-flash"]
    assert again.record.engines["reference/clef-flash"]["code"] == "0123456789ab"


def test_the_reference_setup_refuses_a_machine_that_runs_a_validator_and_unknown_models(reference_machine) -> None:
    common.Record(role="validator", manager="systemd", services=["verathos-validator"]).save(reference_machine.home)
    with pytest.raises(SystemExit, match="runs a validator"):
        reference_machine.run("--allow", "198.51.100.9")
    common.Record.path(reference_machine.home, "validator").unlink()
    with pytest.raises(SystemExit, match="kev-27b is not an active or candidate model"):
        reference_machine.run("--allow", "198.51.100.9", "--model", "kev-27b")
    with pytest.raises(SystemExit, match="--allow: '10.0.0.300' is not an IP address"):
        reference_machine.run("--allow", "10.0.0.300")


def test_uninstall_removes_the_reference_machine_and_purge_its_key(tmp_path) -> None:
    home = tmp_path
    engine_dir = home / "engines" / "kev-x"
    engine_dir.mkdir(parents=True)
    (home / "hf").mkdir()
    config = {"public_host": "203.0.113.7", "port": 8443, "allow": ["198.51.100.9"], "engines": {},
              "state_dir": str(home / "reference")}
    common.write_json(home / "reference.json", config)
    api_key(home / "reference")
    common.write_json(home / common.ENGINES_FILE, {"references": {}}, private=True)
    common.Record(role="reference", config=str(home / "reference.json"), paths=[str(engine_dir)]).save(home)
    common.uninstall("reference", ["--home", str(home)])
    assert not engine_dir.exists() and not (home / "hf").exists() and not common.Record.path(home, "reference").exists()
    assert (home / "reference" / "api.key").exists()  # kept: the next setup serves with the same key and pin
    common.Record(role="reference", config=str(home / "reference.json")).save(home)
    common.uninstall("reference", ["--home", str(home), "--purge"])
    assert sorted(p.name for p in home.iterdir()) == ["install"]


# ---- the validator setup that takes the file ----

def engines_file(path: Path, pin: str, key: str, models=("kev-4b", "kev-9b", "clef-flash"), screens=("kev-4b",),
                 host_: str = "203.0.113.7") -> Path:
    config = ReferenceConfig.model_validate({
        "public_host": host_, "port": 20004, "allow": ["198.51.100.9"], "state_dir": str(path.parent),
        "engines": {m: {"reference": "http://127.0.0.1:8101", **({"screen": "http://127.0.0.1:8201"}
                                                                  if m in screens else {})} for m in models}})
    common.write_json(path, engines_document(config, pin, key), private=True)
    return path


def test_the_validator_configuration_takes_the_files_engines_and_keeps_them(tmp_path) -> None:
    pin, key = "ab" * 32, "k" * 43
    path = engines_file(tmp_path / "a.json", pin, key, models=("kev-4b", "kev-9b"))
    references, screens = validator_setup.read_remote_engines([str(path)])
    assert references["kev-9b"] == {"url": "https://203.0.113.7:20004/reference/kev-9b", "tls_pin": pin,
                                    "api_key": key}
    assert list(screens) == ["kev-4b"]

    def configure(existing: dict, ports: dict[str, int], managed_before: set[str], remote: dict, screens_=None,
                  owner_screens: bool = True) -> dict:
        config = validator_setup.validator_config(
            existing, network="test", netuid=405, wallet_name="v", hotkey_name="default", ports=ports, dry_run=False,
            managed_before=managed_before, home=tmp_path, remote=remote, screens=screens_)
        ValidatorConfig.model_validate(config)
        return config

    before = {"references": {"kev-4b": "http://127.0.0.1:8101", "kev-9b": "http://127.0.0.1:8102",
                             "clef-flash": {"url": "https://203.0.113.80:9000/x", "tls_pin": "cd" * 32}},
              "audit_screens": {"kev-4b": "http://127.0.0.1:8010"}, "public_host": "203.0.113.9"}
    config = configure(before, {"clef-flash": 8101}, {"kev-4b", "kev-9b"}, references, screens)
    # the file's engines replace the local references managed before, and the remote one configured before
    assert config["references"] == {**references, "clef-flash": "http://127.0.0.1:8101"}
    assert config["audit_screens"] == {"kev-4b": screens["kev-4b"]}  # the file's screen replaces the local one
    assert config["public_host"] == "203.0.113.9"
    # a run without the option keeps them: remote engines are those of the configuration
    kept = configure(config, {"clef-flash": 8101}, {"clef-flash"},
                     validator_setup.remote_entries(config["references"]))
    assert kept["references"] == config["references"] and kept["audit_screens"] == config["audit_screens"]
    # --remote-engines '': none; the setup places local references again
    cleared = configure(kept, {"kev-4b": 8102, "kev-9b": 8103, "clef-flash": 8101}, {"clef-flash"}, {}, {})
    assert all(v.startswith("http://127.0.0.1:") for v in cleared["references"].values())
    assert "audit_screens" not in cleared


def test_a_file_that_is_not_a_reference_machines_is_refused(tmp_path) -> None:
    good = engines_file(tmp_path / "a.json", "ab" * 32, "k" * 43, models=("kev-4b",))
    for document, why in (({"references": {"kev-4b": {"url": "http://127.0.0.1:8101"}}}, "not an engine on another"),
                          ({"references": {"kev-4b": {"url": "https://203.0.113.7:1/x", "api_key": "k"}}}, "TLS pin"),
                          ({"engines": {}}, "references"), ([], "not a reference machine's file")):
        bad = tmp_path / "bad.json"
        bad.write_text(json.dumps(document))
        with pytest.raises(SystemExit, match=why):
            validator_setup.read_remote_engines([str(bad)])
    with pytest.raises(SystemExit, match="kev-4b in two files"):
        validator_setup.read_remote_engines([str(good), str(engines_file(tmp_path / "b.json", "cd" * 32, "j" * 43,
                                                                         models=("kev-4b",), host_="203.0.113.8"))])
    with pytest.raises(SystemExit, match="not a reference machine's file"):
        validator_setup.read_remote_engines([str(tmp_path / "missing.json")])


@pytest.fixture()
def validator_machine(tmp_path, monkeypatch, lab_network):
    """The validator setup on a CPU machine: no GPU, 16 GiB of RAM, its chain and downloads stood in for."""
    managers: list[ServiceManager] = []
    owner = {"hotkey": "5Owner"}
    chain = SimpleNamespace(
        get_neuron_for_pubkey_and_subnet=lambda hotkey, netuid: SimpleNamespace(is_null=False, uid=5,
                                                                                validator_permit=True),
        get_subnet_owner_hotkey=lambda netuid: owner["hotkey"])
    monkeypatch.setattr(common, "manager_for", lambda args, record: managers[-1])
    monkeypatch.setattr(common, "load_hotkey", lambda w, h: SimpleNamespace(hotkey=SimpleNamespace(
        ss58_address="5Validator")))
    monkeypatch.setattr(common, "connect", lambda network: chain)
    monkeypatch.setattr(common, "wait_engines", lambda manager, ports, timeout_s: None)
    monkeypatch.setattr(validator_setup, "current_catalog", lambda *args: two_active_and_a_candidate())
    monkeypatch.setattr(host, "wallets", lambda: [])
    monkeypatch.setattr(host, "gpus", lambda: [])
    monkeypatch.setattr(host, "memory_available", lambda: (16 * GiB, 12 * GiB))
    monkeypatch.setattr(host, "port_free", lambda port, address="0.0.0.0": True)
    monkeypatch.setattr(host, "detect_public_ip", lambda: "203.0.113.9")
    monkeypatch.setattr(engines, "parameters", lambda m: PARAMETERS[m.model_id])
    for family in (engines.KevEngine, engines.ClefEngine):
        monkeypatch.setattr(family, "install", lambda self, purpose: pytest.fail("an engine was installed"))
    monkeypatch.setattr(engines, "download", lambda python, m, hf_home: pytest.fail("weights were downloaded"))

    async def remote_lines(references, screens, cat):
        return [f"{m}: checked" for m in sorted(references)] + [f"{m} screen: checked" for m in sorted(screens)]

    monkeypatch.setattr(validator_setup, "remote_lines", remote_lines)

    def run(*extra: str, owned: bool = False, mode: str = "--full") -> tuple[ServiceManager, dict]:
        owner["hotkey"] = "5Validator" if owned else "5Owner"
        lines = [["throughput tests are run by 5Owner; applying its verdicts" if not owned
                  else "throughput auditor running (enforce mode)"], ["epoch 7: 3 miners, 3 open endpoints"]]
        managers.append(ServiceManager(lines))
        validator_setup.init([mode, "--yes", "--wallet", "v", "--hotkey", "default", "--home", str(tmp_path),
                              *extra])
        return managers[-1], common.read_json(tmp_path / "validator.json")

    return SimpleNamespace(run=run, home=tmp_path)


def test_a_full_validator_on_a_cpu_machine_takes_its_engines_from_the_file_and_installs_none(
        validator_machine, capsys) -> None:
    """Every active model's reference on the reference machine: no engine, no weights, only the validator service; a
    later run keeps the remote entries; the auditor also takes the file's screens."""
    v = validator_machine
    path = engines_file(v.home / "reference-engines.json", "ab" * 32, "k" * 43)
    manager, config = v.run("--remote-engines", str(path))
    assert [s.name for s in manager.applied] == [validator_setup.VALIDATOR] and manager.applied[0].after == ()
    parsed = ValidatorConfig.model_validate(config)
    assert {m: (e.url, e.tls_pin, e.api_key) for m, e in parsed.references.items()} == {
        m: (f"https://203.0.113.7:20004/reference/{m}", "ab" * 32, "k" * 43) for m in ("kev-4b", "kev-9b",
                                                                                       "clef-flash")}
    assert parsed.audit_screens == {} and parsed.reference_mode == "resident"
    assert (v.home / "validator.json").stat().st_mode & 0o777 == 0o600  # it holds the key
    out = capsys.readouterr().out
    assert "kev-4b: checked" in out and "kev-4b screen: checked" not in out
    assert "the screens of kev-4b are the auditor's: not configured here" in out
    assert common.Record.load(v.home, "validator").engines == {}

    _, again = v.run()  # bash install.sh validator --yes
    assert again["references"] == config["references"]
    _, auditor = v.run("--remote-engines", str(path), owned=True)
    assert set(auditor["audit_screens"]) == {"kev-4b"} and auditor["audit_screens"]["kev-4b"]["url"].endswith(
        "/screen/kev-4b")
    assert "kev-4b screen: checked" in capsys.readouterr().out
    with pytest.raises(SystemExit, match="a light one runs none"):
        v.run("--remote-engines", str(path), mode="--light")


def test_a_validator_with_part_of_its_references_elsewhere_runs_the_others_itself(validator_machine,
                                                                                  monkeypatch) -> None:
    v = validator_machine
    installed: list[str] = []
    monkeypatch.setattr(engines.KevEngine, "install", lambda self, purpose: installed.append(purpose))
    monkeypatch.setattr(engines, "download", lambda python, m, hf_home: installed.append(m.model_id))
    monkeypatch.setattr(host, "memory_available", lambda: (64 * GiB, 60 * GiB))
    path = engines_file(v.home / "reference-engines.json", "ab" * 32, "k" * 43, models=("kev-9b",), screens=())
    manager, config = v.run("--remote-engines", str(path))
    assert config["references"]["kev-9b"]["url"] == "https://203.0.113.7:20004/reference/kev-9b"
    assert config["references"]["kev-4b"] == "http://127.0.0.1:8101" and installed == ["reference", "kev-4b"]
    assert [s.name for s in manager.applied] == ["verathos-reference-kev-4b", validator_setup.VALIDATOR]


def test_the_validator_setup_shows_what_each_remote_engine_serves_as_the_validator_reaches_it(tmp_path) -> None:
    """Its revision against the catalog's pin (the stand-in engines name other revisions), and the engine code it
    runs against this release's."""
    codes = {"fp32": code_version(), "bf16": "0123456789ab"}
    pinned = catalog.load().get("kev-4b").hf_revision[:12]
    with machine(tmp_path, ["198.51.100.9"], codes=codes) as box:
        def entry(kind: str, key: str = box.key) -> dict:
            return {"url": f"{box.base}/{kind}/kev-4b", "tls_pin": box.pin, "api_key": key}

        lines = asyncio.run(validator_setup.remote_lines({"kev-4b": entry("reference")}, {"kev-4b": entry("screen")},
                                                         catalog.load()))
        down = asyncio.run(validator_setup.remote_lines({"kev-4b": entry("reference", "wrong")}, {}, catalog.load()))
    assert lines == [
        f"kev-4b: float32 reference on another machine, {box.base}/reference/kev-4b (pin {box.pin[:12]}...): serves "
        f"revision fp32-revisio, the catalog pins {pinned}: not used until it serves the pinned one, engine code "
        f"{code_version()} (this release's)",
        f"kev-4b: bfloat16 screen on another machine, {box.base}/screen/kev-4b (pin {box.pin[:12]}...): serves "
        f"revision bf16-revisio, the catalog pins {pinned}: not used until it serves the pinned one, engine code "
        f"0123456789ab, not this release's {code_version()}: update its machine"]
    assert down == [f"kev-4b: float32 reference on another machine, {box.base}/reference/kev-4b (pin "
                    f"{box.pin[:12]}...): it does not serve now, refuses the key (HTTP 401); the validator waits for "
                    "it at its start (15 minutes at most) and checks it each epoch"]


def test_a_validator_setup_refuses_a_reference_machine(validator_machine) -> None:
    common.Record(role="reference", manager="supervisor", services=["verathos-reference"]).save(validator_machine.home)
    with pytest.raises(SystemExit, match="this machine is a reference machine"):
        validator_machine.run()


# ---- the validator while a remote engine does not serve ----

def test_the_validator_warns_once_per_epoch_while_an_engine_on_another_machine_does_not_serve(tmp_path,
                                                                                              caplog) -> None:
    caplog.set_level(logging.INFO, logger="verathos.validator")
    codes = {"fp32": code_version(), "bf16": code_version()}  # verathos-engine names its code; Kev's names none

    def warnings() -> list[str]:
        return [r.getMessage() for r in caplog.records if r.levelno == logging.WARNING
                and r.name == "verathos.validator"]

    with machine(tmp_path, ["198.51.100.9"], codes=codes) as box:
        def cfg(key: str, screen_port: int) -> ValidatorConfig:
            return ValidatorConfig.model_validate({
                "netuid": 405, "wallet_name": "w", "hotkey_name": "h",
                "references": {"kev-4b": {"url": f"{box.base}/reference/kev-4b", "tls_pin": box.pin, "api_key": key},
                               "kev-9b": "http://127.0.0.1:1"},  # local: started and checked as before
                "audit_screens": {"kev-4b": {"url": f"https://127.0.0.1:{screen_port}/screen/kev-4b",
                                             "tls_pin": box.pin, "api_key": key}}})

        asyncio.run(validator_main.check_remote_engines(cfg(box.key, box.port), 41, auditor=True))
        assert warnings() == []  # all of them serve, on this release's engine code
        gone = free_port()
        asyncio.run(validator_main.check_remote_engines(cfg("wrong", gone), 42, auditor=True))
        asyncio.run(validator_main.check_remote_engines(cfg("wrong", gone), 43, auditor=False))
        asyncio.run(validator_main.check_remote_engines(cfg(box.key, box.port), 44, auditor=True))
        down = warnings()
        assert len(down) == 2 and all("stays unchecked, not penalized" in w for w in down)
        assert down[0].startswith("epoch 42: engines on other machines not serving: the kev-4b reference at "
                                  f"{box.base}/reference/kev-4b: refuses the key (HTTP 401); the kev-4b screen at "
                                  f"https://127.0.0.1:{gone}/screen/kev-4b: unreachable (ConnectError")
        assert "screen" not in down[1]  # only the auditor uses screens
        assert "epoch 44: the engines on other machines serve again" in caplog.messages

        codes["bf16"] = "0123456789ab"  # the screen runs another release's engine code
        caplog.clear()
        asyncio.run(validator_main.check_remote_engines(cfg(box.key, box.port), 45, auditor=False))
        assert warnings() == []  # the screen is the auditor's
        asyncio.run(validator_main.check_remote_engines(cfg(box.key, box.port), 46, auditor=True))
        assert warnings() == [
            f"epoch 46: engines on other machines run other engine code than this validator's release "
            f"({code_version()}): the kev-4b screen at https://127.0.0.1:{box.port}/screen/kev-4b runs 0123456789ab; "
            "update their machine to this release (install.sh reference --yes there)"]


def test_install_script_offers_the_reference_role() -> None:
    import subprocess

    from tests.test_install import ROOT

    usage = subprocess.run(["bash", str(ROOT / "scripts" / "install.sh"), "--help"], capture_output=True, text=True)
    assert "reference" in usage.stdout and "--allow" in usage.stdout and "--remote-engines" in usage.stdout
