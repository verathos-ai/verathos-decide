"""``verathos-gateway``: run the gateway, and administer its database.

    verathos-gateway serve --config gateway.json
    verathos-gateway db upgrade --config gateway.json
    verathos-gateway accounts create --name acme --rate 50 --burst 100 [--monthly-quota 1000000] [--prepaid yes]
    verathos-gateway accounts set --name acme [--rate R] [--burst B] [--monthly-quota Q] [--status suspended]
                                              [--prepaid yes|no]
    verathos-gateway accounts list
    verathos-gateway keys create --account acme --name prod [--expires 2027-01-01]
    verathos-gateway keys list --account acme
    verathos-gateway keys revoke --account acme --prefix vd_1a2b3c4d5e6f7
    verathos-gateway usage --account acme [--since 2026-10-01]
    verathos-gateway credits add --account acme --usd 5 --kind topup --reference <payment id> [--note N]
    verathos-gateway credits list --account acme
"""

from __future__ import annotations

import argparse
import asyncio
import contextlib
import datetime as dt
import importlib.util
import json
import logging
import random
import socket
from collections.abc import AsyncIterator
from pathlib import Path
from typing import Any

import bittensor as bt
import httpx
import uvicorn
from pydantic import ValidationError

from verathos import catalog as catalog_module
from verathos.gateway import db
from verathos.gateway.accounts import Accounts, Conflict
from verathos.gateway.admin import AccountChange, ByPort, NewAccount, NewCredit, admin_app, read_token
from verathos.gateway.app import (
    ACCOUNTS_WAIT_S,
    Gateway,
    create_app,
    final_writes,
    mark_states_forever,
    validation_message,
)
from verathos.gateway.config import GatewayConfig
from verathos.gateway.limits import RateLimiter
from verathos.gateway.media import landlock_abi
from verathos.gateway.router import Router
from verathos.gateway.routes import Refused, RouteTable, dropped_models, stale_after_s
from verathos.gateway.sampler import Sampler, TestFollower
from verathos.gateway.served import ServedCounter
from verathos.gateway.upstream import Upstream
from verathos.gateway.usage import UsageMeter
from verathos.protocol import tls
from verathos.protocol.gateway import Routes
from verathos.protocol.http import raise_open_file_limit
from verathos.protocol.media import keep_out_of_dumps
from verathos.throughput.chain import SubtensorChainView
from verathos.throughput.follow import ScheduleFollower
from verathos.throughput.policy import AuditPolicy
from verathos.validator.follow import CatalogFollower

log = logging.getLogger("verathos.gateway")
RESERVED_FILES = 512  # the database pool, the listeners, the metrics and log files
MIN_OUTBOUND = 256
# An idle connection to a miner or the auditor is closed after this, before their server (uvicorn, 5 s) closes it: a
# connection the server closes as it is reused fails the request sent on it.
KEEPALIVE_S = 3.0


async def serve(config: GatewayConfig) -> None:
    if importlib.util.find_spec("PIL") is None:
        raise SystemExit("Pillow is missing: the gateway decodes users' images with it (install the gateway extra)")
    keep_out_of_dumps()
    if landlock_abi() < 1:
        log.error("the kernel offers no Landlock (Linux 5.13 or later, with landlock among its security modules): "
                  "this gateway decodes no image, and requests with images get 503")
    engine = db.create_engine(config.database_url)
    if await db.schema_revision(engine) != db.head_revision():
        raise SystemExit("the database schema is not current: run `verathos-gateway db upgrade` first")
    keypair = bt.Wallet(name=config.wallet_name, hotkey=config.hotkey_name).hotkey
    chain = SubtensorChainView(config.network, config.netuid)
    st = await asyncio.to_thread(bt.Subtensor, network=config.network)
    auditor_hotkey = await asyncio.to_thread(st.get_subnet_owner_hotkey, config.netuid)
    state_dir = config.state_dir.expanduser()
    store = catalog_module.CatalogStore(config.catalog_dir)
    # the committed catalog names the auditor's pin: followed from the chain, so a new pin is taken within a minute
    committed = await asyncio.to_thread(CatalogFollower, st, config.network, config.netuid, store, [],
                                        state_path=state_dir / "catalog_followed.json")
    st.close()
    committed.start()

    async def auditor() -> tls.Service | None:
        """The auditor's service, at its pin as the committed catalog names it (raises catalog.AuditorUnknown)."""
        url = config.auditor_url or await asyncio.to_thread(chain.auditor_url)
        return tls.Service(url, committed.catalog.auditor_pin()) if url else None

    # Every connection holds a descriptor: the users' requests in flight, and at most the rest to miners and auditor.
    files = raise_open_file_limit()
    outbound = files - config.max_in_flight - RESERVED_FILES
    if outbound < MIN_OUTBOUND:
        log.warning("the open-file limit %d cannot hold max_in_flight %d and connections to the miners: raise it "
                    "(LimitNOFILE=65536 in the systemd unit) or lower max_in_flight", files, config.max_in_flight)
    outbound = max(outbound, MIN_OUTBOUND)
    # one connection pool per miner machine (its pin), each within the limits
    client = tls.client(limits=httpx.Limits(max_connections=outbound, keepalive_expiry=KEEPALIVE_S,
                                            max_keepalive_connections=min(outbound, 4096)),
                        timeout=httpx.Timeout(30.0, connect=3.0))

    routes = RouteTable(client=client, auditor=auditor, auditor_hotkey=auditor_hotkey, netuid=config.netuid,
                        path=state_dir / "routes.json")
    current: list[catalog_module.Catalog | None] = [None]
    known: list[set[tuple[str, str, str]] | None] = [None]

    async def adopt(table: Routes) -> None:
        cat = current[0]
        if cat is None or cat.digest() != table.catalog_digest:
            url = config.auditor_url or await asyncio.to_thread(chain.auditor_url)
            sources = [url.rstrip("/") + catalog_module.CATALOG_PATH] if url else []
            cat = await asyncio.to_thread(catalog_module.resolve, table.catalog_digest, store=store, sources=sources)
            if keypair.ss58_address not in cat.gateway_hotkeys():
                log.error("this gateway's hotkey %s is not among catalog %s's gateways: miners will refuse it",
                          keypair.ss58_address, cat.digest()[:12])
        dropped = dropped_models(routes.table, table, cat)
        if dropped and not routes.stale():
            raise Refused(f"it drops every route of {', '.join(dropped)}, still active in its catalog; keeping the "
                          f"routes of epoch {routes.table.epoch}")
        current[0] = cat
        router.max_latency_s = cat.audit_policy().max_latency_s
        routes.stale_s = stale_after_s(cat)
        endpoints = {(r.hotkey, r.endpoint_id, r.model_id) for r in table.routes}
        router.update(table)
        if endpoints != known[0]:
            known[0] = endpoints
            log.info("routes of epoch %d: %d endpoints for %s", table.epoch, len(table.routes),
                     ", ".join(sorted(router.models())) or "no model")

    routes.on_change.append(adopt)
    meter = UsageMeter(engine)
    accounts = Accounts(engine, meter)
    sampler = Sampler(keypair, client, auditor=auditor, auditor_hotkey=auditor_hotkey,
                      window_s=config.sample_window_s, rng=random.SystemRandom())
    served = ServedCounter(keypair, client, auditor=auditor, auditor_hotkey=auditor_hotkey)
    tests = TestFollower(sampler, ScheduleFollower(chain, AuditPolicy()),
                         lambda: current[0].audit_policy() if current[0] is not None else None)
    router = Router(share=1 / config.instances, max_latency_s=2.0, observer=served)
    gateway = Gateway(config=config, hotkey=keypair.ss58_address, catalog=lambda: current[0], accounts=accounts,
                      meter=meter, router=router, sampler=sampler, served=served, routes=lambda: routes.table,
                      upstream=Upstream(keypair, client, netuid=config.netuid),
                      limiter=RateLimiter())

    @contextlib.asynccontextmanager
    async def lifespan(app: Any) -> AsyncIterator[None]:
        """The background work runs with the server; on shutdown (SIGTERM included) usage, samples and served
        counts are written before the process ends."""
        await routes.load()
        try:
            await accounts.load_keys()
        except Exception as exc:
            log.warning("keys not loaded, every key is looked up until they are: %s: %s", type(exc).__name__, exc)
        tasks = [asyncio.create_task(t) for t in (routes.run(), meter.run(), sampler.run(), served.run(),
                                                     accounts.run(), tests.run(), mark_states_forever(gateway))]
        log.info("gateway %s on %s:%d (auditor %s, %d instance(s))", keypair.ss58_address, config.listen_host,
                 config.port, auditor_hotkey, config.instances)
        try:
            yield
        finally:
            for task in tasks:
                task.cancel()
            await asyncio.gather(*tasks, return_exceptions=True)
            await final_writes(gateway)
            gateway.decoders.close()
            await client.aclose()
            await engine.dispose()

    if config.metrics_port:
        from prometheus_client import start_http_server
        start_http_server(config.metrics_port, addr="127.0.0.1")
    app: Any = create_app(gateway, lifespan=lifespan)
    sockets = [listen(config.listen_host, config.port)]
    if config.admin_port is not None:
        token = read_token(config.admin_token_file)
        accounts_token = read_token(config.accounts_token_file) if config.accounts_token_file else None
        if accounts_token == token:
            raise SystemExit("the account service's token must differ from the admin token")
        app = ByPort(app, config.admin_port, admin_app(gateway, token=token, accounts_token=accounts_token))
        sockets.append(listen(config.admin_host, config.admin_port))
        log.info("admin API on %s:%d", config.admin_host, config.admin_port)
    # On SIGTERM: stop accepting (the reverse proxy turns to the standby), finish what is in flight (its account
    # checks and its miners' deadline), write usage.
    in_flight_s = 3 * ACCOUNTS_WAIT_S + config.request_timeout_s
    served_tls = ({"ssl_certfile": str(config.tls_cert_file), "ssl_keyfile": str(config.tls_key_file)}
                  if config.tls_cert_file is not None else {})
    server = uvicorn.Server(uvicorn.Config(app, log_level="warning", access_log=False, server_header=False,
                                           timeout_graceful_shutdown=in_flight_s + 2, **served_tls))
    await server.serve(sockets=sockets)


def listen(host: str, port: int) -> socket.socket:
    return socket.create_server((host, port), family=socket.AF_INET6 if ":" in host else socket.AF_INET)


def run(coro: Any) -> Any:
    try:
        import uvloop
    except ImportError:
        return asyncio.run(coro)
    return asyncio.run(coro, loop_factory=uvloop.new_event_loop)


async def admin(config: GatewayConfig, args: argparse.Namespace) -> Any:
    engine = db.create_engine(config.database_url)
    accounts = Accounts(engine)
    try:
        if args.command == "accounts":  # checked as the admin API checks them
            if args.action == "create":
                new = NewAccount(name=args.name, rate_rps=args.rate, burst=args.burst,
                                 monthly_quota=args.monthly_quota, prepaid=args.prepaid == "yes")
                return {"id": await accounts.create_account(new.name, rate_rps=new.rate_rps, burst=new.burst,
                                                            monthly_quota=new.monthly_quota, prepaid=new.prepaid)}
            if args.action == "set":
                prepaid = None if args.prepaid is None else args.prepaid == "yes"
                change = AccountChange(**{k: v for k, v in (
                    ("rate_rps", args.rate), ("burst", args.burst), ("monthly_quota", args.monthly_quota),
                    ("status", args.status), ("prepaid", prepaid)) if v is not None})
                return {"updated": await accounts.set_account(args.name, **change.model_dump(exclude_unset=True))}
            return await accounts.list_accounts()
        if args.command == "keys":
            if args.action == "create":
                expires = None
                if args.expires:
                    expires = dt.datetime.fromisoformat(args.expires)
                    expires = expires.replace(tzinfo=dt.UTC) if expires.tzinfo is None else expires.astimezone(dt.UTC)
                return {"key": await accounts.create_key(args.account, args.name, expires),
                        "note": "shown once; store it now"}
            if args.action == "revoke":
                return {"revoked": await accounts.revoke_key(args.account, args.prefix)}
            return await accounts.list_keys(args.account)
        if args.command == "credits":
            if args.action == "add":
                new = NewCredit(amount=round(args.usd * 1_000_000), kind=args.kind, reference=args.reference,
                                note=args.note)
                entry, created = await accounts.add_credit(args.account, new.amount, kind=new.kind,
                                                           reference=new.reference, note=new.note)
                return {"credit": entry, "created": created}
            return {"balance": (await accounts.account(args.account) or {}).get("balance"),
                    "credits": await accounts.credits(args.account)}
        since = dt.date.fromisoformat(args.since) if args.since else dt.datetime.now(dt.UTC).date().replace(day=1)
        return await accounts.usage(args.account, since)
    finally:
        await engine.dispose()


def main() -> None:
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s %(message)s")
    logging.getLogger("httpx").setLevel(logging.WARNING)
    parser = argparse.ArgumentParser(prog="verathos-gateway")
    parser.add_argument("--config", type=Path, default=Path("gateway.json"))
    commands = parser.add_subparsers(dest="command", required=True)
    commands.add_parser("serve")
    db_parser = commands.add_parser("db")
    db_parser.add_argument("action", choices=["upgrade"])
    acc = commands.add_parser("accounts")
    acc.add_argument("action", choices=["create", "set", "list"])
    acc.add_argument("--name")
    acc.add_argument("--rate", type=float, help="requests per second")
    acc.add_argument("--burst", type=int)
    acc.add_argument("--monthly-quota", type=int)
    acc.add_argument("--status", choices=["active", "suspended"])
    acc.add_argument("--prepaid", choices=["yes", "no"], help="served only while the balance is positive")
    keys = commands.add_parser("keys")
    keys.add_argument("action", choices=["create", "list", "revoke"])
    keys.add_argument("--account")
    keys.add_argument("--name")
    keys.add_argument("--expires", help="ISO date or datetime (UTC unless an offset is given)")
    keys.add_argument("--prefix")
    usage = commands.add_parser("usage")
    usage.add_argument("--account", required=True)
    usage.add_argument("--since", help="ISO date; default: the first of this month")
    credit = commands.add_parser("credits")
    credit.add_argument("action", choices=["add", "list"])
    credit.add_argument("--account", required=True)
    credit.add_argument("--usd", type=float)
    credit.add_argument("--kind", choices=["grant", "topup", "adjust"])
    credit.add_argument("--reference", help="the payment or grant it records; recorded once")
    credit.add_argument("--note")
    args = parser.parse_args()
    config = GatewayConfig.load(args.config)
    if args.command == "serve":
        run(serve(config))
    elif args.command == "db":
        db.upgrade(config.database_url)
        print(f"database at revision {db.head_revision()}")
    else:
        if args.command == "accounts" and args.action == "create" and (args.rate is None or args.burst is None):
            parser.error("accounts create needs --name, --rate and --burst")
        if args.command == "credits" and args.action == "add" and None in (args.usd, args.kind, args.reference):
            parser.error("credits add needs --usd, --kind and --reference")
        try:
            print(json.dumps(run(admin(config, args)), indent=2, default=str))
        except ValidationError as exc:
            parser.error(validation_message(exc))
        except (ValueError, LookupError, Conflict) as exc:
            parser.error(str(exc))


if __name__ == "__main__":
    main()
