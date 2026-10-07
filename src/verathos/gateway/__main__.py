"""``verathos-gateway``: run the gateway, and administer its database.

    verathos-gateway serve --config gateway.json
    verathos-gateway db upgrade --config gateway.json
    verathos-gateway accounts create --name acme --rate 50 --burst 100 [--monthly-quota 1000000]
    verathos-gateway accounts set --name acme [--rate R] [--burst B] [--monthly-quota Q] [--status suspended]
    verathos-gateway accounts list
    verathos-gateway keys create --account acme --name prod [--expires 2027-01-01]
    verathos-gateway keys list --account acme
    verathos-gateway keys revoke --prefix vd_1a2b3c4
    verathos-gateway usage --account acme [--since 2026-10-01]
"""

from __future__ import annotations

import argparse
import asyncio
import datetime as dt
import json
import logging
import random
from pathlib import Path
from typing import Any

import bittensor as bt
import httpx
import uvicorn

from verathos import catalog as catalog_module
from verathos.gateway import db
from verathos.gateway.accounts import Accounts
from verathos.gateway.app import Gateway, create_app
from verathos.gateway.config import GatewayConfig
from verathos.gateway.limits import RateLimiter
from verathos.gateway.router import Router
from verathos.gateway.routes import RouteTable
from verathos.gateway.sampler import Sampler
from verathos.gateway.upstream import Upstream
from verathos.gateway.usage import UsageMeter
from verathos.protocol.gateway import Routes
from verathos.throughput.chain import SubtensorChainView

log = logging.getLogger("verathos.gateway")


async def serve(config: GatewayConfig) -> None:
    engine = db.create_engine(config.database_url)
    if await db.schema_revision(engine) != db.head_revision():
        raise SystemExit("the database schema is not current: run `verathos-gateway db upgrade` first")
    keypair = bt.Wallet(name=config.wallet_name, hotkey=config.hotkey_name).hotkey
    chain = SubtensorChainView(config.network, config.netuid)
    st = await asyncio.to_thread(bt.Subtensor, network=config.network)
    auditor_hotkey = await asyncio.to_thread(st.get_subnet_owner_hotkey, config.netuid)
    st.close()

    async def auditor_url() -> str | None:
        return config.auditor_url or await asyncio.to_thread(chain.auditor_url)

    client = httpx.AsyncClient(limits=httpx.Limits(max_connections=None, max_keepalive_connections=4096),
                               timeout=httpx.Timeout(30.0, connect=3.0))
    state_dir = config.state_dir.expanduser()
    router = Router(share=1 / config.instances, max_latency_s=2.0)
    routes = RouteTable(client=client, auditor_url=auditor_url, auditor_hotkey=auditor_hotkey, netuid=config.netuid,
                        path=state_dir / "routes.json")
    store = catalog_module.CatalogStore(config.catalog_dir)
    current: list[catalog_module.Catalog | None] = [None]

    async def adopt(table: Routes) -> None:
        cat = current[0]
        if cat is None or cat.digest() != table.catalog_digest:
            url = await auditor_url()
            sources = [url.rstrip("/") + catalog_module.CATALOG_PATH] if url else []
            cat = await asyncio.to_thread(catalog_module.resolve, table.catalog_digest, store=store, sources=sources)
            current[0] = cat
            if keypair.ss58_address not in cat.gateways:
                log.error("this gateway's hotkey %s is not among catalog %s's gateways: miners will refuse it",
                          keypair.ss58_address, cat.digest()[:12])
        router.max_latency_s = cat.audit_policy().max_latency_s
        router.update(table)
        log.info("routes of epoch %d: %d endpoints for %s", table.epoch, len(table.routes),
                 ", ".join(sorted(router.models())) or "no model")

    routes.on_change.append(adopt)
    meter = UsageMeter(engine)
    accounts = Accounts(engine, meter)
    sampler = Sampler(keypair, client, auditor_url=auditor_url, auditor_hotkey=auditor_hotkey,
                      window_s=config.sample_window_s, rng=random.SystemRandom())
    gateway = Gateway(config=config, hotkey=keypair.ss58_address, catalog=lambda: current[0], accounts=accounts,
                      meter=meter, router=router, sampler=sampler, routes_loaded=lambda: routes.table is not None,
                      upstream=Upstream(keypair, client, netuid=config.netuid, mechid=config.mechid),
                      limiter=RateLimiter())
    await routes.load()
    if config.metrics_port:
        from prometheus_client import start_http_server
        start_http_server(config.metrics_port, addr="127.0.0.1")
    tasks = [asyncio.create_task(t) for t in (routes.run(), meter.run(), sampler.run())]
    server = uvicorn.Server(uvicorn.Config(create_app(gateway), host=config.listen_host, port=config.port,
                                           log_level="warning", access_log=False, server_header=False))
    log.info("gateway %s on %s:%d (auditor %s, %d instance(s))", keypair.ss58_address, config.listen_host,
             config.port, auditor_hotkey, config.instances)
    try:
        await server.serve()
    finally:
        for task in tasks:
            task.cancel()
        await asyncio.gather(*tasks, return_exceptions=True)
        for final in (meter.flush(), sampler.flush()):
            try:
                await final
            except Exception:
                log.exception("final flush failed")
        await client.aclose()
        await engine.dispose()


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
        if args.command == "accounts":
            if args.action == "create":
                return {"id": await accounts.create_account(args.name, rate_rps=args.rate, burst=args.burst,
                                                            monthly_quota=args.monthly_quota)}
            if args.action == "set":
                values = {k: v for k, v in (("rate_rps", args.rate), ("burst", args.burst),
                                            ("monthly_quota", args.monthly_quota), ("status", args.status))
                          if v is not None}
                return {"updated": await accounts.set_account(args.name, **values)}
            return await accounts.list_accounts()
        if args.command == "keys":
            if args.action == "create":
                expires = dt.datetime.fromisoformat(args.expires).replace(tzinfo=dt.UTC) if args.expires else None
                return {"key": await accounts.create_key(args.account, args.name, expires),
                        "note": "shown once; store it now"}
            if args.action == "revoke":
                return {"revoked": await accounts.revoke_key(args.prefix)}
            return await accounts.list_keys(args.account)
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
    keys = commands.add_parser("keys")
    keys.add_argument("action", choices=["create", "list", "revoke"])
    keys.add_argument("--account")
    keys.add_argument("--name")
    keys.add_argument("--expires", help="ISO date or datetime, UTC")
    keys.add_argument("--prefix")
    usage = commands.add_parser("usage")
    usage.add_argument("--account", required=True)
    usage.add_argument("--since", help="ISO date; default: the first of this month")
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
        print(json.dumps(run(admin(config, args)), indent=2, default=str))


if __name__ == "__main__":
    main()
