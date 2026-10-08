"""The subnet's gateway: user traffic to the miners' proven capacity.

A user sends a SystemOne request (``POST /v1/systemone``) with an API key. The gateway checks the key, the account's
rate and quota and the request, picks an endpoint serving the model among the auditor's routes (by proven capacity,
current load and health), signs the request with its hotkey (a gateway the catalog names, which miners serve without
the per-caller limit) and verifies the miner's receipt before returning the answers. A failed attempt is retried on
another miner within the request's deadline. Usage is counted per account, key, model and day. One exchange per
endpoint and sampling window, picked after the answers are in, goes to the auditor, which recomputes it: a wrong
answer is proof against the miner.

    config.py     configuration
    db.py         tables, the database engine and its schema check (migrations in ``migrations/``)
    accounts.py   accounts and API keys, with a short-lived cache
    limits.py     rate limits per key (GCRA)
    usage.py      usage and its cost counted in memory and written in batches; monthly quotas and prepaid credit
    routes.py     the auditor's signed routes, polled and kept
    router.py     endpoint choice: proven capacity, load, circuit breakers
    upstream.py   signed requests to miners, receipt checks
    sampler.py    the exchanges submitted to the auditor
    served.py     what the miners did for users, per endpoint and second, reported to the auditor
    stats.py      what users get from each model, for the public status
    app.py        the HTTP API
    admin.py      the operators' API: inspection, debugging, accounts, keys and credit
    logs.py       log lines that repeat with the traffic, throttled
    metrics.py    Prometheus metrics
"""
