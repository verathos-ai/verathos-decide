<!-- Pull requests go to dev (CONTRIBUTING.md). Report a vulnerability privately (SECURITY.md), not here. -->

## Change

<!-- What the code does after this change, and why. The issue it resolves. -->

## Nodes

<!-- What changes for miners, validators, reference machines, the gateway or the auditor; how nodes on the previous
release work with nodes on this one while they update; anything operators have to do. "None" if nothing. -->

## Tests

<!-- The tests this adds or changes, and anything checked beyond them (testnet, a GPU). -->

- [ ] The base branch is `dev`.
- [ ] `.venv/bin/python -m pytest`, `.venv/bin/ruff check src tests scripts` and, for a change in `native/loadgen`,
      `cargo test` pass.
- [ ] The docs say what the code now does.
