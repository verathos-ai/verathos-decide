# Security

This code decides who earns the subnet's emission and spends users' prepaid credit: a flaw that lets a miner score
without doing the work, steers validators' weights, forges a signature or a release, or uses credit that is not the
caller's is a vulnerability. Report it privately.

## Reporting

- By email to info@verathos.ai, or with GitHub's private vulnerability reporting: the repository's Security and
  quality tab, Report a vulnerability.
- Never in a public issue, pull request or chat before a release fixes it.
- With what it affects (component, release or commit), how to reproduce it, its impact, and a fix if you have one.
- Reproduce it on testnet (netuid 405) or on machines of your own; never against mainnet (netuid 96) or other
  people's machines, funds or data.

The maintainers confirm a report within three working days, keep its reporter informed, and agree with them on when
it is published. The fix ships as a signed release, which nodes with automatic updates install within about two
hours of its tag; a security advisory then describes the issue and credits the reporter, unless they prefer
otherwise.

## Scope

This repository and the releases and Docker images built from it, in particular:

- verification and scoring: answers that pass the checks without the model's weights, the throughput audit
  crediting capacity an endpoint does not have, one machine counted as several, canaries told apart from users'
  requests;
- weights, the auditor's verdicts and records, the usage and referral pools;
- signatures and TLS: manifests, receipts, request signing, certificate pins, release tags and the self-update;
- the gateway: API keys, credit, limits, image decoding;
- the installer, the services it sets up and the miner's firewall;
- anything that exposes keys, tokens or wallet files.

Not in scope: the Bittensor chain and SDK and the models' repositories (report to their maintainers), denial of
service by traffic volume alone, and a miner competing within the rules. The services at verathos.ai are not in this
repository; their vulnerabilities go to the same address.

Only the latest release is supported.
