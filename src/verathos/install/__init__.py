"""Guided setup of a miner or validator machine: ``verathos-miner init``, ``verathos-validator init`` and
``uninstall``.

``init`` checks the wallet and the chain, installs each catalog model's engine at its pinned revision, downloads the
pinned weights, writes the configuration and runs everything as system services (systemd, or supervisord where
there is no systemd, as in most containers). It keeps what it set up in ``<home>/install/<role>.json``, so running
it again updates the machine with the previous answers as defaults, and ``uninstall`` removes exactly that.
"""
