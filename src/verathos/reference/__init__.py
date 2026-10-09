"""A reference machine: a GPU machine that lends compute to validators on other machines.

It runs float32 reference engines and bfloat16 screens on its loopback and serves them through one TLS front
(``verathos-reference``) to its callers' addresses only, with a bearer key. Validators reach them as remote engines
(``validator.config.ReferenceEngine``: URL, the front's TLS pin, the key). It runs no validator, signs nothing and holds
no wallet. ``verathos-reference init`` sets it up (``install.reference``).
"""
