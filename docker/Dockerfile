# syntax=docker/dockerfile:1
# A release's images (docs/DOCKER.md); docker/build.sh builds and tags them from a release:
#   verathos          target node:    the code; a light validator, a full validator or the auditor with its engines on
#                                     reference machines (the load generator built in)
#   verathos-engines  target engines: the code and the engines its shipped catalog pins; a miner, a reference machine
# Build context: the repository root, docker build -f docker/Dockerfile --target <node|engines> .

ARG UBUNTU=ubuntu:24.04
ARG UV=ghcr.io/astral-sh/uv:0.10.9
# The load generator's toolchain: the channel native/loadgen/rust-toolchain.toml names
ARG RUST=rust:1.85-slim-bookworm

FROM ${UV} AS uv

FROM ${RUST} AS loadgen
ARG JOBS=
WORKDIR /loadgen
COPY native/loadgen/ ./
# The image's toolchain is the pinned channel; naming it keeps rustup from fetching that channel again.
RUN toolchain="$(rustup default)" \
    && RUSTUP_TOOLCHAIN="${toolchain%% *}" cargo build --release --locked ${JOBS:+--jobs "$JOBS"} \
    && cp target/release/verathos-loadgen /verathos-loadgen

FROM ${UBUNTU} AS base
ENV DEBIAN_FRONTEND=noninteractive \
    LANG=C.UTF-8 \
    PYTHONUNBUFFERED=1 \
    UV_PYTHON_INSTALL_DIR=/opt/uv/python \
    UV_PYTHON_PREFERENCE=only-managed \
    VERATHOS_DATA=/data \
    PATH=/opt/verathos/.venv/bin:${PATH}
# The packages follow the release's updates (the base image pins the release).
# hadolint ignore=DL3008
RUN apt-get update \
    && apt-get install -y --no-install-recommends ca-certificates tini \
    && rm -rf /var/lib/apt/lists/*
COPY --from=uv /uv /usr/local/bin/uv
# Python 3.12 as the installer gets it: uv's own build, with the headers Triton compiles against
RUN uv python install 3.12
ENTRYPOINT ["/usr/bin/tini", "--", "verathos-container"]

# What the engines' environments depend on, read from the release (docker/pins.py): the layer installing them is
# keyed by its content, so a release that changes neither keeps that layer.
FROM base AS pins
COPY pyproject.toml docker/pins.py /pins/
COPY src/verathos/catalog/catalog.json /pins/
RUN "$(uv python find 3.12)" /pins/pins.py /pins/pyproject.toml /pins/catalog.json /pins/out

FROM base AS engines
ARG JOBS=
# Triton compiles its launchers with a C compiler when a kernel first runs; the Kev engine is a git checkout; the
# miner's firewall keeps its kernel table with nft.
# hadolint ignore=DL3008
RUN apt-get update \
    && apt-get install -y --no-install-recommends build-essential git nftables \
    && rm -rf /var/lib/apt/lists/*
ENV NVIDIA_DRIVER_CAPABILITIES=compute,utility \
    TRITON_CACHE_DIR=/data/cache/triton
COPY --from=pins /pins/out/ /pins/
# The engines' environments, filled as engines.KevEngine and engines.ClefEngine fill them, in one layer: files the
# environments share (most of the CUDA libraries) are stored once. The installer's code installs every engine again
# below and finds these in place.
RUN set -eu; \
    if [ -n "${JOBS}" ]; then export UV_CONCURRENT_INSTALLS="${JOBS}" UV_CONCURRENT_BUILDS="${JOBS}"; fi; \
    while read -r source revision; do \
        dir="/opt/engines/kev-$(printf '%.12s' "${revision}")"; \
        git clone --quiet "${source}" "${dir}"; \
        git -C "${dir}" checkout --quiet --force --detach "${revision}"; \
        UV_PROJECT_ENVIRONMENT="${dir}/.venv-reference" uv --directory "${dir}" sync --quiet --extra serve -p 3.12; \
        UV_PROJECT_ENVIRONMENT="${dir}/.venv" uv --directory "${dir}" sync --quiet --inexact --extra serve -p 3.12; \
        fla="$(sed -n 's/^FLA_VERSION[[:space:]]*=[[:space:]]*"\([^"]*\)".*/\1/p' "${dir}/kev/fused_qwen35.py" \
            2>/dev/null || true)"; \
        if [ -n "${fla}" ]; then \
            uv pip install --quiet --python "${dir}/.venv/bin/python" "flash-linear-attention==${fla}"; \
        fi; \
    done < /pins/kev-engines.txt; \
    for environment in .venv-reference .venv; do \
        uv venv --quiet -p 3.12 "/opt/engines/verathos-clef/${environment}"; \
        uv pip install --quiet --python "/opt/engines/verathos-clef/${environment}/bin/python" \
            -r /pins/clef-requirements.txt; \
    done; \
    rm -rf /root/.cache/uv
WORKDIR /opt/verathos
# The code's environment as install.sh makes it: its dependencies first, then the code (only its layers change with
# a release that changes the code alone), then every engine of the shipped catalog by the installer's own code.
COPY pyproject.toml ./
RUN uv venv --quiet -p 3.12 .venv \
    && VIRTUAL_ENV=/opt/verathos/.venv uv pip install --quiet -r pyproject.toml --extra server --extra chain \
    && rm -rf /root/.cache/uv
COPY . ./
RUN VIRTUAL_ENV=/opt/verathos/.venv uv pip install --quiet -e "/opt/verathos[server,chain]" \
    && verathos-container engines \
    && rm -rf /root/.cache/uv
COPY --from=loadgen /verathos-loadgen /opt/verathos/src/verathos/throughput/verathos-loadgen
WORKDIR /data
ARG VERSION=
ARG REVISION=
LABEL org.opencontainers.image.title="verathos-engines" \
      org.opencontainers.image.description="Verathos: the code and the engines its shipped catalog pins" \
      org.opencontainers.image.version="${VERSION}" \
      org.opencontainers.image.revision="${REVISION}" \
      org.opencontainers.image.licenses="MIT"
CMD ["miner"]

FROM base AS node
WORKDIR /opt/verathos
COPY pyproject.toml ./
RUN uv venv --quiet -p 3.12 .venv \
    && VIRTUAL_ENV=/opt/verathos/.venv uv pip install --quiet -r pyproject.toml --extra server --extra chain \
    && rm -rf /root/.cache/uv
COPY . ./
RUN VIRTUAL_ENV=/opt/verathos/.venv uv pip install --quiet -e "/opt/verathos[server,chain]" \
    && rm -rf /root/.cache/uv
COPY --from=loadgen /verathos-loadgen /opt/verathos/src/verathos/throughput/verathos-loadgen
WORKDIR /data
ARG VERSION=
ARG REVISION=
LABEL org.opencontainers.image.title="verathos" \
      org.opencontainers.image.description="Verathos: the code, for validators and the auditor" \
      org.opencontainers.image.version="${VERSION}" \
      org.opencontainers.image.revision="${REVISION}" \
      org.opencontainers.image.licenses="MIT"
CMD ["validator"]
