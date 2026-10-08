#!/usr/bin/env bash
# Install or update a Verathos miner or validator (Ubuntu 22.04 or 24.04), then set it up.
#
#   curl -LsSf https://raw.githubusercontent.com/<org>/verathos/main/scripts/install.sh | bash -s -- miner
#   curl -LsSf https://raw.githubusercontent.com/<org>/verathos/main/scripts/install.sh | bash -s -- validator
#   bash ~/verathos/scripts/install.sh uninstall [--purge]
#
# Installs the system packages it needs, uv, the code (in ~/verathos) and its Python environment, then runs
# `verathos-miner init` or `verathos-validator init`, which asks a few questions (or takes --yes and options) and
# starts everything as system services. Run it again to update. It only uses a wallet's hotkey and coldkeypub.txt.
#
# Options:
#   --source <src>  git URL, path to a git bundle, or path to a source tarball (default: the public repository;
#                   when updating, where the installed code came from)
#   --ref <ref>     branch, tag or commit to install (default: the source's default branch)
#   --dir <dir>     where the code goes (default ~/verathos)
#   --purge         with uninstall: also delete ~/.verathos (configuration, state, measurements)
# Every other option goes to `verathos-<role> init`, for example:
#   --yes --wallet <name> --hotkey <name> --network finney|test|<ws url> --netuid <n>
#   miner: --public-ip <ip> --port <port> --model <id>[,<id>...] --gpus <i,j> --no-publish
#   validator: --device auto|cpu|gpu --dry-run
set -euo pipefail

PUBLIC_REPOSITORY="https://github.com/<org>/verathos.git"
PYTHON_VERSION="3.12"

say() { printf '\n==> %s\n' "$*"; }
die() { printf '\nerror: %s\n' "$*" >&2; exit 1; }

usage() {
  sed -n '2,/^set -euo/p' "$0" 2>/dev/null | sed '$d; s/^# \{0,1\}//' || true
  echo "usage: install.sh miner|validator|uninstall [options]"
}

main() {
  local role="" source="" ref="" dir="$HOME/verathos" purge="" home="$HOME/.verathos"
  local -a init_args=()
  while [ $# -gt 0 ]; do
    case "$1" in
      miner|validator|uninstall) [ -z "$role" ] || die "one of miner, validator, uninstall"; role="$1" ;;
      --source) source="${2:?--source needs a value}"; shift ;;
      --source=*) source="${1#*=}" ;;
      --ref) ref="${2:?--ref needs a value}"; shift ;;
      --ref=*) ref="${1#*=}" ;;
      --dir) dir="${2:?--dir needs a value}"; shift ;;
      --dir=*) dir="${1#*=}" ;;
      --purge) purge=1 ;;
      --home) home="${2:?--home needs a value}"; init_args+=("$1" "$2"); shift ;;
      -h|--help) usage; exit 0 ;;
      *) init_args+=("$1") ;;
    esac
    shift
  done
  [ -n "$role" ] || { usage; exit 2; }
  dir="$(absolute "$dir")"
  sudo_setup
  if [ "$role" = uninstall ]; then
    uninstall "$dir" "$home" "$purge" "${init_args[@]+"${init_args[@]}"}"
    return
  fi
  check_os
  system_packages "$role"
  install_uv
  fetch_code "$dir" "$source" "$ref"
  python_environment "$dir"
  link_commands "$dir"
  say "Setting up the $role"
  local -a cmd=("$dir/.venv/bin/verathos-$role" init "${init_args[@]+"${init_args[@]}"}")
  if (exec </dev/tty) 2>/dev/null; then
    "${cmd[@]}" </dev/tty
  else
    "${cmd[@]}" </dev/null
  fi
}

absolute() {
  case "$1" in
    /*) printf '%s' "$1" ;;
    *) printf '%s/%s' "$PWD" "$1" ;;
  esac
}

sudo_setup() {
  SUDO=()
  if [ "$(id -u)" -ne 0 ]; then
    command -v sudo >/dev/null || die "run this as root, or install sudo"
    SUDO=(sudo)
  fi
}

check_os() {
  command -v apt-get >/dev/null || die "this installer needs apt-get (Ubuntu 22.04 or 24.04); see the manual installation in the docs"
  local id="" version=""
  if [ -r /etc/os-release ]; then
    id="$(. /etc/os-release && echo "${ID:-}")"
    version="$(. /etc/os-release && echo "${VERSION_ID:-}")"
  fi
  case "$id $version" in
    "ubuntu 22.04"|"ubuntu 24.04") ;;
    *) echo "note: made for Ubuntu 22.04 and 24.04; this is ${id:-unknown} ${version:-}, continuing" ;;
  esac
}

system_packages() {
  local -a wanted=(ca-certificates curl git) missing=()
  # Miners: the fused engine kernels are compiled on first use.
  [ "$1" = miner ] && wanted+=(build-essential)
  # Containers without systemd run the services under supervisord.
  [ -d /run/systemd/system ] || wanted+=(supervisor)
  local p
  for p in "${wanted[@]}"; do
    dpkg-query -W -f='${Status}' "$p" 2>/dev/null | grep -q "install ok installed" || missing+=("$p")
  done
  [ ${#missing[@]} -eq 0 ] && return
  say "Installing system packages: ${missing[*]}"
  "${SUDO[@]+"${SUDO[@]}"}" env DEBIAN_FRONTEND=noninteractive apt-get update -qq
  "${SUDO[@]+"${SUDO[@]}"}" env DEBIAN_FRONTEND=noninteractive apt-get install -y -qq --no-install-recommends "${missing[@]}" >/dev/null
}

install_uv() {
  export PATH="$HOME/.local/bin:$PATH"
  command -v uv >/dev/null && return
  say "Installing uv"
  curl -LsSf https://astral.sh/uv/install.sh | env UV_NO_MODIFY_PATH=1 sh >/dev/null
  command -v uv >/dev/null || die "uv was not installed"
}

is_tarball() {
  [ -f "$1" ] && ! head -c 32 "$1" | grep -q "git bundle" && tar -tf "$1" >/dev/null 2>&1
}

default_ref() {
  # The source's HEAD if it has one (a URL, or a bundle made with HEAD), else its first branch.
  local refs
  refs="$(git -C "$1" ls-remote "$2")" || die "cannot read $2"
  if printf '%s\n' "$refs" | awk '$2 == "HEAD" { found = 1 } END { exit !found }'; then
    echo HEAD
    return
  fi
  printf '%s\n' "$refs" | awk '$2 ~ /^refs\/heads\// { print $2; exit }' | grep . || die "no branch in $2"
}

fetch_code() {
  local dir="$1" source="$2" ref="$3"
  if [ -z "$source" ] && [ -d "$dir/.git" ]; then
    source="$(git -C "$dir" remote get-url origin 2>/dev/null || true)"
  fi
  source="${source:-$PUBLIC_REPOSITORY}"
  [ -e "$source" ] && source="$(absolute "$source")"
  case "$source" in
    *"<org>"*) die "no public repository is configured in this installer yet: pass --source <git url | git bundle | tarball>" ;;
  esac
  [ -e "$dir" ] && [ ! -f "$dir/pyproject.toml" ] && [ -n "$(ls -A "$dir" 2>/dev/null)" ] \
    && die "$dir exists and is not a Verathos checkout; choose another --dir"
  if is_tarball "$source"; then
    say "Installing the code from $source in $dir"
    local tmp root
    tmp="$(mktemp -d)"
    tar -xf "$source" -C "$tmp"
    root="$(dirname "$(find "$tmp" -maxdepth 2 -name pyproject.toml | head -n 1)")"
    [ -f "$root/pyproject.toml" ] || die "$source holds no pyproject.toml"
    mkdir -p "$dir"
    find "$dir" -mindepth 1 -maxdepth 1 ! -name .venv -exec rm -rf {} +
    cp -a "$root"/. "$dir"/
    rm -rf "$tmp"
    return
  fi
  if [ ! -d "$dir/.git" ]; then
    rm -rf "$dir.partial"
    git clone --quiet --no-checkout "$source" "$dir.partial" || die "cannot clone $source"
    [ -d "$dir/.venv" ] && mv "$dir/.venv" "$dir.partial/.venv"
    rm -rf "$dir"
    mv "$dir.partial" "$dir"
  fi
  git -C "$dir" remote set-url origin "$source"
  [ -n "$ref" ] || ref="$(default_ref "$dir" "$source")"
  say "Installing the code from $source ($ref) in $dir"
  git -C "$dir" fetch --quiet --tags --force "$source" "$ref" || die "cannot fetch $ref from $source"
  git -C "$dir" checkout --quiet --detach FETCH_HEAD \
    || die "local changes in $dir stop the update; see: git -C $dir status"
  echo "    $(git -C "$dir" log -1 --format='%h %s')"
}

python_environment() {
  local dir="$1"
  say "Python environment"
  [ -x "$dir/.venv/bin/python" ] || uv venv --quiet -p "$PYTHON_VERSION" "$dir/.venv"
  VIRTUAL_ENV="$dir/.venv" uv pip install --quiet -e "$dir[server,chain]"
}

link_commands() {
  local dir="$1" bin="/usr/local/bin" c
  [ "$(id -u)" -eq 0 ] || bin="$HOME/.local/bin"
  mkdir -p "$bin"
  for c in verathos-miner verathos-validator; do
    ln -sf "$dir/.venv/bin/$c" "$bin/$c"
  done
}

uninstall() {
  local dir="$1" home="$2" purge="$3"
  shift 3
  local role bin="/usr/local/bin" c
  for role in miner validator; do
    if [ -x "$dir/.venv/bin/verathos-$role" ] && [ -f "$home/install/$role.json" ]; then
      say "Removing the $role"
      "$dir/.venv/bin/verathos-$role" uninstall ${purge:+--purge} "$@"
    fi
  done
  [ "$(id -u)" -eq 0 ] || bin="$HOME/.local/bin"
  for c in verathos-miner verathos-validator; do
    [ -L "$bin/$c" ] && rm -f "$bin/$c"
  done
  say "Removing the code in $dir"
  rm -rf "$dir"
  if [ -n "$purge" ]; then
    rm -rf "$home"
    echo "    removed $home"
  fi
  echo "    wallets in ~/.bittensor were not touched"
}

main "$@"
