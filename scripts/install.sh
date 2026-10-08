#!/usr/bin/env bash
# Install or update a Verathos miner or validator (Ubuntu 22.04 or 24.04), then set it up.
#
#   curl -LsSf https://raw.githubusercontent.com/<org>/verathos/main/scripts/install.sh | bash -s -- miner
#   curl -LsSf https://raw.githubusercontent.com/<org>/verathos/main/scripts/install.sh | bash -s -- validator [--full]
#   bash ~/verathos-decide/scripts/install.sh uninstall [--purge]
#
# Installs the system packages it needs, uv, the code (in ~/verathos-decide) and its Python environment, then runs
# `verathos-miner init` or `verathos-validator init`, which asks a few questions (or takes --yes and options) and
# starts everything as system services. Run it again to update. It only uses a wallet's hotkey and coldkeypub.txt.
# With --no-service it needs no root and no sudo: it installs nothing system-wide (git and curl must be there; a
# miner also needs a C compiler), sets everything up in the user's home and prints the commands to run the services
# under a process manager the user runs (pm2 or another).
#
# The first argument is miner, validator or uninstall. Options:
#   --source <src>  git URL, path to a git bundle, or path to a source tarball (default: the public repository;
#                   when updating, where the installed code came from)
#   --ref <ref>     branch, tag or commit to install (default: the source's default branch)
#   --dir <dir>     where the code goes (default: the checkout this script is in, else ~/verathos if an earlier
#                   install put this code there, else ~/verathos-decide); another checkout there is never touched
#   --purge         with uninstall: also delete what the setups keep in ~/.verathos (configuration, state,
#                   measurements); other files there stay
#   --no-service    no system packages, services or sudo; print the commands to run instead
# Every other option goes to `verathos-<role> init`, for example:
#   --yes --wallet <name> --hotkey <name> --network finney|test|<ws url> --netuid <n>
#   --no-auto-update  the miner or validator does not install new releases by itself (on by default, except for the
#                     subnet owner's validator; --auto-update turns it on again)
#   validator mode: --light (the default, except for the subnet owner's hotkey: sets the auditor's published weights;
#                   no GPU, no reference engines, no requests to miners) or --full (checks miners on references)
#   miner: --public-ip <ip> --port <port> --model <id>[,<id>...] --gpus <i,j> --no-publish
#          --model: one model for every GPU or one per GPU in order, auto: the setup chooses (default: the
#                   active model that pays best on each GPU, measured from the network)
#          --worker [--name <name>]  (a worker machine of the hotkey; set up the workers first)
#          --peers <ip:port>[,...]   (on the primary: every worker; replaces the list)
#   validator: --device auto|cpu|gpu --dry-run --candidates (also references for candidate models)
#              --references resident|sequential  (full: every reference running, or one at a time, which needs
#                                                 memory for the largest only; default: resident if they fit)
set -euo pipefail

PUBLIC_REPOSITORY="https://github.com/<org>/verathos.git"
DEFAULT_DIR="$HOME/verathos-decide"
PYTHON_VERSION="3.12"
SOURCE_FILE=".install-source"  # where a checkout installed from a tarball came from

say() { printf '\n==> %s\n' "$*"; }
die() { printf '\nerror: %s\n' "$*" >&2; exit 1; }

usage() {
  if [ -f "${BASH_SOURCE[0]:-}" ]; then
    sed -n '2,/^set -euo/p' "${BASH_SOURCE[0]}" | sed '$d; s/^# \{0,1\}//'
  fi
  echo "usage: install.sh miner|validator|uninstall [options]"
}

main() {
  local role="" source="" ref="" dir="" purge="" no_service="" home="$HOME/.verathos"
  local -a init_args=()
  case "${1:-}" in
    miner|validator|uninstall) role="$1"; shift ;;
    -h|--help) usage; exit 0 ;;
    *) usage; exit 2 ;;
  esac
  while [ $# -gt 0 ]; do
    case "$1" in
      --source) source="${2:?--source needs a value}"; shift ;;
      --source=*) source="${1#*=}" ;;
      --ref) ref="${2:?--ref needs a value}"; shift ;;
      --ref=*) ref="${1#*=}" ;;
      --dir) dir="${2:?--dir needs a value}"; shift ;;
      --dir=*) dir="${1#*=}" ;;
      --purge) purge=1 ;;
      --no-service) no_service=1; init_args+=("$1") ;;
      --home) home="${2:?--home needs a value}"; shift ;;
      --home=*) home="${1#*=}" ;;
      -h|--help) usage; exit 0 ;;
      *) init_args+=("$1") ;;
    esac
    shift
  done
  dir="$(realpath -m -- "${dir:-$(default_dir)}")"
  home="$(realpath -m -- "$home")"
  [ "$home" = "$HOME/.verathos" ] || init_args+=(--home "$home")
  if [ "$role" = uninstall ]; then
    uninstall "$dir" "$home" "$purge"
    return
  fi
  check_dir "$dir"
  if [ -n "$no_service" ]; then
    user_tools "$role"
  else
    sudo_setup
    check_os
    system_packages "$role"
  fi
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

is_checkout() {
  # This code: its package name and the setup package, which other repositories of that name do not have.
  [ -f "$1/pyproject.toml" ] && grep -q '^name = "verathos"' "$1/pyproject.toml" \
    && [ -f "$1/src/verathos/install/__init__.py" ]
}

default_dir() {
  # The checkout this script is in (as when updating), else an earlier install in ~/verathos, else ~/verathos-decide.
  script_checkout && return
  if is_checkout "$HOME/verathos"; then
    printf '%s' "$HOME/verathos"
  else
    printf '%s' "$DEFAULT_DIR"
  fi
}

script_checkout() {
  # The checkout this script belongs to, when it runs from a file in one (as when updating).
  local here
  [ -f "${BASH_SOURCE[0]:-}" ] || return 1
  here="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)" || return 1
  is_checkout "$here" && printf '%s' "$here"
}

check_dir() {
  # The code goes into a missing or empty directory, or a checkout of this code; never into another one.
  [ -e "$1" ] && [ -n "$(ls -A "$1" 2>/dev/null)" ] || return 0
  is_checkout "$1" && return
  [ -e "$1/.git" ] && die "$1 is a checkout of other code; it is left as it is: choose another --dir"
  die "$1 exists and holds other files; choose another --dir"
}

NO_ROOT="or pass --no-service to set up without root, under a process manager you run (pm2 or another)"

sudo_setup() {
  SUDO=()
  [ "$(id -u)" -eq 0 ] && return
  command -v sudo >/dev/null || die "system services need root: run this as root or with sudo, $NO_ROOT"
  if ! sudo -n true 2>/dev/null && ! (exec </dev/tty) 2>/dev/null; then
    die "sudo asks for a password and there is no terminal to type it in: run this as root, $NO_ROOT"
  fi
  SUDO=(sudo)
}

user_tools() {
  # Without services nothing is installed system-wide: the tools the setup uses must be there.
  local -a missing=()
  command -v git >/dev/null || missing+=(git)
  command -v uv >/dev/null || [ -x "$HOME/.local/bin/uv" ] || command -v curl >/dev/null || missing+=(curl)
  # Miners: the fused engine kernels are compiled on first use.
  [ "$1" = miner ] && ! command -v cc >/dev/null && missing+=(build-essential)
  [ ${#missing[@]} -eq 0 ] && return
  die "missing: ${missing[*]}. With --no-service nothing is installed system-wide: have them installed (apt-get install -y ${missing[*]}), or run this without --no-service as root or with sudo"
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
  # In the user's home, as any user: ~/.local/bin/uv
  curl -LsSf https://astral.sh/uv/install.sh | env UV_INSTALL_DIR="$HOME/.local/bin" UV_NO_MODIFY_PATH=1 sh >/dev/null
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
  elif [ -z "$source" ] && [ -f "$dir/$SOURCE_FILE" ]; then
    source="$(cat "$dir/$SOURCE_FILE")"
  fi
  source="${source:-$PUBLIC_REPOSITORY}"
  case "$source" in
    *"<org>"*) die "no public repository is configured in this installer yet: pass --source <git url | git bundle | tarball>" ;;
  esac
  if [ -e "$source" ]; then
    source="$(realpath -- "$source")"
  fi
  if is_tarball "$source"; then
    install_tarball "$dir" "$source"
    return
  fi
  local partial="$dir.partial"
  if [ ! -d "$dir/.git" ]; then
    rm -rf "$partial"
    git clone --quiet --no-checkout "$source" "$partial" || die "cannot clone $source"
    [ -n "$ref" ] || ref="$(default_ref "$partial" "$source")"
    git -C "$partial" fetch --quiet --tags --force "$source" "$ref" || die "cannot fetch $ref from $source"
    git -C "$partial" checkout --quiet --detach FETCH_HEAD || die "cannot check out $ref"
    [ -d "$dir/.venv" ] && mv "$dir/.venv" "$partial/.venv"
    rm -rf "$dir"
    mv "$partial" "$dir"
  else
    git -C "$dir" remote set-url origin "$source"
    [ -n "$ref" ] || ref="$(default_ref "$dir" "$source")"
    git -C "$dir" fetch --quiet --tags --force "$source" "$ref" || die "cannot fetch $ref from $source"
    git -C "$dir" checkout --quiet --detach FETCH_HEAD \
      || die "local changes in $dir stop the update; see: git -C $dir status"
  fi
  say "Installed the code from $source ($ref) in $dir"
  echo "    $(git -C "$dir" log -1 --format='%h %s')"
}

install_tarball() {
  local dir="$1" source="$2" tmp found
  tmp="$(mktemp -d)"
  # shellcheck disable=SC2064  # the directory is fixed now
  trap "rm -rf '$tmp'" EXIT
  tar -xf "$source" -C "$tmp"
  found="$(find "$tmp" -maxdepth 2 -name pyproject.toml | head -n 1)"
  [ -n "$found" ] && is_checkout "$(dirname "$found")" || die "$source holds no source tree of this code"
  mkdir -p "$dir"
  find "$dir" -mindepth 1 -maxdepth 1 ! -name .venv -exec rm -rf {} +
  cp -a "$(dirname "$found")"/. "$dir"/
  printf '%s\n' "$source" > "$dir/$SOURCE_FILE"
  say "Installed the code from $source in $dir"
}

python_environment() {
  local dir="$1"
  say "Python environment"
  [ -x "$dir/.venv/bin/python" ] || uv venv --quiet -p "$PYTHON_VERSION" "$dir/.venv"
  VIRTUAL_ENV="$dir/.venv" uv pip install --quiet -e "$dir[server,chain]"
}

command_dir() {
  if [ "$(id -u)" -eq 0 ]; then echo /usr/local/bin; else echo "$HOME/.local/bin"; fi
}

link_commands() {
  local dir="$1" bin c
  bin="$(command_dir)"
  mkdir -p "$bin"
  for c in verathos-miner verathos-validator; do
    ln -sf "$dir/.venv/bin/$c" "$bin/$c"
  done
}

uninstall() {
  local dir="$1" home="$2" purge="$3" role bin c
  local -a home_arg=()
  [ "$home" = "$HOME/.verathos" ] || home_arg=(--home "$home")
  for role in miner validator; do
    if [ -f "$home/install/$role.json" ]; then
      [ -x "$dir/.venv/bin/verathos-$role" ] || die "the $role was set up from another directory: pass its --dir"
      say "Removing the $role"
      "$dir/.venv/bin/verathos-$role" uninstall ${purge:+--purge} "${home_arg[@]+"${home_arg[@]}"}"
    fi
  done
  bin="$(command_dir)"
  for c in verathos-miner verathos-validator; do
    if [ -L "$bin/$c" ] && [ "$(readlink "$bin/$c")" = "$dir/.venv/bin/$c" ]; then
      rm -f "$bin/$c"
    fi
  done
  if is_checkout "$dir"; then
    say "Removing the code in $dir"
    rm -rf "$dir"
  elif [ -e "$dir" ]; then
    echo "    $dir is not a checkout of this code; left as it is"
  fi
  if [ -n "$purge" ]; then
    # What the setups keep there; other programs may keep files in the same directory.
    for c in install catalog engines hf miner validator miner.json validator.json; do
      rm -rf "${home:?}/$c"
    done
    if rmdir "$home" 2>/dev/null; then
      echo "    removed $home"
    elif [ -e "$home" ]; then
      echo "    removed the configuration and state in $home; other files there stay"
    fi
  fi
  echo "    wallets in ~/.bittensor were not touched"
}

main "$@"
