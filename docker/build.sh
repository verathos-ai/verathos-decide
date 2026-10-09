#!/usr/bin/env bash
# Build a release's Docker images from the git repository this script is in (docs/DOCKER.md, Building the images):
#   <registry>/verathos:<tag>           the code (docker/Dockerfile, target node)
#   <registry>/verathos-engines:<tag>   the code and the engines its shipped catalog pins (target engines)
#
#   bash docker/build.sh v1.2.0           local/verathos:1.2.0 and local/verathos-engines:1.2.0, for this machine
#   bash docker/build.sh v1.2.0 --push    ghcr.io/verathos-ai/..., pushed once built
#
# The first argument is what to build:
#   v<version>      a release: built only when its tag is signed by a key that release/allowed_signers of this
#                   checkout lists, as the installer checks releases; tagged <version>
#   <branch|commit> built as it is (unverified), tagged <version>-<commit, 12 digits>
#   --working-tree  this checkout as it is, uncommitted changes too, tagged <version>-dev
# Each image has its version's tag only, never latest.
# Options:
#   --push          a release only: tag for the registry and push the tags once the images are built
#   --registry <r>  the registry and owner the tags name (default: local, for this machine; with --push,
#                   ghcr.io/verathos-ai)
#   --only <target> node or engines
#   --jobs <n>      parallel jobs of the compilers and installers in the build (default: theirs)
#   --dry-run       print the commands instead of running them
set -euo pipefail

REPO="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
RELEASE_TAG='^v[0-9]+\.[0-9]+\.[0-9]+$'
REGISTRY="ghcr.io/verathos-ai"  # where releases are published (docs/DOCKER.md)
RELEASE_KEY='^[^#]*[[:space:]](ssh-ed25519|ssh-rsa|ecdsa-sha2-nistp(256|384|521)|sk-ssh-ed25519@openssh\.com|sk-ecdsa-sha2-nistp256@openssh\.com)[[:space:]]+AAAA[A-Za-z0-9+/]+=*([[:space:]]|$)'
DRY=""

say() { printf '\n==> %s\n' "$*"; }
die() { printf '\nerror: %s\n' "$*" >&2; exit 1; }

usage() {
  sed -n '2,/^set -euo/p' "${BASH_SOURCE[0]}" | sed '$d; s/^# \{0,1\}//'
}

run() {
  # A step of the build; with --dry-run, printed instead.
  if [ -n "$DRY" ]; then
    printf '%q ' "$@" | sed 's/ $//'
    echo
  else
    "$@"
  fi
}

verify_release() {
  # The tag, signed with SSH by a key release/allowed_signers lists; else nothing is built.
  local tag="$1" signers="$REPO/release/allowed_signers" object out status=0
  grep -Eq "$RELEASE_KEY" "$signers" \
    || die "cannot verify $tag: $signers lists no release signing key yet; build its commit instead, unverified ($(git -C "$REPO" rev-parse --short=12 "$tag^{commit}"))"
  object="$(git -C "$REPO" rev-parse "refs/tags/$tag")"
  [ "$(git -C "$REPO" cat-file -t "$object")" = tag ] || die "$tag is not signed (a lightweight tag): not building it"
  [ "$(git -C "$REPO" cat-file tag "$object" | sed -n '/^$/q; s/^tag //p')" = "$tag" ] \
    || die "$tag names the tag object of another tag: not building it"
  out="$(git -C "$REPO" -c gpg.ssh.allowedSignersFile="$signers" -c gpg.ssh.program=ssh-keygen -c gpg.program=false \
    -c gpg.x509.program=false verify-tag "$object" 2>&1)" || status=$?
  if [ "$status" -ne 0 ] || ! grep -q '^Good "git" signature for ' <<<"$out"; then
    die "$tag is not signed by a release key ($(tail -n 1 <<<"$out")): not building it"
  fi
  echo "    $tag: $(grep '^Good "git" signature for ' <<<"$out")"
}

main() {
  local what="" registry="" push="" only="" jobs="" release="" tmp=""
  while [ $# -gt 0 ]; do
    case "$1" in
      -h|--help) usage; exit 0 ;;
      --registry) registry="${2:?--registry needs a value}"; shift ;;
      --registry=*) registry="${1#*=}" ;;
      --push) push=1 ;;
      --only) only="${2:?--only needs node or engines}"; shift ;;
      --only=*) only="${1#*=}" ;;
      --jobs) jobs="${2:?--jobs needs a number}"; shift ;;
      --jobs=*) jobs="${1#*=}" ;;
      --dry-run) DRY=1 ;;
      -*) [ "$1" = --working-tree ] || { usage >&2; exit 2; }
          [ -z "$what" ] || die "one thing to build: $what or $1"; what="$1" ;;
      *) [ -z "$what" ] || die "one thing to build: $what or $1"; what="$1" ;;
    esac
    shift
  done
  [ -n "$what" ] || { usage >&2; exit 2; }
  case "$only" in ""|node|engines) ;; *) die "--only takes node or engines, not $only" ;; esac
  [ -z "$jobs" ] || [[ "$jobs" =~ ^[1-9][0-9]*$ ]] || die "--jobs takes a number, not $jobs"
  if [ -z "$registry" ]; then
    registry=local
    [ -z "$push" ] || registry="$REGISTRY"
  fi
  [[ "$registry" =~ ^[a-z0-9][a-z0-9._:/-]*[a-z0-9]$ ]] \
    || die "--registry: a registry and owner in lower case, as in $REGISTRY; not $registry"
  [ -z "$push" ] || [ "$registry" != local ] || die "--push needs a registry, not local"
  [ -n "$DRY" ] || command -v docker >/dev/null || die "docker not found"

  local context commit
  if [ "$what" = --working-tree ]; then
    context="$REPO"
    commit="$(git -C "$REPO" rev-parse HEAD 2>/dev/null || echo unknown)"
    if [ -n "$(git -C "$REPO" status --porcelain 2>/dev/null)" ]; then
      echo "    note: $REPO has uncommitted changes; they are built too"
    fi
  else
    commit="$(git -C "$REPO" rev-parse --verify --quiet "$what^{commit}")" \
      || die "$what: no such tag, branch or commit in $REPO"
    if [[ "$what" =~ $RELEASE_TAG ]] && git -C "$REPO" show-ref --verify --quiet "refs/tags/$what"; then
      verify_release "$what"
      release=1
    else
      echo "    note: $what is not a release tag; it is built as it is, unverified"
    fi
    tmp="$(mktemp -d)"
    # shellcheck disable=SC2064  # the directory is fixed now
    trap "rm -rf '$tmp'" EXIT
    git -C "$REPO" archive --format=tar "$commit" | tar -x -C "$tmp"
    context="$tmp"
  fi
  [ -f "$context/docker/Dockerfile" ] || die "$what holds no docker/Dockerfile"
  local version tag
  version="$(sed -n 's/^__version__ = "\([^"]*\)".*/\1/p' "$context/src/verathos/__init__.py")"
  [ -n "$version" ] || die "$what names no version in src/verathos/__init__.py"
  if [ -n "$release" ]; then
    [ "v$version" = "$what" ] || die "$what holds version $version: not building it"
    tag="$version"
  elif [ "$what" = --working-tree ]; then
    tag="$version-dev"
  else
    tag="$version-${commit:0:12}"
  fi
  [ -z "$push" ] || [ -n "$release" ] \
    || die "--push is for a release (v<version>, its tag signed by a release key): $what builds images for this machine"

  local -a args=(--build-arg "VERSION=$version" --build-arg "REVISION=$commit") built=()
  [ -z "$jobs" ] || args+=(--build-arg "JOBS=$jobs")
  local target name
  for target in node engines; do
    [ -z "$only" ] || [ "$only" = "$target" ] || continue
    name="verathos"
    [ "$target" = node ] || name="verathos-$target"
    say "Building $registry/$name:$tag ($what, $commit)"
    run docker build -f "$context/docker/Dockerfile" --target "$target" -t "$registry/$name:$tag" "${args[@]}" \
      "$context"
    built+=("$registry/$name:$tag")
  done
  if [ -n "$push" ]; then
    say "Pushing"
    for name in "${built[@]}"; do
      run docker push "$name"
    done
  fi
  say "Images"
  printf '    %s\n' "${built[@]}"
}

main "$@"
