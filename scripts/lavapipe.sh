#!/bin/sh
set -eu

if [ "$#" -lt 2 ]; then
    echo "Usage: sh scripts/lavapipe.sh <revision> <command> [args...]" >&2
    exit 2
fi
cd "$(git rev-parse --show-toplevel)"
revision=$1
shift
# Hosted Ubuntu runners are x86_64, where Chromium's SwiftShader compiles shaders with Subzero; on arm64 it
# uses LLVM, which drew snow-trails' frames about 4.7 times as fast. QEMU's amd64 is too slow to model timing.
platform=linux/amd64
amd64() {
    [ "$(docker info --format '{{.Architecture}}')" = x86_64 ] \
        || docker run --rm --platform "$platform" ubuntu:24.04 grep -q rosetta /proc/self/maps
}
# A Mac's default Colima VM runs without Rosetta; a profile named hostedx86 runs amd64 under it beside that VM.
if ! amd64 && docker context inspect colima-hostedx86 >/dev/null 2>&1; then
    export DOCKER_CONTEXT=colima-hostedx86
fi
if ! amd64; then
    echo "scripts/lavapipe.sh needs native or Rosetta amd64 containers, for example: colima start -p hostedx86 --vm-type vz --vz-rosetta --cpu 4 --memory 16" >&2
    exit 1
fi
context=$(mktemp -d)
trap 'rm -rf "$context"' EXIT HUP INT TERM
cp scripts/lavapipe.Dockerfile "$context/Dockerfile"
cp .bun-version .node-version package.json bun.lock "$context/"
image=shallot-lavapipe-amd64:$(cat "$context/Dockerfile" "$context/.bun-version" "$context/.node-version" "$context/package.json" "$context/bun.lock" | shasum -a 256 | cut -c1-16)
if ! docker image inspect "$image" >/dev/null 2>&1; then
    docker build --platform "$platform" -t "$image" "$context"
fi
git archive "$revision" > "$context/source.tar"
docker run --rm -i --platform "$platform" --shm-size=1g \
    -v shallot-lavapipe-amd64-modules:/work/node_modules \
    -v shallot-lavapipe-amd64-playwright:/root/.cache/ms-playwright \
    "$image" sh -ec '
        tar -xf - -C /work
        git init -q
        lock=$(sha256sum bun.lock | cut -d " " -f 1)
        if [ "$(cat node_modules/.lavapipe-lock 2>/dev/null || true)" != "$lock" ]; then
            bun install --frozen-lockfile
            echo "$lock" > node_modules/.lavapipe-lock
        fi
        playwright=$(bun -e "console.log(require(\"playwright/package.json\").version)")
        if [ ! -f "$PLAYWRIGHT_BROWSERS_PATH/.lavapipe-$playwright" ]; then
            bunx playwright install chromium
            touch "$PLAYWRIGHT_BROWSERS_PATH/.lavapipe-$playwright"
        fi
        exec "$@"
    ' sh "$@" < "$context/source.tar"
