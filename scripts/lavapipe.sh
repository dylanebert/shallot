#!/bin/sh
set -eu

if [ "$#" -lt 2 ]; then
    echo "Usage: sh scripts/lavapipe.sh <revision> <command> [args...]" >&2
    exit 2
fi
cd "$(git rev-parse --show-toplevel)"
revision=$1
shift
context=$(mktemp -d)
trap 'rm -rf "$context"' EXIT HUP INT TERM
cp scripts/lavapipe.Dockerfile "$context/Dockerfile"
cp .bun-version .node-version package.json bun.lock "$context/"
image=shallot-lavapipe:$(cat "$context/Dockerfile" "$context/.bun-version" "$context/.node-version" "$context/package.json" "$context/bun.lock" | shasum -a 256 | cut -c1-16)
if ! docker image inspect "$image" >/dev/null 2>&1; then
    docker build -t "$image" "$context"
fi
git archive "$revision" > "$context/source.tar"
docker run --rm -i --shm-size=1g \
    -v shallot-lavapipe-modules:/work/node_modules \
    -v shallot-lavapipe-playwright:/root/.cache/ms-playwright \
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
