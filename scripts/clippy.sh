#!/bin/sh
set -eu

base=${CLIPPY_BASE:-origin/main}
git rev-parse --verify "$base" >/dev/null

for manifest in crates/*/Cargo.toml; do
    crate=${manifest%/Cargo.toml}
    git diff --quiet "$base" -- "$crate" && continue
    package=shallot-${crate##*/}
    cargo clippy -p "$package" --all-targets -- -D warnings
    # Native declares shell binaries, no library, and OS-specific webview dependencies.
    # The other crates build WASM libraries.
    if [ "$crate" != crates/native ]; then
        cargo clippy -p "$package" --lib --target wasm32-unknown-unknown -- -D warnings
    fi
done
