Read `CONTRIBUTING.md`.

Pre-land check:

```sh
bun check && bun run lint &&
bun run test --path-ignore-patterns '**/*.gpu.test.ts' &&
bun test --todo ./scripts/check-*.node.ts &&
{ git diff --quiet origin/main -- crates || cargo fmt --check; } &&
(
    for manifest in crates/*/Cargo.toml; do
        crate=${manifest%/Cargo.toml}
        git diff --quiet origin/main -- "$crate" && continue
        package=shallot-${crate##*/}
        cargo clippy -p "$package" --all-targets -- -D warnings || exit 1
        if [ "$crate" != crates/native ]; then
            cargo clippy -p "$package" --lib --target wasm32-unknown-unknown -- -D warnings || exit 1
        fi
    done
)
```

`crates/native` is host-only: its manifest declares shell binaries, no library, and OS-specific webview dependencies. The other crates build WASM libraries.
