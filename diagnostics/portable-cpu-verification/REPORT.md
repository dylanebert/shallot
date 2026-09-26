# Portable CPU verification qualification

Qualification ran once on the exact source candidate `11bb59d9827e5acbe630ffc43f186a486e69314f`, including the stage-2 workflow, CONTRIBUTING coverage wording and explicit empty Rust flags. The temporary qualification workflow was commit `44790c4aad495dba58b3e8eeaef2db10b9453962`; it checked out and verified the source SHA on both hosts. Run: [36257783461](https://github.com/dylanebert/shallot/actions/runs/36257783461). The workflow failed as expected; these results are not an all-green qualification.

## Hosts and runtimes

| Runner | Actual OS | Bun | Node | Rust / Cargo |
| --- | --- | --- | --- | --- |
| `ubuntu-latest` | Ubuntu 24.4.0 (noble), Linux x86_64, kernel `6.17.0-1022-azure` | 1.4.2 | v26.8.1 | rustc 1.98.1, cargo 1.98.1 |
| `macos-15` | macOS 15.7.9, Darwin 24.6.0, arm64 | 1.4.2 | v26.8.1 | rustc 1.98.1, cargo 1.98.1 |

`rustup` was 1.29.1 on Ubuntu and 1.29.0 on macOS. Full version output and OS records are in the host-record artifacts below.

## Results

- Static gates ran on both hosts. `tsc`, Biome, `check-commands`, `check-device-tiers`, `check-pack`, `check-pins`, `check-vite`, test-population, examples-index, scene-format, `cargo fmt` and `cargo clippy (physics)` passed. `check-imports` failed on both hosts for the known sibling import `src/core/rendering/view.ts` → `core/input` and all 12 transitional Destination rows (`audio`, `bvh`, `cells`, `character`, `glaze`, `gltf`, `mirror`, `part`, `physics`, `skin`, `slab`, `transforms`).
- Unit sweep ran on both hosts: 288 passed, 0 failed, 0 refused, 50 skipped.
- The full CPU integration population ran on both hosts with `--integration --all --requires '!gpu' --requires '!browser' --requires '!display' --no-unit-fallback`. Each host selected 43 rows: 42 passed, 1 failed, 0 refused, 0 unrun. The sole failure was the expected first-person steady-allocation subject. Ubuntu sampled 115,200 B at `point` and `rotate`; macOS sampled 115,200 B at `point`. GPU, browser and display populations were not run.
- Every requested invocation produced and uploaded its evidence. Static logs and unit/integration JUnit plus child output are available below. Artifact uploads succeeded on both hosts.

## Run artifacts

All links are artifacts for run 36257783461:

| Host | Invocation | Artifact |
| --- | --- | --- |
| Ubuntu | Static gates | [10911049156](https://github.com/dylanebert/shallot/actions/runs/36257783461/artifacts/10911049156) |
| Ubuntu | Unit sweep | [10910834573](https://github.com/dylanebert/shallot/actions/runs/36257783461/artifacts/10910834573) |
| Ubuntu | Full CPU integrations | [10910653289](https://github.com/dylanebert/shallot/actions/runs/36257783461/artifacts/10910653289) |
| Ubuntu | Host/source record | [10911292440](https://github.com/dylanebert/shallot/actions/runs/36257783461/artifacts/10911292440) |
| macOS | Static gates | [10911352292](https://github.com/dylanebert/shallot/actions/runs/36257783461/artifacts/10911352292) |
| macOS | Unit sweep | [10911610450](https://github.com/dylanebert/shallot/actions/runs/36257783461/artifacts/10911610450) |
| macOS | Full CPU integrations | [10910658249](https://github.com/dylanebert/shallot/actions/runs/36257783461/artifacts/10910658249) |
| macOS | Host/source record | [10911092822](https://github.com/dylanebert/shallot/actions/runs/36257783461/artifacts/10911092822) |
