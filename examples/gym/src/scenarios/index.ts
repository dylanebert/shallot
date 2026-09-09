// Importing a scenario module runs its `register(...)` — adding a scenario is a new file here plus one
// line in this barrel. main.ts imports this for the side effects. What each scenario covers, and why, is
// the per-scenario `covers` table in `timeouts.ts`; the tier doctrine is `examples.md` + `testing.md`.
//
// `render` is one registered scenario carrying many `mode`-selected rows — the structural
// cull/cluster/light/shadow oracles, the shaded-look framebuffer probes, typed variants, the glTF import
// rows, skin-live, transparency and the background/sky backdrops — which is why its coverage folds into
// one table key rather than one key per row. `sat` is the validation-only GPU-SAT codegen gate the f64
// oracle cannot reach. `accel` gates sort → build → traverse on both the subgroup and LDS-fallback
// builder arms each run. `text` and `cells` gate their producers' real-GPU dispatch, `cells` with
// `noRender: true` like `gpu-diagnostic`. `pile`, `constraints` and `character` are the three physics
// scenarios by simulation type, each gated against the f64 oracle; `backend` is the substrate swap gate,
// behavioural parity rather than bit-exact, since two solvers cannot hash-match a trajectory. `queries`,
// `rotation` and `raining` gate the `Tumble.world` escape hatch past the substrate. `motor` gates the
// angular motor constraint: the motor drive HOLDS its target ω under load where the forced-velocity
// drive stalls. `chain` is the phase-boundary microbench the physics waste audit reads, `stress` the
// bottleneck-saturation atom, `mesh-fixture` the final-compositor hardening fixture, and `orbit-touch`
// the driver fixture whose verdict lives entirely in `test/touch.playwright.ts`.
import "./accel";
import "./backend";
import "./cells";
import "./chain";
import "./character";
import "./constraints";
import "./gltf";
import "./gpu-diagnostic";
import "./mesh-fixture";
import "./motor";
import "./orbit-touch";
import "./outline";
import "./pile";
import "./queries";
import "./raining";
import "./render";
import "./rotation";
import "./sat";
import "./sprite";
import "./stress";
import "./text";
