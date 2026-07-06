// Importing a scenario module runs its `register(...)` — adding a scenario is a new file here plus one
// line in this barrel. main.ts imports this for the side effects. The scenes carry the engine's GPU-driven
// coverage: `render` (the forward-pipeline atom — `mode`-selected: the structural cull/cluster/light/shadow
// metadata oracles + transport + prepass lanes, the shaded-look framebuffer probe rows lit / spec /
// spot / spotShadow / pointShadow / cascade / cascade-ortho / cascade-boundary / acne / zfight — the
// cascade trio covers the CSM receiver: perspective N-cascade, the ortho single-box fit, and no
// boundary bleed — and the asset-import rows gltf-model / gltf-animated / gltf-spill / gltf-multi
// (multi-asset palette + VAT accumulation) / gltf-worker (off-thread decode-pool byte-identity + codec
// wasm, plus the declarative by-name gate: preloader import + route sync with no import code),
// transparency, and the backdrop rows background / sky (the bindings-free + uniform-bound
// `Backgrounds` recipes filling un-rendered pixels)), `gltf` (the asset lifecycle atom — load → dispose/rebuild cache hit →
// unload/invalidate → reload, the editor play/stop + HMR substrate: the union must re-publish into the
// rebuild-wiped Compute maps with no re-decode, no re-upload), `sprite` (the 2D/billboard path — procedural canvas
// icons over the three billboard modes × clip/alpha, holed shadows, the ortho top-down camera; gated on
// the per-bucket indirect instance counts), `outline` (the screen-space JFA highlight — per-entity
// color/width, the reverse-Z occlusion gate, the fog post-color seam), `sat` (the validation-only GPU-SAT
// codegen gate the f64 oracle can't reach — the 14 C++ gold configs + the hull/rounded narrowphase matrix
// vs byte-exact readback), `accel` (the acceleration-structure pipeline: sort → build → traverse, gated on
// both the subgroup and LDS-fallback builder arms each run), and the
// three §6 physics scenarios by simulation type — `pile` (contact-settling
// rigidbodies), `constraints` (springs + joints), `character` (the kinematic controller), each gated
// against the f64 oracle. `kincarry` is an in-flight investigation scene (a rotating kinematic cube's
// corner vs a dynamic box) that measures the kinematic-angular-carry gap — spin imparts no tangential
// drag (COM-linear-only); `jointcarry` is its counterpart, validating the spindle game's actual mechanism —
// a dynamic ball pinned to a spinning kinematic capsule by a spherical JOINT orbits at the drive rate (a
// joint rides the rotating pose where a contact can't), gating the carry + the held radius. `motor` gates the
// angular motor constraint (a 1-DOF force-clamped angular drive, the spindle game's robust drive): a light
// spindle rigidly coupled to a heavy flywheel HOLDS its target ω under the load with the motor (`drive=motor`,
// the default + WGSL compile gate) where the forced-velocity drive STALLS (`drive=forced`, the contrast).
// `dyncarry` is the spindle game's drum assembly isolated — a thin driven box drum on
// a two-pin axle + a thin-spike crown + a capsule rope — gating stability, clean startup, and the rope
// construction (a spinning dynamic body DOES carry by friction, vs kincarry's null). `spikecatch` is an
// in-flight investigation scene (a rope draped over a vertically-bobbing thin rod) that contrasts
// dynamic-vs-kinematic carry: a DYNAMIC spike carries the rope, a KINEMATIC one loses it (the rigidbody
// kinematic-platform-carry gap, kex `backlog.md`); also gates valid-rope (non-overlapping links don't
// self-toss) + level-spike. `chain` is the phase-boundary microbench the physics waste audit reads
// (scripts/physics-bench.ts --audit), and `stress` the bottleneck-saturation atom (induce one resource
// axis, attribute the load via the per-pass profiler metric). The tier doctrine — targeted real-device
// tier run per-scenario, triple-duty atoms, in-flight dogfoods — is `CLAUDE.md` Examples + `testing.md`.
import "./accel";
import "./chain";
import "./character";
import "./constraints";
import "./dyncarry";
import "./gltf";
import "./jointcarry";
import "./kincarry";
import "./motor";
import "./outline";
import "./pile";
import "./render";
import "./sat";
import "./spikecatch";
import "./sprite";
import "./stress";
