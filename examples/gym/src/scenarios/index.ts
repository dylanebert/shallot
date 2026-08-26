// Importing a scenario module runs its `register(...)` — adding a scenario is a new file here plus one
// line in this barrel. main.ts imports this for the side effects. The scenes carry the engine's GPU-driven
// coverage: `render` (the forward-pipeline atom — `mode`-selected: the structural cull/cluster/light/shadow
// metadata oracles + transport + prepass lanes, the shaded-look framebuffer probe rows lit / spec /
// spot / spotShadow / pointShadow / cascade / cascade-ortho / cascade-boundary / acne / zfight, plus
// typed-variant (nonzero build-time specialization warm + post-build lazy draw compilation) — the
// cascade trio covers the CSM receiver: perspective N-cascade, the ortho single-box fit, and no
// boundary bleed — and the asset-import rows gltf-model / gltf-animated / gltf-spill / gltf-multi
// (multi-asset palette + VAT accumulation) / gltf-worker (off-thread decode-pool byte-identity + codec
// wasm, plus the declarative by-name gate: preloader import + route sync with no import code), skin-live
// (the live joint-palette substrate — a hand-built 2-bone rig posed through LiveSkin, deform + moving
// shadow + reach-bound probes, no glTF asset), ragdoll (the live palette's physics producer — RiggedFigure
// imported {live} + an 11-capsule tumble ragdoll on the Tumble.world escape hatch, a readBody → skinMatrix
// pose driver; upright→crumple deform + reach-survivor probes),
// transparency, and the backdrop rows background / sky (the bindings-free + uniform-bound
// `Backgrounds` recipes filling un-rendered pixels)), `gltf` (the asset lifecycle atom — load → dispose/rebuild cache hit →
// unload/invalidate → reload, the live-host play/stop + HMR substrate: the union must re-publish into the
// rebuild-wiped Compute maps with no re-decode, no re-upload), `sprite` (the 2D/billboard path — procedural canvas
// icons over the three billboard modes × clip/alpha, holed shadows, the ortho top-down camera; gated on
// the per-bucket indirect instance counts), `outline` (the screen-space JFA highlight — per-entity
// color/width, the reverse-Z occlusion gate, the fog post-color seam), `orbit-touch` (an Orbit camera
// targeting a box, headless-deterministic with no `assert` — the verdict lives in the driver-level touch
// gate `test/touch.playwright.ts`, which drives real CDP multi-touch and reads `window.__orbitPose()`),
// `sat` (the validation-only GPU-SAT codegen gate the f64 oracle can't reach — the 14 C++ gold configs + the hull/rounded narrowphase matrix
// vs byte-exact readback), `accel` (the acceleration-structure pipeline: sort → build → traverse, gated on
// both the subgroup and LDS-fallback builder arms each run), `text` (the typed SDF text producer's
// real-GPU draw — a local-fixture-font label, gated by the same chroma-against-clear framebuffer probe
// pattern `accel`'s line draw uses), and the
// three §6 physics scenarios by simulation type — `pile` (contact-settling
// rigidbodies), `constraints` (springs + joints), `character` (the kinematic controller), each gated
// against the f64 oracle. `backend` is the substrate swap gate: one
// scene authored purely against `standard/physics` (settle, no-fall-through, raycast, kinematic drive +
// firehose writeback) that runs unmodified under `--param backend=tumble|avbd` — behavioral parity, not
// bit-exact (two solvers can't hash-match a trajectory), plus a per-system CPU-span perf snapshot. Three
// scenarios gate the tumble `Tumble.world` escape-hatch surface past the substrate: `queries` (the spatial
// query trio — `castRayClosest` / `castShape` / `overlapAABB` over one deterministic obstacle scene),
// `rotation` (free angular dynamics in a zero-g world — the Dzhanibekov intermediate-axis flip + a parallel
// joint locking a panel's orientation), and `raining` (the streaming-spawn stress: bodies rain onto a pile
// and recycle at a cap, gating the live create/destroy marshal path under constant churn).
// `motor` gates the
// angular motor constraint (a 1-DOF force-clamped angular drive, a robust drive): a light
// spindle rigidly coupled to a heavy flywheel HOLDS its target ω under the load with the motor (`drive=motor`,
// the default + WGSL compile gate) where the forced-velocity drive STALLS (`drive=forced`, the contrast).
// `chain` is the phase-boundary microbench the physics waste audit reads
// (scripts/physics-bench.ts --audit), and `stress` the bottleneck-saturation atom (induce one resource
// axis, attribute the load via the per-pass profiler metric). `gpu-diagnostic` directly drives
// `validateGpu` (gpu.ts), `drainLog` (log.ts), and `probeBuffer`/`probeTexture` (probe.ts) — verified by
// import, not guessed. `mesh-fixture` is the release-prerequisite final-compositor hardening fixture: a
// custom-registered surface + a Part entity on the built-in `unlit` surface, both drawn through the real
// part/render/sear pipeline. `joints-paddle` is the tumble sample-host
// pilot: a tumble.js sample (`Paddle`) authored through the escape hatch
// (`tumble-sample.ts` + `tumble-paddle.ts`), verified bit-exact against its committed gold and rendered via
// the source-faithful debug-draw + mouse-grab layer every stage-4 sample twin reuses (the red-first oracle
// proof is `tumble-pilot.test.ts`). The stage-4 burn-down adds one gym twin per tumble.js sample the same
// way: `bodies-body-type` (the kinematic-platform sweep — the `update()` seam the pilot test proves),
// `stacking-arch` (a friction-only masonry showpiece, no knobs), `stacking-box-pyramid` (the canonical settling-pyramid
// resting-contact test), `stacking-dominoes` (a concentric-ring toppling chain reaction), `shapes-inclined-plane` (a
// friction ladder on a tilted plane, no knobs), `shapes-restitution` (a bounce-ladder row, sphere or box),
// `shapes-shape-soup` (a mixed sphere/capsule/box drop grid), `bodies-motion-locks` (hovering cubes under constant torque,
// angular locks per axis), `bodies-spinning-book` (three books in free fall — the Dzhanibekov intermediate-axis
// flip), `continuous-bullet-vs-stack` (a swept bullet punching a box stack), `continuous-thin-wall` (three fast projectiles
// CCD-swept against a thin static wall), `collision-overlap-box` (a circling query box outlining every shape it
// overlaps), `collision-ray-curtain` (a sweeping ray curtain over four spinning obstacles), `collision-shape-cast` (a swept-sphere cast
// resolved to its first contact over three rows of spinning obstacles), `compound-simple` (four tilted
// slabs baked into one static compound, a staircase spheres cascade down), `compound-spheres` (a boulder of
// eighteen spheres in one static compound, pelted by a rain of boxes), `compound-tile-floor` (a 10x10
// grid of tiles at random heights baked into one body — an uneven floor from a single compound), `events-hit`
// (fast contacts producing hit events, drawn as impact markers), `events-joint-break` (a row of hung boxes under a
// rising load, joints cut past a force threshold in `update()`), `events-sensor-sweep` (a kinematic sensor box
// sweeping vertically through a resting stack in `update()`), `geometry-convex-hull` (random point clouds reduced to
// convex polytopes via `createHull`), `geometry-convex-primitives` (a grid of the built-in cylinder/cone/rock hull
// generators), `geometry-hull-reduction` (sphere point clouds capped to N hull vertices), `joints-bridge` (a suspension
// bridge of paired-spherical-hinged planks sagging under dropped boxes), `joints-cantilever` (welded boxes
// jutting from a wall — the weld's angular spring), `joints-elevator` (a motor-driven prismatic platform
// reversing direction at its translation limits in `update()`), `joints-filter` (a filter joint muting one
// box-platform contact so it falls through, its twin resting on top), `joints-pendulum` (a chain of
// revolute-hinged boxes swinging out and settling), `joints-driving` (a wheel-jointed car — throttle bakes into
// the rear spin motors at build time, since the sample's live throttle/steer knobs only apply through
// `act()`, never called by the mint), `joints-parallel` (two hovering panels
// under constant torque — a parallel joint locks one level while the free panel tumbles, no knobs), `joints-rope` (a
// capsule chain on spherical joints, released from horizontal — it swings and coils in 3D), `joints-suspension` (a
// platform hung from four distance-joint springs — drop crates and it bobs), `mesh-terrain` (a sine-wave
// triangle-mesh ground — a grid of the `shape` knob's shape rolls down the hills and settles in the
// troughs), `mesh-torus` (a flat torus mesh floating hole-up — balls balance on its ring or drop straight
// through the hole, non-convex triangle-mesh collision), `ragdoll-ragdoll` (a capsule humanoid dropped from a
// height, folding at its cone/twist-limited joints as it settles), `determinism-falling-ragdolls` (a grid of ragdolls
// — the `grid` knob — piling up together, bit-exact deterministic by construction), `character-mover` (a
// self-driven kinematic capsule mover patrolling a walled arena on the plane solver — pogo ground-follow up
// a ramp and steps, shoving crates; the drive lives in `update()`), with more
// landing as the tumble-inline sample set is ported;
// the full list + the shared gold-match test live in `tumble-registry.ts` / `tumble-golds.test.ts`. The tier doctrine — targeted real-device tier run per-scenario, triple-duty
// atoms, in-flight dogfoods — is `CLAUDE.md` Examples + `testing.md`.
import "./accel";
import "./arch";
import "./backend";
import "./body-type";
import "./box-pyramid";
import "./bridge";
import "./bullet-vs-stack";
import "./cantilever";
import "./chain";
import "./character";
import "./character-mover";
import "./compound-simple";
import "./compound-spheres";
import "./compound-tile-floor";
import "./constraints";
import "./convex-hull";
import "./convex-primitives";
import "./dominoes";
import "./driving";
import "./elevator";
import "./falling-ragdolls";
import "./filter";
import "./gltf";
import "./gpu-diagnostic";
import "./hit";
import "./hull-reduction";
import "./inclined-plane";
import "./joint-break";
import "./mesh-fixture";
import "./motion-locks";
import "./motor";
import "./orbit-touch";
import "./outline";
import "./overlap-box";
import "./paddle";
import "./parallel";
import "./pendulum";
import "./pile";
import "./queries";
import "./ragdoll";
import "./raining";
import "./ray-curtain";
import "./render";
import "./restitution";
import "./rope";
import "./rotation";
import "./sat";
import "./sensor-sweep";
import "./shape-cast";
import "./shape-soup";
import "./spinning-book";
import "./sprite";
import "./stress";
import "./suspension";
import "./terrain";
import "./text";
import "./thin-wall";
import "./torus";
