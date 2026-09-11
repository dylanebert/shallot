// The solver gold: one declared check per reference scene. Each replays the scene through the public
// API and asserts the FNV-1a world-state hash equals the pinned Box3D C reference's at every step, so
// any drift in integration, collision, the solver or the joint blocks reds against the C oracle rather
// than against a blessed snapshot. Fixtures under fixtures/ are reference fixtures and are never edited
// to match; a mismatch names the scene and its first divergent step.

import { afterAll, beforeAll } from "bun:test";
import { check } from "../../../harness/check";
import { runScene, startKernel, stopKernel } from "./step.fixture";

beforeAll(async () => {
    await startKernel();
});

afterAll(async () => {
    await stopKernel();
});

check(
    "gold free-fall",
    {
        claim: "a body under gravity alone integrates bit-exactly like the C reference, so a changed gravity or velocity integration step reds.",
    },
    () => {
        runScene("free-fall", false, false);
    },
);

check(
    "gold sphere-drop",
    {
        claim: "a sphere dropped on a hull ground resolves its contact bit-exactly, so a changed sphere-hull manifold or contact solve reds.",
    },
    () => {
        runScene("sphere-drop", false, false);
    },
);

check(
    "gold box-stack",
    {
        claim: "a stack of boxes settles bit-exactly, so a changed hull-hull manifold, warm start or relax pass reds.",
    },
    () => {
        runScene("box-stack", false, false);
    },
);

check(
    "gold sphere-sleep",
    {
        claim: "a resting sphere falls asleep on the same step as the C reference, so a changed sleep threshold or island sleep bookkeeping reds.",
    },
    () => {
        runScene("sphere-sleep", true, false);
    },
);

check(
    "gold box-sleep",
    {
        claim: "a settled box stack falls asleep on the same step as the C reference, so a changed island sleep time accumulation reds.",
    },
    () => {
        runScene("box-sleep", true, false);
    },
);

check(
    "gold wake-drop",
    {
        claim: "a body dropped onto a sleeping island wakes it on the same step as the C reference, so a changed wake propagation reds.",
    },
    () => {
        runScene("wake-drop", true, false);
    },
);

check(
    "gold split-slide",
    {
        claim: "a sliding body splits its island bit-exactly, so a changed island split or constraint-graph removal reds.",
    },
    () => {
        runScene("split-slide", true, false);
    },
);

check(
    "gold revolute-dd",
    {
        claim: "a revolute joint between two dynamic bodies solves bit-exactly, so a changed revolute constraint block reds.",
    },
    () => {
        runScene("revolute-dd", false, false);
    },
);

check(
    "gold revolute-pendulum",
    {
        claim: "a revolute pendulum swings bit-exactly, so a changed joint frame or bias term reds.",
    },
    () => {
        runScene("revolute-pendulum", false, false);
    },
);

check(
    "gold revolute-motor",
    {
        claim: "a revolute motor drives its body bit-exactly, so a changed motor impulse clamp reds.",
    },
    () => {
        runScene("revolute-motor", false, false);
    },
);

check(
    "gold revolute-limit",
    {
        claim: "a revolute joint holds its angular limits bit-exactly, so a changed limit constraint or angle unwrap reds.",
    },
    () => {
        runScene("revolute-limit", false, false);
    },
);

check(
    "gold revolute-chain",
    {
        claim: "a chain of revolute joints sleeps and solves bit-exactly, so a changed joint island colouring reds.",
    },
    () => {
        runScene("revolute-chain", true, false);
    },
);

check(
    "gold weld-dd",
    {
        claim: "a weld joint between two dynamic bodies holds bit-exactly, so a changed weld linear or angular block reds.",
    },
    () => {
        runScene("weld-dd", false, false);
    },
);

check(
    "gold parallel",
    {
        claim: "a parallel joint keeps its axes aligned bit-exactly, so a changed parallel constraint basis reds.",
    },
    () => {
        runScene("parallel", false, false);
    },
);

check(
    "gold joint-contacts",
    {
        claim: "a jointed pair that also touches solves joints and contacts in the C reference's order, so a changed solve ordering reds.",
    },
    () => {
        runScene("joint-contacts", false, false);
    },
);

check(
    "gold motor",
    {
        claim: "a motor joint drives to its target bit-exactly, so a changed motor joint impulse or max-force clamp reds.",
    },
    () => {
        runScene("motor", false, false);
    },
);

check(
    "gold motor-spring",
    {
        claim: "a springy motor joint oscillates bit-exactly, so a changed soft-constraint softness derivation reds.",
    },
    () => {
        runScene("motor-spring", false, false);
    },
);

check(
    "gold distance",
    {
        claim: "a rigid distance joint holds its length bit-exactly, so a changed distance constraint block reds.",
    },
    () => {
        runScene("distance", false, false);
    },
);

check(
    "gold distance-spring",
    {
        claim: "a spring distance joint oscillates bit-exactly, so a changed hertz or damping-ratio softness reds.",
    },
    () => {
        runScene("distance-spring", false, false);
    },
);

check(
    "gold prismatic",
    {
        claim: "a prismatic joint slides along its axis bit-exactly, so a changed prismatic constraint basis reds.",
    },
    () => {
        runScene("prismatic", false, false);
    },
);

check(
    "gold prismatic-motor",
    {
        claim: "a prismatic motor drives along its axis bit-exactly, so a changed prismatic motor clamp reds.",
    },
    () => {
        runScene("prismatic-motor", false, false);
    },
);

check(
    "gold spherical",
    {
        claim: "a spherical joint holds its pivot bit-exactly, so a changed spherical point constraint reds.",
    },
    () => {
        runScene("spherical", false, false);
    },
);

check(
    "gold spherical-limits",
    {
        claim: "a spherical joint holds its cone and twist limits bit-exactly, so a changed swing-twist decomposition reds.",
    },
    () => {
        runScene("spherical-limits", false, false);
    },
);

check(
    "gold spherical-motor",
    {
        claim: "a spherical motor drives its orientation bit-exactly, so a changed spherical motor torque clamp reds.",
    },
    () => {
        runScene("spherical-motor", false, false);
    },
);

check(
    "gold wheel",
    {
        claim: "a wheel joint carries its suspension bit-exactly, so a changed wheel spring or lateral constraint reds.",
    },
    () => {
        runScene("wheel", false, false);
    },
);

check(
    "gold wheel-spin",
    { claim: "a spinning wheel joint solves bit-exactly, so a changed wheel motor axis reds." },
    () => {
        runScene("wheel-spin", false, false);
    },
);

check(
    "gold wheel-steer",
    { claim: "a steered wheel joint solves bit-exactly, so a changed wheel steering frame reds." },
    () => {
        runScene("wheel-steer", false, false);
    },
);

check(
    "gold ragdoll",
    {
        claim: "a fourteen-bone ragdoll island solves and sleeps bit-exactly, so a changed articulated joint ordering reds.",
    },
    () => {
        runScene("ragdoll", true, false);
    },
);

check(
    "gold ccd-drop",
    {
        claim: "a fast body dropped onto the ground is caught by continuous collision on the C reference's step, so a changed sweep or time-of-impact root find reds.",
    },
    () => {
        runScene("ccd-drop", false, true);
    },
);

check(
    "gold ccd-bullet",
    {
        claim: "a bullet body is swept against the ground bit-exactly, so a changed bullet classification or conservative advancement reds.",
    },
    () => {
        runScene("ccd-bullet", false, true);
    },
);

check(
    "gold mesh-box",
    {
        claim: "a box dropped on a static grid mesh resolves its triangle contacts bit-exactly, so a changed mesh-hull manifold reds.",
    },
    () => {
        runScene("mesh-box", false, false);
    },
);

check(
    "gold mesh-sphere",
    {
        claim: "a sphere dropped on a static grid mesh resolves bit-exactly, so a changed mesh-sphere manifold reds.",
    },
    () => {
        runScene("mesh-sphere", false, false);
    },
);

check(
    "gold mesh-capsule",
    {
        claim: "a capsule dropped on a static grid mesh resolves bit-exactly, so a changed mesh-capsule manifold reds.",
    },
    () => {
        runScene("mesh-capsule", false, false);
    },
);

check(
    "gold mesh-ccd",
    {
        claim: "a fast box swept onto a static mesh floor is caught bit-exactly, so a changed mesh sweep reds.",
    },
    () => {
        runScene("mesh-ccd", false, true);
    },
);

check(
    "gold height-box",
    {
        claim: "a box dropped on a static height field resolves bit-exactly, so a changed height-field cell lookup or triangulation reds.",
    },
    () => {
        runScene("height-box", false, false);
    },
);

check(
    "gold height-sphere",
    {
        claim: "a sphere dropped on a static height field resolves bit-exactly, so a changed height-field sphere manifold reds.",
    },
    () => {
        runScene("height-sphere", false, false);
    },
);

check(
    "gold height-capsule",
    {
        claim: "a capsule dropped on a static height field resolves bit-exactly, so a changed height-field capsule manifold reds.",
    },
    () => {
        runScene("height-capsule", false, false);
    },
);

check(
    "gold height-ccd",
    {
        claim: "a fast box swept onto a static height field is caught bit-exactly, so a changed height-field sweep reds.",
    },
    () => {
        runScene("height-ccd", false, true);
    },
);

check(
    "gold compound-hull",
    {
        claim: "a box dropped on a compound hull floor resolves every child bit-exactly, so a changed compound child transform reds.",
    },
    () => {
        runScene("compound-hull", false, false);
    },
);

check(
    "gold compound-capsule",
    {
        claim: "a box dropped on a compound capsule floor resolves bit-exactly, so a changed compound capsule child reds.",
    },
    () => {
        runScene("compound-capsule", false, false);
    },
);

check(
    "gold compound-sphere",
    {
        claim: "a box dropped on a compound sphere floor resolves bit-exactly, so a changed compound sphere child reds.",
    },
    () => {
        runScene("compound-sphere", false, false);
    },
);

check(
    "gold compound-mesh",
    {
        claim: "a box dropped on a compound mesh floor resolves bit-exactly, so a changed compound mesh child reds.",
    },
    () => {
        runScene("compound-mesh", false, false);
    },
);

check(
    "gold compound-ccd",
    {
        claim: "a fast box swept onto a two-hull compound floor is caught bit-exactly, so a changed compound sweep reds.",
    },
    () => {
        runScene("compound-ccd", false, true);
    },
);

check(
    "gold sensor",
    {
        claim: "a static sensor volume never perturbs the dynamics that pass through it, so a sensor leaking contact impulses reds.",
    },
    () => {
        runScene("sensor", false, true);
    },
);

check(
    "gold bench-pyramid",
    {
        claim: "a large single-island pyramid solves bit-exactly at scale, so a changed graph colouring or overflow-colour fallback reds.",
    },
    () => {
        runScene("bench-pyramid", false, false);
    },
);

check(
    "gold bench-many-pyramids",
    {
        claim: "many separate pyramid islands solve bit-exactly, so a changed island partition or solve order across islands reds.",
    },
    () => {
        runScene("bench-many-pyramids", false, false);
    },
);

check(
    "gold bench-joint-grid",
    { claim: "a large joint grid solves bit-exactly, so a changed joint colouring at scale reds." },
    () => {
        runScene("bench-joint-grid", false, false);
    },
);

check(
    "gold bench-washer",
    {
        claim: "kinematic contact churn against a washer solves bit-exactly, so a changed contact begin-touch ordering reds.",
    },
    () => {
        runScene("bench-washer", false, false);
    },
);

check(
    "gold bench-large-world",
    {
        claim: "spheres spawned into a large world over time solve bit-exactly, so a changed broad-phase move buffer or proxy rebuild reds.",
    },
    () => {
        runScene("bench-large-world", false, false);
    },
);

check(
    "gold bench-trees",
    {
        claim: "cylinder stacks on the reference's wavy mesh ground solve bit-exactly, so a changed cylinder hull or mesh contact at scale reds.",
    },
    () => {
        runScene("bench-trees", false, false);
    },
);

check(
    "gold bench-junkyard",
    {
        claim: "a kinematic pusher swept by setTargetTransform pushes a rock pile bit-exactly, so a changed kinematic target pose or rock hull reds.",
    },
    () => {
        runScene("bench-junkyard", false, false);
    },
);

check(
    "gold bench-rain",
    {
        claim: "ragdolls created mid-replay with their joints solve bit-exactly, so a changed mid-step body or joint creation path reds.",
    },
    () => {
        runScene("bench-rain", false, false);
    },
);

check(
    "gold drift",
    {
        claim: "a settled box stack holds for two thousand steps bit-exactly, so slow numerical drift away from the C reference reds.",
    },
    () => {
        runScene("drift", false, false);
    },
);
