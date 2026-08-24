// The scene roster for the tumble fixture gate — scene name → [enableSleep, enableContinuous],
// matching gen.c's per-scene world flags. Extracted from step.fixture.ts so the default-suite parity
// arm in step.test.ts can import SCENES without importing the fixture file itself (which would register
// the heavy bit-exact tests in the default `bun test` suite). The count of entries here is derived by
// that arm against the committed fixture files in tests/tumble/fixtures/, so a prose restatement of the
// number anywhere in the corpus is stale the moment a scene is added or removed.

export const SCENES: [string, boolean, boolean][] = [
    ["free-fall", false, false],
    ["sphere-drop", false, false],
    ["box-stack", false, false],
    ["sphere-sleep", true, false],
    ["box-sleep", true, false],
    ["wake-drop", true, false],
    ["split-slide", true, false],
    ["revolute-dd", false, false],
    ["revolute-pendulum", false, false],
    ["revolute-motor", false, false],
    ["revolute-limit", false, false],
    ["revolute-chain", true, false],
    ["weld-dd", false, false],
    ["parallel", false, false],
    ["joint-contacts", false, false],
    ["motor", false, false],
    ["motor-spring", false, false],
    ["distance", false, false],
    ["distance-spring", false, false],
    ["prismatic", false, false],
    ["prismatic-motor", false, false],
    ["spherical", false, false],
    ["spherical-limits", false, false],
    ["spherical-motor", false, false],
    ["wheel", false, false],
    ["wheel-spin", false, false],
    ["wheel-steer", false, false],
    ["ragdoll", true, false],
    // CCD: continuous on.
    ["ccd-drop", false, true],
    ["ccd-bullet", false, true],
    // mesh contacts: a box / sphere / capsule dropped onto a static grid-mesh floor.
    ["mesh-box", false, false],
    ["mesh-sphere", false, false],
    ["mesh-capsule", false, false],
    // mesh CCD: a fast box swept onto the static mesh floor (continuous on).
    ["mesh-ccd", false, true],
    // height fields: box / sphere / capsule dropped onto a static grid height field, then
    // a fast box for the height-field CCD path (continuous on).
    ["height-box", false, false],
    ["height-sphere", false, false],
    ["height-capsule", false, false],
    ["height-ccd", false, true],
    // compound contacts: a box dropped onto a static compound floor built from hull /
    // capsule / sphere / mesh children (sleep off).
    ["compound-hull", false, false],
    ["compound-capsule", false, false],
    ["compound-sphere", false, false],
    ["compound-mesh", false, false],
    // compound CCD: a fast box swept onto the static two-hull compound floor (continuous on).
    ["compound-ccd", false, true],
    // sensors: a static sensor volume that dynamic bodies fall / sweep through — the sensor
    // must not perturb dynamics, so the body hashes match the C oracle (continuous on for the fast body).
    ["sensor", false, true],
    // hardening: the benchmark scenes at reduced scale — the bit-exact contract at scale
    // (large islands, many islands, a big joint grid, kinematic contact churn, the move buffer), plus a
    // 2000-step drift gate. Sleep off; large-world spawns spheres via its stepFactory.
    ["bench-pyramid", false, false],
    ["bench-many-pyramids", false, false],
    ["bench-joint-grid", false, false],
    ["bench-washer", false, false],
    ["bench-large-world", false, false],
    // Trees: cylinder stacks on a wavy mesh ground whose libm-sinf vertices are loaded from the fixture
    // (the port cannot reproduce sinf bit-exactly); the dynamics on that fixed ground are bit-exact.
    ["bench-trees", false, false],
    // Junkyard: rock pile + a kinematic cylinder swept by setTargetTransform (bit-exact pusher pose).
    ["bench-junkyard", false, false],
    // Rain: ragdolls (14-bone articulated joint island) dropped onto a mesh ground, spawned over time.
    ["bench-rain", false, false],
    ["drift", false, false],
];
