import { readFileSync } from "node:fs";
import { check } from "../../../harness/check";
import { runLegacyScenario } from "../solver/step.fixture";
import { loadScenarioCorpus, runScenario, type ScenarioOutput } from "./scenario";

function compareOutputs(
    generic: ScenarioOutput,
    legacy: ReturnType<typeof runLegacyScenario>,
): void {
    if (JSON.stringify(generic.hashes.map((item) => item.value)) !== JSON.stringify(legacy.hashes))
        throw new Error(`${generic.name}: world hash mismatch`);
    if (generic.observations.length !== legacy.observations.length)
        throw new Error(`${generic.name}: observation count mismatch`);
    for (let step = 0; step < legacy.observations.length; step++) {
        const got = generic.observations[step] as { bodies: Array<Record<string, unknown>> };
        const withoutLogicalIds = got.bodies.map(({ id: _id, ...body }) => body);
        if (JSON.stringify(withoutLogicalIds) !== JSON.stringify(legacy.observations[step].bodies))
            throw new Error(`${generic.name}: body observation mismatch at step ${step}`);
    }
}

const FOUNDATION_ROSTER = [
    "free-fall",
    "sphere-drop",
    "box-stack",
    "sphere-sleep",
    "box-sleep",
    "wake-drop",
    "split-slide",
];
const JOINT_ROSTER = [
    "revolute-dd",
    "revolute-pendulum",
    "revolute-motor",
    "revolute-limit",
    "revolute-chain",
    "weld-dd",
    "parallel",
    "joint-contacts",
    "motor",
    "motor-spring",
    "distance",
    "distance-spring",
    "prismatic",
    "prismatic-motor",
    "spherical",
    "spherical-limits",
    "spherical-motor",
    "wheel",
    "wheel-spin",
    "wheel-steer",
    "ragdoll",
];
const SURFACE_ROSTER = [
    "ccd-drop",
    "ccd-bullet",
    "mesh-box",
    "mesh-sphere",
    "mesh-capsule",
    "mesh-ccd",
    "height-box",
    "height-sphere",
    "height-capsule",
    "height-ccd",
];
const COMPOUND_SENSOR_ROSTER = [
    "compound-hull",
    "compound-capsule",
    "compound-sphere",
    "compound-mesh",
    "compound-ccd",
    "sensor",
];
const BENCHMARK_ROSTER = [
    "bench-pyramid",
    "bench-many-pyramids",
    "bench-joint-grid",
    "bench-washer",
    "bench-large-world",
    "bench-trees",
    "bench-junkyard",
    "bench-rain",
    "drift",
];
function compareFamily(roster: string[]): void {
    const { corpus, digest } = loadScenarioCorpus();
    const cumulative = [
        ...FOUNDATION_ROSTER,
        ...JOINT_ROSTER,
        ...SURFACE_ROSTER,
        ...COMPOUND_SENSOR_ROSTER,
        ...BENCHMARK_ROSTER,
    ];
    if (
        JSON.stringify(corpus.scenarios.map((scenario) => scenario.name)) !==
            JSON.stringify(cumulative) ||
        JSON.stringify(corpus.scenarios.map((scenario) => scenario.id)) !==
            JSON.stringify(cumulative.map((name) => `s1.${name}.v1`))
    )
        throw new Error("scenario corpus roster and IDs are not the exact cumulative O5d order");
    const selected = corpus.scenarios.filter((scenario) => roster.includes(scenario.name));
    if (JSON.stringify(selected.map((scenario) => scenario.name)) !== JSON.stringify(roster))
        throw new Error("scenario family roster is not exact");
    for (const scenario of selected) {
        const world = scenario.commands.find((command) => command.op === "world.create");
        compareOutputs(
            runScenario(scenario, digest),
            runLegacyScenario(
                scenario.name,
                world?.enableSleep === true,
                world?.enableContinuous === true,
                scenario.stepCount,
            ),
        );
    }
}
check(
    "O5a Shallot command interpreter migration",
    { claim: "box3d-scenario-migration-foundation", size: "integration" },
    () => compareFamily(FOUNDATION_ROSTER),
);
check(
    "O5b Shallot command interpreter migration",
    { claim: "box3d-scenario-migration-joints", size: "integration" },
    () => compareFamily(JOINT_ROSTER),
);
check(
    "O5c Shallot command interpreter migration",
    { claim: "box3d-scenario-migration-surfaces", size: "integration" },
    () => compareFamily(SURFACE_ROSTER),
);
check(
    "O5d Shallot command interpreter migration",
    { claim: "box3d-scenario-migration-compound-sensor", size: "integration" },
    () => compareFamily(COMPOUND_SENSOR_ROSTER),
);
check(
    "O5e Shallot benchmark command interpreter migration",
    { claim: "box3d-scenario-migration-benchmarks", size: "integration" },
    () => compareFamily(BENCHMARK_ROSTER),
);
check(
    "O5e Shallot benchmark timing mutations",
    { claim: "box3d-scenario-migration-benchmark-timing", size: "integration" },
    () => {
        const { corpus, digest } = loadScenarioCorpus();
        const moveCommands = (
            name: string,
            predicate: (command: (typeof corpus.scenarios)[number]["commands"][number]) => boolean,
            count: number,
            afterStep: string,
        ): void => {
            const scenario = structuredClone(corpus.scenarios.find((item) => item.name === name));
            if (!scenario) throw new Error(`${name} timing target is missing`);
            const index = scenario.commands.findIndex(predicate);
            const stepIndex = scenario.commands.findIndex(
                (command) => command.id === afterStep && command.op === "step",
            );
            if (
                index < 0 ||
                stepIndex < 0 ||
                index > stepIndex ||
                index + count > scenario.commands.length
            )
                throw new Error(`${name} timing command is not before its scheduled step`);
            const moved = scenario.commands.splice(index, count);
            const newStepIndex = scenario.commands.findIndex(
                (command) => command.id === afterStep && command.op === "step",
            );
            scenario.commands.splice(newStepIndex + 1, 0, ...moved);
            const baseline = runScenario(
                corpus.scenarios.find((item) => item.name === name)!,
                digest,
            );
            const changed = runScenario(scenario, digest);
            if (
                JSON.stringify(baseline.observations) === JSON.stringify(changed.observations) ||
                JSON.stringify(baseline.hashes) === JSON.stringify(changed.hashes)
            )
                throw new Error(`${name} timing mutation did not change the replay`);
        };
        moveCommands(
            "bench-large-world",
            (command) => command.op === "body.spawn" && command.id === "b144",
            2,
            "step-005",
        );
        moveCommands(
            "bench-junkyard",
            (command) => command.op === "body.target-transform" && command.id === "target-000",
            1,
            "step-000",
        );
    },
);

check(
    "O5a command receipt and adversarial gates",
    { claim: "box3d-scenario-command-corpus", size: "integration" },
    () => {
        const { corpus, digest } = loadScenarioCorpus();
        const freeFall = structuredClone(corpus.scenarios[0]);
        const baseline = runScenario(freeFall, digest);
        if (
            JSON.stringify(baseline.receipt.consumedCommands) !==
            JSON.stringify(freeFall.commands.map((command) => command.id))
        )
            throw new Error("receipt omitted a consumed command");
        if (baseline.receipt.observationIds.length !== freeFall.stepCount)
            throw new Error("receipt omitted an observation ID");
        const mutation = structuredClone(freeFall);
        const body = mutation.commands.find((command) => command.op === "body.create");
        if (!body) throw new Error("free-fall body command is missing");
        body.angularVelocity = ["0x40000000", "0x40a00000", "0x40000000"];
        const mutated = runScenario(mutation, digest, true);
        if (
            JSON.stringify(mutated.observations) === JSON.stringify(baseline.observations) ||
            JSON.stringify(mutated.hashes) === JSON.stringify(baseline.hashes)
        )
            throw new Error("angular-velocity mutation did not reach the TypeScript adapter");
        const revoluteMotor = structuredClone(
            corpus.scenarios.find((scenario) => scenario.name === "revolute-motor"),
        );
        if (!revoluteMotor) throw new Error("revolute-motor mutation target is missing");
        const motorCommand = revoluteMotor.commands.find(
            (command) => command.op === "joint.revolute",
        );
        if (!motorCommand) throw new Error("revolute-motor joint command is missing");
        const motorBaseline = runScenario(revoluteMotor, digest);
        motorCommand.motorSpeed = "0xc0400000";
        const motorChanged = runScenario(revoluteMotor, digest);
        if (
            JSON.stringify(motorBaseline.observations) ===
                JSON.stringify(motorChanged.observations) ||
            JSON.stringify(motorBaseline.hashes) === JSON.stringify(motorChanged.hashes)
        )
            throw new Error(
                "revolute-motor motor-speed mutation did not reach the TypeScript adapter",
            );
        const compoundHull = structuredClone(
            corpus.scenarios.find((scenario) => scenario.name === "compound-hull"),
        );
        if (!compoundHull) throw new Error("compound-hull mutation target is missing");
        const compoundBaseline = runScenario(compoundHull, digest);
        const firstChildCommand = compoundHull.commands.find(
            (command) => command.op === "resource.compound",
        );
        if (
            !firstChildCommand ||
            !Array.isArray(firstChildCommand.hulls) ||
            firstChildCommand.hulls.length === 0
        )
            throw new Error("compound-hull first child is missing");
        const firstTransform = (firstChildCommand.hulls[0] as Record<string, unknown>)
            .transform as Record<string, unknown>;
        (firstTransform.p as string[])[0] = "0x3f800000";
        const compoundChanged = runScenario(compoundHull, digest);
        if (
            JSON.stringify(compoundBaseline.observations) ===
                JSON.stringify(compoundChanged.observations) ||
            JSON.stringify(compoundBaseline.hashes) === JSON.stringify(compoundChanged.hashes)
        )
            throw new Error(
                "compound-hull first-child transform mutation did not reach the TypeScript adapter",
            );
        const ccdBullet = structuredClone(
            corpus.scenarios.find((scenario) => scenario.name === "ccd-bullet"),
        );
        if (!ccdBullet) throw new Error("ccd-bullet mutation target is missing");
        const bulletBaseline = runScenario(ccdBullet, digest);
        const bulletBody = ccdBullet.commands.find(
            (command) => command.op === "body.create" && command.id === "b2",
        );
        if (!bulletBody) throw new Error("ccd-bullet body command is missing");
        bulletBody.linearVelocity = ["0x42c80000", "0x00000000", "0x00000000"];
        const bulletChanged = runScenario(ccdBullet, digest);
        if (
            JSON.stringify(bulletBaseline.observations) ===
                JSON.stringify(bulletChanged.observations) ||
            JSON.stringify(bulletBaseline.hashes) === JSON.stringify(bulletChanged.hashes)
        )
            throw new Error("ccd-bullet x-velocity mutation did not reach the TypeScript adapter");
        const deletedJoint = structuredClone(revoluteMotor);
        deletedJoint.commands = deletedJoint.commands.filter(
            (command) => command.op !== "joint.revolute",
        );
        try {
            runScenario(deletedJoint, digest);
            throw new Error("deleting a joint command unexpectedly succeeded");
        } catch (error) {
            if (!(error instanceof Error) || !error.message.includes("missing joint commands"))
                throw error;
        }
        const missingGeometry = structuredClone(
            corpus.scenarios.find((scenario) => scenario.name === "mesh-box"),
        );
        if (!missingGeometry) throw new Error("mesh-box geometry target is missing");
        missingGeometry.commands = missingGeometry.commands.filter(
            (command) => command.op !== "resource.mesh",
        );
        try {
            runScenario(missingGeometry, digest);
            throw new Error("missing geometry unexpectedly succeeded");
        } catch (error) {
            if (!(error instanceof Error) || !error.message.includes("unknown mesh")) throw error;
        }
        const missingReference = structuredClone(
            corpus.scenarios.find((scenario) => scenario.name === "height-sphere"),
        );
        if (!missingReference) throw new Error("height-sphere reference target is missing");
        missingReference.commands = missingReference.commands.filter(
            (command) => command.id !== "b1",
        );
        try {
            runScenario(missingReference, digest);
            throw new Error("missing body reference unexpectedly succeeded");
        } catch (error) {
            if (!(error instanceof Error) || !error.message.includes("unknown body")) throw error;
        }
        const malformedCompound = structuredClone(compoundHull);
        const malformedResource = malformedCompound.commands.find(
            (command) => command.op === "resource.compound",
        );
        if (!malformedResource || !Array.isArray(malformedResource.hulls))
            throw new Error("compound-hull malformed target is missing");
        malformedResource.hulls = malformedResource.hulls.slice(1);
        try {
            runScenario(malformedCompound, digest);
            throw new Error("missing compound child unexpectedly succeeded");
        } catch (error) {
            if (!(error instanceof Error) || !error.message.includes("child order")) throw error;
        }
        const reorderedCompound = structuredClone(compoundHull);
        const reorderedResource = reorderedCompound.commands.find(
            (command) => command.op === "resource.compound",
        );
        if (!reorderedResource || !Array.isArray(reorderedResource.hulls))
            throw new Error("compound-hull reorder target is missing");
        reorderedResource.hulls.reverse();
        try {
            runScenario(reorderedCompound, digest);
            throw new Error("reordered compound child unexpectedly succeeded");
        } catch (error) {
            if (!(error instanceof Error) || !error.message.includes("child order")) throw error;
        }
        const unresolvedCompound = structuredClone(compoundHull);
        const unresolvedResource = unresolvedCompound.commands.find(
            (command) => command.op === "resource.compound",
        );
        if (
            !unresolvedResource ||
            !Array.isArray(unresolvedResource.hulls) ||
            !unresolvedResource.hulls[0]
        )
            throw new Error("compound-hull unresolved target is missing");
        unresolvedResource.hulls[0].resource = "r999";
        try {
            runScenario(unresolvedCompound, digest);
            throw new Error("unresolved compound resource unexpectedly succeeded");
        } catch (error) {
            if (!(error instanceof Error) || !error.message.includes("unknown hull")) throw error;
        }
        const omittedSensorEvent = structuredClone(
            corpus.scenarios.find((scenario) => scenario.name === "sensor"),
        );
        if (!omittedSensorEvent) throw new Error("sensor event target is missing");
        omittedSensorEvent.commands = omittedSensorEvent.commands.filter(
            (command) => command.id !== "events-000",
        );
        try {
            runScenario(omittedSensorEvent, digest);
            throw new Error("omitted sensor event unexpectedly succeeded");
        } catch (error) {
            if (!(error instanceof Error) || !error.message.includes("unconsumed schedule"))
                throw error;
        }
        const unknown = structuredClone(freeFall);
        unknown.commands[0].op = "scenario-name-dispatch";
        try {
            runScenario(unknown, digest);
            throw new Error("unknown command unexpectedly succeeded");
        } catch (error) {
            if (!(error instanceof Error) || !error.message.includes("unknown command op"))
                throw error;
        }
        const unconsumed = structuredClone(freeFall);
        unconsumed.commands = unconsumed.commands.filter(
            (command) => command.op !== "hash" || command.step !== 0,
        );
        try {
            runScenario(unconsumed, digest);
            throw new Error("unconsumed command unexpectedly succeeded");
        } catch (error) {
            if (!(error instanceof Error) || !error.message.includes("unconsumed schedule"))
                throw error;
        }
        const source = readFileSync(new URL("./scenario.ts", import.meta.url), "utf8");
        if (/legacyBuilder|setupKind|builders\s*\[|scenario\.(name|id)\s*===/.test(source))
            throw new Error("generic interpreter contains forbidden name dispatch");
    },
);
