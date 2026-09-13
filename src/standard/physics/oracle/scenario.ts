import { createHash } from "node:crypto";
import { readFileSync } from "node:fs";
import { BodyType, type Body } from "../api/index";
import { createHeightField, createMesh, defaultFilter, defaultSurfaceMaterial, makeBoxHull, type MeshData } from "../api/index";
import { hashWorldState } from "../world/hash";
import { World } from "../api/world";

type Command = {
    op: string;
    id: string;
    [key: string]: unknown;
};
type Scenario = { id: string; name: string; commands: Command[]; stepCount: number; requiredJointIds?: string[] };
type Corpus = { schema: string; corpusVersion: number; scenarios: Scenario[] };
export type ScenarioOutput = {
    schema: "box3d-oracle/scenario-output/v1";
    id: string;
    name: string;
    corpusDigest: string;
    observations: unknown[];
    hashes: { step: number; value: string; receiptId: string }[];
    receipt: { corpusDigest: string; consumedCommands: string[]; observationIds: string[] };
};

const f32 = (bits: string): number => {
    if (!/^0x[0-9a-f]{8}$/.test(bits)) throw new Error(`invalid f32 bits: ${bits}`);
    const word = Number.parseInt(bits.slice(2), 16);
    const view = new DataView(new ArrayBuffer(4));
    view.setUint32(0, word, true);
    return view.getFloat32(0, true);
};
const bits = (value: number): string => {
    const view = new DataView(new ArrayBuffer(4));
    view.setFloat32(0, value, true);
    return `0x${view.getUint32(0, true).toString(16).padStart(8, "0")}`;
};
const quat = (values: unknown[]): { v: { x: number; y: number; z: number }; s: number } => { if (!Array.isArray(values) || values.length !== 4) throw new Error("expected a four-component f32 quaternion"); return { v: vec3(values.slice(0, 3)), s: f32(String(values[3])) }; };
const frame = (value: unknown): { p: { x: number; y: number; z: number }; q: { v: { x: number; y: number; z: number }; s: number } } => { if (!value || typeof value !== "object") throw new Error("expected a joint frame"); const record = value as Record<string, unknown>; return { p: vec3(record.p as unknown[]), q: quat(record.q as unknown[]) }; };
const vec3 = (values: unknown[]): { x: number; y: number; z: number } => {
    if (!Array.isArray(values) || values.length !== 3) throw new Error("expected a three-component f32 vector");
    return { x: f32(String(values[0])), y: f32(String(values[1])), z: f32(String(values[2])) };
};
const corpusDigest = (bytes: Uint8Array): string => createHash("sha256").update(bytes).digest("hex");

export function loadScenarioCorpus(path = process.env.BOX3D_SCENARIO_CORPUS): { corpus: Corpus; digest: string } {
    if (!path) throw new Error("BOX3D_SCENARIO_CORPUS is required; the TypeScript interpreter has no embedded corpus");
    const bytes = readFileSync(path);
    const corpus = JSON.parse(bytes.toString()) as Corpus;
    if (corpus.schema !== "box3d-oracle/scenario-command/v1" || corpus.corpusVersion !== 1 || !Array.isArray(corpus.scenarios)) throw new Error("invalid scenario command corpus");
    return { corpus, digest: corpusDigest(bytes) };
}

function bodyType(value: unknown): BodyType {
    if (value === "static") return BodyType.Static;
    if (value === "kinematic") return BodyType.Kinematic;
    if (value === "dynamic") return BodyType.Dynamic;
    throw new Error(`unknown body type: ${String(value)}`);
}

export function runScenario(scenario: Scenario, digest: string, mutateAngularVelocity = false): ScenarioOutput {
    const bodies = new Map<string, Body>();
    const joints = new Set<string>();
    const boxes = new Map<string, ReturnType<typeof makeBoxHull>>();
    const spheres = new Map<string, { center: { x: number; y: number; z: number }; radius: number }>();
    const capsules = new Map<string, { center1: { x: number; y: number; z: number }; center2: { x: number; y: number; z: number }; radius: number }>();
    const meshes = new Map<string, MeshData>();
    const heightFields = new Map<string, ReturnType<typeof createHeightField>>();
    const consumed: string[] = [];
    const observationIds: string[] = [];
    const observations: unknown[] = [];
    const hashes: { step: number; value: string; receiptId: string }[] = [];
    let world: World | undefined;
    let observed = false;
    for (const command of scenario.commands) {
        if (consumed.includes(command.id)) throw new Error(`duplicate consumed command ${command.id}`);
        consumed.push(command.id);
        switch (command.op) {
            case "world.create": {
                if (world !== undefined) throw new Error("duplicate world.create");
                const gravity = vec3(command.gravity as unknown[]);
                world = new World({ gravity, enableSleep: command.enableSleep === true, enableContinuous: command.enableContinuous === true });
                break;
            }
            case "body.create": {
                if (!world) throw new Error("body.create before world.create");
                const position = vec3(command.position as unknown[]);
                const linearVelocity = vec3(command.linearVelocity as unknown[]);
                const angularVelocity = vec3(command.angularVelocity as unknown[]);
                if (mutateAngularVelocity && bodies.size === 0) angularVelocity.z = Math.fround(angularVelocity.z + 1);
                const body = world.createBody({ type: bodyType(command.type), isBullet: command.isBullet === true, position, rotation: quat((command.rotation ?? ["0x00000000", "0x00000000", "0x00000000", "0x3f800000"]) as unknown[]), linearVelocity, angularVelocity, ...(command.linearDamping === undefined ? {} : { linearDamping: f32(String(command.linearDamping)) }), ...(command.angularDamping === undefined ? {} : { angularDamping: f32(String(command.angularDamping)) }) });
                bodies.set(command.id, body);
                break;
            }
            case "resource.box": {
                const halfExtents = vec3(command.halfExtents as unknown[]);
                boxes.set(command.id, makeBoxHull(halfExtents.x, halfExtents.y, halfExtents.z));
                break;
            }
            case "resource.sphere":
                spheres.set(command.id, { center: { x: 0, y: 0, z: 0 }, radius: f32(String(command.radius)) });
                break;
            case "resource.capsule":
                capsules.set(command.id, { center1: vec3(command.center1 as unknown[]), center2: vec3(command.center2 as unknown[]), radius: f32(String(command.radius)) });
                break;
            case "resource.mesh": {
                const vertices = (command.vertices as unknown[]).map((value) => f32(String(value)));
                const points = []; for (let i = 0; i < vertices.length; i += 3) points.push({ x: vertices[i], y: vertices[i + 1], z: vertices[i + 2] });
                const mesh = createMesh({ vertices: points, indices: command.indices as number[], useMedianSplit: command.useMedianSplit === true, identifyEdges: command.identifyEdges === true });
                if (!mesh) throw new Error(`mesh resource ${command.id} could not be built`);
                meshes.set(command.id, mesh);
                break;
            }
            case "resource.height-field": {
                const samples = (command.samples as unknown[]).map((value) => f32(String(value)));
                const field = createHeightField({ heights: samples, materialIndices: (command.materialIndices as number[]) ?? null, scale: { x: f32(String(command.scaleX)), y: f32(String(command.scaleY)), z: f32(String(command.scaleZ)) }, countX: Number(command.countX), countZ: Number(command.countZ), globalMinimumHeight: f32(String(command.globalMinimumHeight)), globalMaximumHeight: f32(String(command.globalMaximumHeight)), clockwiseWinding: command.clockwiseWinding === true });
                heightFields.set(command.id, field);
                break;
            }
            case "shape.create": {
                if (!world) throw new Error("shape.create before world.create");
                const body = bodies.get(String(command.body));
                if (!body) throw new Error(`shape references unknown body ${String(command.body)}`);
                if (command.kind === "box") {
                    const box = boxes.get(String(command.resource));
                    if (!box) throw new Error(`shape references unknown box ${String(command.resource)}`);
                    body.createHull({ baseMaterial: { ...defaultSurfaceMaterial(), rollingResistance: f32(String(command.rollingResistance ?? "0x00000000")) }, filter: { ...defaultFilter(), groupIndex: Number(command.groupIndex ?? 0) } }, box);
                } else if (command.kind === "sphere") {
                    const sphere = spheres.get(String(command.resource));
                    if (!sphere) throw new Error(`shape references unknown sphere ${String(command.resource)}`);
                    body.createSphere({ baseMaterial: { ...defaultSurfaceMaterial(), rollingResistance: f32(String(command.rollingResistance ?? "0x00000000")) }, filter: { ...defaultFilter(), groupIndex: Number(command.groupIndex ?? 0) } }, sphere);
                } else if (command.kind === "capsule") {
                    const capsule = capsules.get(String(command.resource));
                    if (!capsule) throw new Error(`shape references unknown capsule ${String(command.resource)}`);
                    body.createCapsule({ baseMaterial: { ...defaultSurfaceMaterial(), rollingResistance: f32(String(command.rollingResistance ?? "0x00000000")) }, filter: { ...defaultFilter(), groupIndex: Number(command.groupIndex ?? 0) } }, capsule);
                } else if (command.kind === "mesh") {
                    const mesh = meshes.get(String(command.resource)); if (!mesh) throw new Error(`shape references unknown mesh ${String(command.resource)}`);
                    body.createMesh({ baseMaterial: { ...defaultSurfaceMaterial(), rollingResistance: f32(String(command.rollingResistance ?? "0x00000000")) }, filter: { ...defaultFilter(), groupIndex: Number(command.groupIndex ?? 0) } }, mesh, vec3((command.scale ?? ["0x3f800000", "0x3f800000", "0x3f800000"]) as unknown[]));
                } else if (command.kind === "height-field") {
                    const field = heightFields.get(String(command.resource)); if (!field) throw new Error(`shape references unknown height field ${String(command.resource)}`);
                    body.createHeightField({ baseMaterial: { ...defaultSurfaceMaterial(), rollingResistance: f32(String(command.rollingResistance ?? "0x00000000")) }, filter: { ...defaultFilter(), groupIndex: Number(command.groupIndex ?? 0) } }, field);
                } else throw new Error(`unknown shape kind ${String(command.kind)}`);
                break;
            }
            case "joint.revolute":
            case "joint.weld":
            case "joint.parallel":
            case "joint.motor":
            case "joint.distance":
            case "joint.prismatic":
            case "joint.spherical":
            case "joint.wheel": {
                if (!world) throw new Error(`joint ${command.id} before world.create`);
                const bodyA = bodies.get(String(command.bodyA)); const bodyB = bodies.get(String(command.bodyB));
                if (!bodyA || !bodyB) throw new Error(`joint ${command.id} references an unknown body`);
                const config: Record<string, unknown> = { localFrameA: frame(command.localFrameA), localFrameB: frame(command.localFrameB) };
                for (const [key, value] of Object.entries(command)) {
                    if (["op", "id", "bodyA", "bodyB", "localFrameA", "localFrameB"].includes(key)) continue;
                    if (typeof value === "boolean") config[key] = value;
                    else if (Array.isArray(value)) config[key] = vec3(value);
                    else config[key] = f32(String(value));
                }
                if (command.op === "joint.revolute") world.createRevoluteJoint(bodyA, bodyB, config as never);
                else if (command.op === "joint.weld") world.createWeldJoint(bodyA, bodyB, config as never);
                else if (command.op === "joint.parallel") world.createParallelJoint(bodyA, bodyB, config as never);
                else if (command.op === "joint.motor") world.createMotorJoint(bodyA, bodyB, config as never);
                else if (command.op === "joint.distance") world.createDistanceJoint(bodyA, bodyB, config as never);
                else if (command.op === "joint.prismatic") world.createPrismaticJoint(bodyA, bodyB, config as never);
                else if (command.op === "joint.spherical") world.createSphericalJoint(bodyA, bodyB, config as never);
                else world.createWheelJoint(bodyA, bodyB, config as never);
                joints.add(command.id);
                break;
            }
            case "step":
                if (!world) throw new Error("step before world.create");
                world.step(f32(String(command.timeStep)), Number(command.subStepCount));
                break;
            case "observe": {
                if (!world) throw new Error("observe before world.create");
                const bodyIds = command.bodies;
                if (!Array.isArray(bodyIds)) throw new Error(`observation ${command.id} has no body list`);
                const result = bodyIds.map((id) => {
                    const body = bodies.get(String(id));
                    if (!body) throw new Error(`observation references unknown body ${String(id)}`);
                    const transform = body.getTransform();
                    const linear = body.getLinearVelocity();
                    const angular = body.getAngularVelocity();
                    return { id: String(id), p: [bits(transform.p.x), bits(transform.p.y), bits(transform.p.z)], q: [bits(transform.q.v.x), bits(transform.q.v.y), bits(transform.q.v.z), bits(transform.q.s)], v: [bits(linear.x), bits(linear.y), bits(linear.z)], w: [bits(angular.x), bits(angular.y), bits(angular.z)] };
                });
                observations.push({ step: Number(command.step), bodies: result, receiptId: command.id });
                observationIds.push(command.id);
                observed = true;
                break;
            }
            case "hash":
                if (!world || !observed) throw new Error(`hash ${command.id} is not after an observation`);
                hashes.push({ step: Number(command.step), value: `0x${hashWorldState(world.state).toString(16).padStart(16, "0")}`, receiptId: command.id });
                break;
            default:
                throw new Error(`unknown command op ${command.op}`);
        }
    }
    const expectedObservations = scenario.stepCount;
    if (observations.length !== expectedObservations || hashes.length !== expectedObservations) throw new Error(`scenario ${scenario.id} has unconsumed schedule commands`);
    const requiredJoints = scenario.requiredJointIds ?? [];
    if (new Set(requiredJoints).size !== requiredJoints.length || joints.size !== requiredJoints.length || requiredJoints.some((id) => !joints.has(id))) throw new Error(`scenario ${scenario.id} has unconsumed or missing joint commands`);
    if (!world) throw new Error(`scenario ${scenario.id} has no world`);
    world.destroy();
    return { schema: "box3d-oracle/scenario-output/v1", id: scenario.id, name: scenario.name, corpusDigest: digest, observations, hashes, receipt: { corpusDigest: digest, consumedCommands: consumed, observationIds } };
}

export function runCorpus(path = process.env.BOX3D_SCENARIO_CORPUS, mutateAngularVelocity = false): ScenarioOutput[] {
    const { corpus, digest } = loadScenarioCorpus(path);
    return corpus.scenarios.map((scenario) => runScenario(scenario, digest, mutateAngularVelocity));
}
