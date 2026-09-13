import { createHash } from "node:crypto";
import { readFileSync } from "node:fs";
import { BodyType, type Body } from "../api/index";
import { createCompound, createHeightField, createHull, createMesh, defaultFilter, defaultSurfaceMaterial, makeBoxHull, makeTransformedBoxHull, type HullData, type MeshData, type Shape } from "../api/index";
import { hashWorldState } from "../world/hash";
import { World } from "../api/world";

type Command = {
    op: string;
    id: string;
    [key: string]: unknown;
};
type Scenario = { id: string; name: string; commands: Command[]; stepCount: number; requiredJointIds?: string[]; requiredSensorEventIds?: string[] };
type Corpus = { schema: string; corpusVersion: number; scenarios: Scenario[] };
export type ScenarioOutput = {
    schema: "box3d-oracle/scenario-output/v1";
    id: string;
    name: string;
    corpusDigest: string;
    observations: unknown[];
    hashes: { step: number; value: string; receiptId: string }[];
    sensorEvents: { step: number; begin: { sensor: string; visitor: string }[]; end: { sensor: string; visitor: string }[]; receiptId: string }[];
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
const frame = (value: unknown, normalize = false): { p: { x: number; y: number; z: number }; q: { v: { x: number; y: number; z: number }; s: number } } => { if (!value || typeof value !== "object") throw new Error("expected a joint frame"); const record = value as Record<string, unknown>; const q = quat(record.q as unknown[]); return { p: vec3(record.p as unknown[]), q: normalize ? quat.normalize(q) : q }; };
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
    const compounds = new Map<string, NonNullable<ReturnType<typeof createCompound>>>();
    const hulls = new Map<string, HullData>();
    const shapes = new Map<string, Shape>();
    const consumed: string[] = [];
    const observationIds: string[] = [];
    const observations: unknown[] = [];
    const hashes: { step: number; value: string; receiptId: string }[] = [];
    const sensorEvents: ScenarioOutput["sensorEvents"] = [];
    let world: World | undefined;
    let observed = false;
    const material = (value: unknown): ReturnType<typeof defaultSurfaceMaterial> => {
        const source = (value ?? {}) as Record<string, unknown>;
        const tangent = vec3((source.tangentVelocity ?? ["0x00000000", "0x00000000", "0x00000000"]) as unknown[]);
        return { friction: f32(String(source.friction ?? "0x3f19999a")), restitution: f32(String(source.restitution ?? "0x00000000")), rollingResistance: f32(String(source.rollingResistance ?? "0x00000000")), tangentVelocity: tangent, userMaterialId: BigInt(String(source.userMaterialId ?? "0x0000000000000000")), customColor: Number(source.customColor ?? 0) };
    };
    const shapeIndex = (shape: Shape): number => (shape as unknown as { id: { index1: number } }).id.index1;
    const shapeName = (shape: Shape): string => { const index = shapeIndex(shape); for (const [id, value] of shapes) if (shapeIndex(value) === index) return id; return "unknown"; };
    const shapeDef = (command: Command): Record<string, unknown> => ({ baseMaterial: { ...defaultSurfaceMaterial(), friction: f32(String(command.friction ?? "0x3f19999a")), restitution: f32(String(command.restitution ?? "0x00000000")), rollingResistance: f32(String(command.rollingResistance ?? "0x00000000")) }, density: f32(String(command.density ?? "0x447a0000")), updateBodyMass: command.updateBodyMass !== false, invokeContactCreation: command.invokeContactCreation !== false, filter: { ...defaultFilter(), groupIndex: Number(command.groupIndex ?? 0) }, isSensor: command.isSensor === true, enableSensorEvents: command.enableSensorEvents === true });
    for (const command of scenario.commands) {
        if (consumed.includes(command.id)) throw new Error(`duplicate consumed command ${command.id}`);
        consumed.push(command.id);
        switch (command.op) {
            case "world.create": {
                if (world !== undefined) throw new Error("duplicate world.create");
                const gravity = vec3(command.gravity as unknown[]);
                world = new World({ gravity, enableSleep: command.enableSleep === true, enableContinuous: command.enableContinuous === true, ...(command.capacity === undefined ? {} : { capacity: command.capacity }) });
                break;
            }
            case "body.create":
            case "body.spawn": {
                if (!world) throw new Error("body.create before world.create");
                const position = vec3(command.position as unknown[]);
                const linearVelocity = vec3(command.linearVelocity as unknown[]);
                const angularVelocity = vec3(command.angularVelocity as unknown[]);
                if (mutateAngularVelocity && bodies.size === 0) angularVelocity.z = Math.fround(angularVelocity.z + 1);
                const body = world.createBody({ type: bodyType(command.type), isBullet: command.isBullet === true, position, rotation: quat((command.rotation ?? ["0x00000000", "0x00000000", "0x00000000", "0x3f800000"]) as unknown[]), linearVelocity, angularVelocity, ...(command.linearDamping === undefined ? {} : { linearDamping: f32(String(command.linearDamping)) }), ...(command.angularDamping === undefined ? {} : { angularDamping: f32(String(command.angularDamping)) }), ...(command.sleepThreshold === undefined ? {} : { sleepThreshold: f32(String(command.sleepThreshold)) }) });
                bodies.set(command.id, body);
                break;
            }
            case "resource.box": {
                const halfExtents = vec3(command.halfExtents as unknown[]); const center = vec3((command.center ?? ["0x00000000", "0x00000000", "0x00000000"]) as unknown[]);
                boxes.set(command.id, center.x === 0 && center.y === 0 && center.z === 0 ? makeBoxHull(halfExtents.x, halfExtents.y, halfExtents.z) : makeTransformedBoxHull(halfExtents.x, halfExtents.y, halfExtents.z, { p: center, q: quat(["0x00000000", "0x00000000", "0x00000000", "0x3f800000"]) }));
                break;
            }
            case "resource.sphere":
                spheres.set(command.id, { center: vec3((command.center ?? ["0x00000000", "0x00000000", "0x00000000"]) as unknown[]), radius: f32(String(command.radius)) });
                break;
            case "resource.capsule":
                capsules.set(command.id, { center1: vec3(command.center1 as unknown[]), center2: vec3(command.center2 as unknown[]), radius: f32(String(command.radius)) });
                break;
            case "resource.hull": {
                const points = (command.points as unknown[]).map((point) => vec3(point as unknown[]));
                const hull = createHull(points, points.length);
                if (!hull) throw new Error(`hull resource ${command.id} could not be built`);
                hulls.set(command.id, hull as HullData);
                break;
            }
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
            case "resource.compound": {
                const children = (field: string): Record<string, unknown>[] => { const value = command[field]; if (value === undefined) return []; if (!Array.isArray(value)) throw new Error(`compound ${command.id} ${field} is not an array`); return value as Record<string, unknown>[]; };
                const childIds = ["capsules", "hulls", "meshes", "spheres"].flatMap((field) => children(field).map((child) => String(child.id ?? "")));
                if (childIds.some((id) => !id) || new Set(childIds).size !== childIds.length || JSON.stringify(childIds) !== JSON.stringify(command.childOrder ?? [])) throw new Error(`compound ${command.id} child order is missing or unconsumed`);
                const compoundCapsules = children("capsules").map((child) => { const resource = capsules.get(String(child.resource)); if (!resource) throw new Error(`compound references unknown capsule ${String(child.resource)}`); return { capsule: resource, material: material(child.material) }; });
                const compoundHulls = children("hulls").map((child) => { const resource = boxes.get(String(child.resource)); if (!resource) throw new Error(`compound references unknown hull ${String(child.resource)}`); return { hull: resource, transform: frame(child.transform), material: material(child.material) }; });
                const compoundMeshes = children("meshes").map((child) => { const resource = meshes.get(String(child.resource)); if (!resource) throw new Error(`compound references unknown mesh ${String(child.resource)}`); const values = (child.materials ?? [child.material]) as unknown[]; if (!Array.isArray(values) || values.length < 1 || values.length > 4) throw new Error(`compound mesh material list is invalid for ${command.id}`); return { meshData: resource, transform: frame(child.transform), scale: vec3((child.scale ?? ["0x3f800000", "0x3f800000", "0x3f800000"]) as unknown[]), materials: values.map(material), materialCount: values.length }; });
                const compoundSpheres = children("spheres").map((child) => { const resource = spheres.get(String(child.resource)); if (!resource) throw new Error(`compound references unknown sphere ${String(child.resource)}`); return { sphere: resource, material: material(child.material) }; });
                const compound = createCompound({ capsules: compoundCapsules, hulls: compoundHulls, meshes: compoundMeshes, spheres: compoundSpheres }); if (!compound) throw new Error(`compound ${command.id} could not be built`); compounds.set(command.id, compound); break;
            }
            case "shape.create": {
                if (!world) throw new Error("shape.create before world.create");
                const body = bodies.get(String(command.body));
                if (!body) throw new Error(`shape references unknown body ${String(command.body)}`);
                const def = shapeDef(command) as never;
                let created: Shape;
                if (command.kind === "box") {
                    const box = boxes.get(String(command.resource)); if (!box) throw new Error(`shape references unknown box ${String(command.resource)}`); created = body.createHull(def, box);
                } else if (command.kind === "sphere") {
                    const sphere = spheres.get(String(command.resource)); if (!sphere) throw new Error(`shape references unknown sphere ${String(command.resource)}`); created = body.createSphere(def, sphere);
                } else if (command.kind === "capsule") {
                    const capsule = capsules.get(String(command.resource)); if (!capsule) throw new Error(`shape references unknown capsule ${String(command.resource)}`); created = body.createCapsule(def, capsule);
                } else if (command.kind === "hull") {
                    const hull = hulls.get(String(command.resource)); if (!hull) throw new Error(`shape references unknown hull ${String(command.resource)}`); created = body.createHull(def, hull);
                } else if (command.kind === "mesh") {
                    const mesh = meshes.get(String(command.resource)); if (!mesh) throw new Error(`shape references unknown mesh ${String(command.resource)}`); created = body.createMesh(def, mesh, vec3((command.scale ?? ["0x3f800000", "0x3f800000", "0x3f800000"]) as unknown[]));
                } else if (command.kind === "height-field") {
                    const field = heightFields.get(String(command.resource)); if (!field) throw new Error(`shape references unknown height field ${String(command.resource)}`); created = body.createHeightField(def, field);
                } else if (command.kind === "compound") {
                    const compound = compounds.get(String(command.resource)); if (!compound) throw new Error(`shape references unknown compound ${String(command.resource)}`); created = body.createCompound(def, compound);
                } else throw new Error(`unknown shape kind ${String(command.kind)}`);
                shapes.set(command.id, created);
                break;
            }
            case "joint.filter": {
                if (!world) throw new Error(`filter joint ${command.id} before world.create`);
                const bodyA = bodies.get(String(command.bodyA)); const bodyB = bodies.get(String(command.bodyB));
                if (!bodyA || !bodyB) throw new Error(`filter joint ${command.id} references an unknown body`);
                world.createFilterJoint(bodyA, bodyB); joints.add(command.id); break;
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
                const config: Record<string, unknown> = { localFrameA: frame(command.localFrameA, command.normalizeFrames === true), localFrameB: frame(command.localFrameB, command.normalizeFrames === true) };
                for (const [key, value] of Object.entries(command)) {
                    if (["op", "id", "bodyA", "bodyB", "localFrameA", "localFrameB", "normalizeFrames"].includes(key)) continue;
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
            case "body.apply-mass": {
                const body = bodies.get(String(command.body)); if (!body) throw new Error(`mass action references unknown body ${String(command.body)}`); body.applyMassFromShapes(); break;
            }
            case "body.set-velocity": {
                const body = bodies.get(String(command.body)); if (!body) throw new Error(`velocity action references unknown body ${String(command.body)}`); body.setLinearVelocity(vec3(command.linearVelocity as unknown[])); body.setAngularVelocity(vec3(command.angularVelocity as unknown[])); break;
            }
            case "body.target-transform": {
                const body = bodies.get(String(command.body)); if (!body) throw new Error(`target action references unknown body ${String(command.body)}`); body.setTargetTransform({ p: vec3((command.target as Record<string, unknown>).p as unknown[]), q: quat((command.target as Record<string, unknown>).q as unknown[]) }, f32(String(command.timeStep)), command.wake === true); break;
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
            case "sensor.events": {
                if (!world) throw new Error(`sensor event ${command.id} before world.create`);
                const sensor = shapes.get(String(command.shape)); if (!sensor || !sensor.isSensor()) throw new Error(`sensor event references unknown sensor ${String(command.shape)}`);
                const events = world.getSensorEvents();
                const select = (shape: Shape): boolean => shapeName(shape) === String(command.shape);
                sensorEvents.push({ step: Number(command.step), begin: events.beginEvents.filter((event) => select(event.sensor)).map((event) => ({ sensor: shapeName(event.sensor), visitor: shapeName(event.visitor) })), end: events.endEvents.filter((event) => select(event.sensor)).map((event) => ({ sensor: shapeName(event.sensor), visitor: shapeName(event.visitor) })), receiptId: command.id });
                break;
            }
            default:
                throw new Error(`unknown command op ${command.op}`);
        }
    }
    const expectedObservations = scenario.stepCount;
    const expectedEvents = scenario.commands.filter((command) => command.op === "sensor.events").map((command) => String(command.id));
    if (observations.length !== expectedObservations || hashes.length !== expectedObservations || sensorEvents.length !== expectedEvents.length || JSON.stringify(expectedEvents) !== JSON.stringify(scenario.requiredSensorEventIds ?? [])) throw new Error(`scenario ${scenario.id} has unconsumed schedule commands`);
    const requiredJoints = scenario.requiredJointIds ?? [];
    if (new Set(requiredJoints).size !== requiredJoints.length || joints.size !== requiredJoints.length || requiredJoints.some((id) => !joints.has(id))) throw new Error(`scenario ${scenario.id} has unconsumed or missing joint commands`);
    if (!world) throw new Error(`scenario ${scenario.id} has no world`);
    world.destroy();
    return { schema: "box3d-oracle/scenario-output/v1", id: scenario.id, name: scenario.name, corpusDigest: digest, observations, hashes, sensorEvents, receipt: { corpusDigest: digest, consumedCommands: consumed, observationIds } };
}

export function runCorpus(path = process.env.BOX3D_SCENARIO_CORPUS, mutateAngularVelocity = false): ScenarioOutput[] {
    const { corpus, digest } = loadScenarioCorpus(path);
    return corpus.scenarios.map((scenario) => runScenario(scenario, digest, mutateAngularVelocity));
}
