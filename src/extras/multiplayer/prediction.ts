import { Body, BodyType } from "../../core/physics";
import { VisualCorrection } from "../../core/rendering";
import { GlobalTransform } from "../../core/transform";
import {
    component,
    type EntityRef,
    type Plugin,
    type System,
    Time,
    u8,
    vec3,
    type World,
    type WorldSnapshot,
} from "../../engine";
import {
    CharacterPlugin,
    getAngularVelocity,
    isAwake,
    StandardPhysicsPlugin,
    StepPhysicsSystem,
    setAngularVelocity,
    setAwake,
    setBodyType,
    setLinearVelocity,
    setTransform,
} from "../../standard/physics";
import { ReplicationReceiver, type ReplicationSchema } from ".";
import type { AcceptedIntent, IntentRecord } from "./intent";
import type { Connection } from "./link";

/** Local display selection. ClientPrediction adds it to the local avatar; other entities opt in. */
export const PredictionTarget = component(
    "PredictionTarget",
    { enabled: u8 },
    { defaults: () => ({ enabled: 1 }) },
);
/** Live solver state required by a correction; Body's velocity and awake defaults are spawn-only. */
export const PredictionMotion = component("PredictionMotion", { angularVelocity: vec3, awake: u8 });

const bodyTerms = [Body];
const motionTerms = [Body, GlobalTransform, PredictionMotion];
const MotionScratch = { create: () => ({ x: 0, y: 0, z: 0 }) };
const SampleMotionSystem: System = {
    name: "sample-prediction-motion",
    group: "fixed",
    after: [StepPhysicsSystem, ...(CharacterPlugin.systems ?? [])],
    update(world) {
        const scratch = world.resource(MotionScratch);
        for (const eid of world.query(bodyTerms)) {
            const angular = getAngularVelocity(world, eid, scratch);
            if (!angular) continue;
            if (!world.has(eid, PredictionMotion)) world.add(eid, PredictionMotion);
            world
                .storage(PredictionMotion)
                .angularVelocity.set(eid, angular.x, angular.y, angular.z);
            world.storage(PredictionMotion).awake.set(eid, Number(isAwake(world, eid)));
        }
    },
};
const PredictionWork = {
    create: (): { prepare?: () => void; finish?: () => void; confirm?: () => void } => ({}),
};
export const PredictionPlugin: Plugin = {
    name: "Prediction",
    dependencies: [StandardPhysicsPlugin],
    components: [PredictionTarget, PredictionMotion, VisualCorrection],
    recovery: "stateless",
    systems: [
        SampleMotionSystem,
        {
            name: "prepare-prediction",
            group: "fixed",
            first: true,
            update: (world) => world.resource(PredictionWork).prepare?.(),
        },
        {
            name: "capture-prediction",
            group: "postFixed",
            last: true,
            update: (world) => world.resource(PredictionWork).finish?.(),
        },
        {
            name: "confirm-prediction",
            group: "preFixed",
            update: (world) => world.resource(PredictionWork).confirm?.(),
        },
    ],
};

/** Presentation deliveries outlive World recovery. The game supplies a stable key within a tick. */
export class FixedOutputs {
    private _delivered = new Map<number, Set<string>>();
    deliver(tick: number, key: string, emit: () => void): void {
        let keys = this._delivered.get(tick);
        if (!keys) this._delivered.set(tick, (keys = new Set()));
        if (keys.has(key)) return;
        keys.add(key);
        emit();
    }
    /** Forget only ticks that can no longer be replayed. */
    discardBefore(tick: number): void {
        for (const old of this._delivered.keys()) if (old < tick) this._delivered.delete(old);
    }
}

/** Whole-World rollback over ordered confirmations. Install PredictionPlugin to prepare and capture
 * explicit ticks and paced steps, and process confirmations at preFixed. One controller per World.
 * maxAhead defaults to 32 ticks beyond the latest confirmation (or construction tick until one
 * arrives). Paced prediction holds at that bound while frames continue, then resumes on confirmation.
 * Explicit tick beyond the bound throws before gameplay; World's failed-tick clock semantics apply.
 * Intent values and permissions belong to the game. */
export class ClientPrediction {
    readonly receiver: ReplicationReceiver;
    readonly records = new Map<number, IntentRecord | AcceptedIntent>();
    readonly outputs = new FixedOutputs();
    /** Routes non-replication packets drained at preFixed, for example to IntentSender.flush. */
    onMessages: (messages: Uint8Array[]) => void = () => {};
    private _history = new Map<number, WorldSnapshot>();
    private _entities = new Map<number, number>();
    private _ordered: IntentRecord[] = [];
    private _active: IntentRecord[] = [];
    private _switches: { tick: number; entity: EntityRef; enabled: boolean }[] = [];
    private _remote: {
        entity: EntityRef;
        translation: readonly [number, number, number];
        rotation: readonly [number, number, number, number];
    }[] = [];
    private _types = new Map<number, number>();
    private _replaying = false;
    private _confirmedTick: number;
    readonly maxAhead: number;
    readonly world: World;
    /** Selects the local avatar's client eid after confirmed rows are applied. Ownership is the game's data. */
    readonly avatar: (world: World, entities: ReadonlyMap<number, number>) => number | undefined;
    /** Runs again on replay. Borrows the cumulative log through tick, ordered by tick then ID;
     * do not retain the array. Apply operations only at their tick or derive cumulative counters. */
    readonly applyIntent: (world: World, tick: number, records: readonly IntentRecord[]) => void;
    readonly observe: (world: World, outputs: FixedOutputs, confirmed: boolean) => void;
    constructor(
        world: World,
        schema: ReplicationSchema,
        connection: Connection,
        avatar: (world: World, entities: ReadonlyMap<number, number>) => number | undefined,
        applyIntent: (world: World, tick: number, records: readonly IntentRecord[]) => void,
        observe: (world: World, outputs: FixedOutputs, confirmed: boolean) => void = () => {},
        maxAhead = 32,
    ) {
        if (!Number.isSafeInteger(maxAhead) || maxAhead < 0)
            throw new Error("Invalid prediction bound");
        this.maxAhead = maxAhead;
        this._confirmedTick = world.time.fixedTick;
        if (
            schema.includes(Body) &&
            (!schema.includes(GlobalTransform) || !schema.includes(PredictionMotion))
        )
            throw new Error(
                "Body prediction requires GlobalTransform and PredictionMotion in the replication schema",
            );
        this.world = world;
        this.avatar = avatar;
        this.applyIntent = applyIntent;
        this.observe = observe;
        this.receiver = new ReplicationReceiver(
            world,
            schema,
            connection,
            () => {},
            (tick, _rows, intents, apply) => this._confirm(tick, intents, apply),
        );
        for (const eid of world.query([Body]))
            this._types.set(eid, world.storage(Body).type.get(eid));
        const work = world.resource(PredictionWork);
        if (work.prepare) throw new Error("Only one ClientPrediction may drive a World");
        work.prepare = () => this._prepare();
        work.finish = () => this._finish();
        work.confirm = () => {
            this.onMessages(this.receiver.flush());
            const remaining = Math.max(
                0,
                this._confirmedTick + this.maxAhead - world.time.fixedTick,
            );
            const accumulated = world.time.fixedAlpha * Time.FIXED_DT + world.time.deltaTime;
            const excess = Math.floor(accumulated / Time.FIXED_DT) - remaining;
            if (excess > 0) world.discardOverstep(excess * Time.FIXED_DT);
        };
        this._save();
    }
    accept(records: Iterable<IntentRecord | AcceptedIntent>): void {
        for (const record of records) {
            const previous = this.records.get(record.id);
            if (previous && "lateness" in previous && !("lateness" in record)) continue;
            this.records.set(record.id, record);
        }
        this._ordered.length = 0;
        for (const record of this.records.values()) this._ordered.push(record);
        this._ordered.sort((a, b) => a.tick - b.tick || a.id - b.id);
    }
    private _save(): void {
        this._history.set(this.world.time.fixedTick, this.world.snapshot());
    }
    private _prepare(): void {
        const tick = this.world.time.fixedTick;
        if (tick > this._confirmedTick + this.maxAhead)
            throw new Error(
                `Prediction tick ${tick} exceeds confirmed tick ${this._confirmedTick} plus ${this.maxAhead}`,
            );
        for (const change of this._switches) {
            if (change.tick > tick) continue;
            const eid = this.world.resolve(change.entity);
            if (!eid) continue;
            if (!this.world.has(eid, PredictionTarget)) this.world.add(eid, PredictionTarget);
            this.world.storage(PredictionTarget).enabled.set(eid, Number(change.enabled));
        }
        for (const eid of this.world.query(bodyTerms)) {
            const type = this.world.storage(Body).type.get(eid);
            if (type === BodyType.Static) continue;
            const predicted = this.predicted(eid);
            const desired = predicted ? (this._types.get(eid) ?? type) : BodyType.Kinematic;
            if (type !== desired) this._mode(eid, desired);
        }
        let count = 0;
        for (let i = 0; i < this._ordered.length; i++) {
            const record = this._ordered[i]!;
            if (record.tick > tick) break;
            this._active[count++] = record;
        }
        this._active.length = count;
        this.applyIntent(this.world, tick, this._active);
    }
    private _finish(): void {
        for (let i = 0; i < this._remote.length; i++) {
            const confirmed = this._remote[i]!;
            const eid = this.world.resolve(confirmed.entity);
            if (!eid || !this.world.has(eid, Body) || this.predicted(eid)) continue;
            const pose = this.world.storage(GlobalTransform);
            if (
                pose.translation.x.get(eid) !== confirmed.translation[0] ||
                pose.translation.y.get(eid) !== confirmed.translation[1] ||
                pose.translation.z.get(eid) !== confirmed.translation[2] ||
                pose.rotation.x.get(eid) !== confirmed.rotation[0] ||
                pose.rotation.y.get(eid) !== confirmed.rotation[1] ||
                pose.rotation.z.get(eid) !== confirmed.rotation[2] ||
                pose.rotation.w.get(eid) !== confirmed.rotation[3]
            )
                setTransform(this.world, eid, confirmed.translation, confirmed.rotation);
        }
        this.observe(this.world, this.outputs, false);
        this._save();
    }
    /** A mode change preserves pose and velocities. The selection is replayed at the same boundary. */
    switchBody(eid: number, enabled: boolean): void {
        this._switches.push({
            tick: this.world.time.fixedTick + 1,
            entity: this.world.ref(eid),
            enabled,
        });
        if (!this.world.has(eid, PredictionTarget)) this.world.add(eid, PredictionTarget);
        this.world.storage(PredictionTarget).enabled.set(eid, Number(enabled));
        this._mode(eid, enabled ? (this._types.get(eid) ?? BodyType.Dynamic) : BodyType.Kinematic);
    }
    predicted(eid: number): boolean {
        return (
            this.world.has(eid, PredictionTarget) &&
            this.world.storage(PredictionTarget).enabled.get(eid) !== 0
        );
    }
    private _mode(eid: number, type: number): void {
        if (this.world.storage(Body).type.get(eid) === type) return;
        const pose = this.world.get(eid, GlobalTransform);
        const angular = getAngularVelocity(this.world, eid);
        setBodyType(this.world, eid, type);
        if (pose)
            setLinearVelocity(this.world, eid, {
                x: pose.linearVelocity[0]!,
                y: pose.linearVelocity[1]!,
                z: pose.linearVelocity[2]!,
            });
        if (angular) setAngularVelocity(this.world, eid, angular);
    }
    private _confirm(tick: number, intents: readonly AcceptedIntent[], apply: () => void): void {
        if (this._replaying) throw new Error("Prediction cannot receive during replay");
        this._confirmedTick = tick;
        this.accept(intents);
        const displayed = new Map<
            number,
            { translation: readonly number[]; rotation: readonly number[] }
        >();
        for (const eid of this.world.query([PredictionTarget, GlobalTransform]))
            if (this.predicted(eid)) displayed.set(eid, this.world.get(eid, GlobalTransform)!);
        const head = Math.max(tick, this.world.time.fixedTick);
        const history = this._history.get(tick);
        if (history) {
            const entities =
                this.receiver.confirmed < 0 ? new Map(this.receiver.entities) : this._entities;
            this.world.restore(history);
            this.receiver.entities.clear();
            for (const [id, eid] of entities) this.receiver.entities.set(id, eid);
        } else if (this.world.time.fixedTick > tick)
            throw new Error(`Missing prediction tick ${tick}`);
        else while (this.world.time.fixedTick < tick) this.world.tick();
        const before = new Map(
            [...this.world.query(motionTerms)].map((eid) => [
                eid,
                {
                    pose: this.world.get(eid, GlobalTransform)!,
                    motion: this.world.get(eid, PredictionMotion)!,
                    type: this.world.storage(Body).type.get(eid),
                },
            ]),
        );
        apply();
        this._entities = new Map(this.receiver.entities);
        const avatar = this.avatar(this.world, this.receiver.entities);
        if (avatar !== undefined && !this.world.has(avatar, PredictionTarget))
            this.world.add(avatar, PredictionTarget, { enabled: 1 });
        this._remote.length = 0;
        for (const eid of this.world.query(motionTerms)) {
            const pose = this.world.get(eid, GlobalTransform)!;
            const motion = this.world.get(eid, PredictionMotion)!;
            const body = this.world.storage(Body);
            const type = body.type.get(eid);
            if (!getAngularVelocity(this.world, eid)) {
                body.linearVelocity.set(eid, ...pose.linearVelocity);
                body.angularVelocity.set(eid, ...motion.angularVelocity);
                body.isAwake.set(eid, motion.awake);
            }
            this._types.set(eid, type);
            const predicted = this.predicted(eid);
            if (!predicted)
                this._remote.push({
                    entity: this.world.ref(eid),
                    translation: pose.translation,
                    rotation: pose.rotation,
                });
            const previous = before.get(eid);
            const desired = type === BodyType.Static || predicted ? type : BodyType.Kinematic;
            if (previous?.type !== desired) setBodyType(this.world, eid, desired);
            else this.world.storage(Body).type.set(eid, desired);
            if (
                !same(previous?.pose.translation, pose.translation) ||
                !same(previous?.pose.rotation, pose.rotation)
            )
                setTransform(this.world, eid, pose.translation, pose.rotation);
            if (!same(previous?.pose.linearVelocity, pose.linearVelocity))
                setLinearVelocity(this.world, eid, {
                    x: pose.linearVelocity[0]!,
                    y: pose.linearVelocity[1]!,
                    z: pose.linearVelocity[2]!,
                });
            if (!same(previous?.motion.angularVelocity, motion.angularVelocity))
                setAngularVelocity(this.world, eid, {
                    x: motion.angularVelocity[0]!,
                    y: motion.angularVelocity[1]!,
                    z: motion.angularVelocity[2]!,
                });
            if (previous?.motion.awake !== motion.awake)
                setAwake(this.world, eid, motion.awake !== 0);
        }
        this.observe(this.world, this.outputs, true);
        this._save();
        this._replaying = true;
        try {
            while (this.world.time.fixedTick < head) this.world.tick();
        } finally {
            this._replaying = false;
        }
        for (const [eid, value] of displayed) {
            if (!this.world.has(eid, GlobalTransform) || !this.predicted(eid)) continue;
            const previous = value;
            const current = this.world.get(eid, GlobalTransform)!;
            if (
                same(previous.translation, current.translation) &&
                same(previous.rotation, current.rotation)
            )
                continue;
            if (!this.world.has(eid, VisualCorrection)) this.world.add(eid, VisualCorrection);
            const correction = this.world.storage(VisualCorrection);
            correction.translation.set(
                eid,
                correction.translation.x.get(eid) +
                    previous.translation[0]! -
                    current.translation[0]!,
                correction.translation.y.get(eid) +
                    previous.translation[1]! -
                    current.translation[1]!,
                correction.translation.z.get(eid) +
                    previous.translation[2]! -
                    current.translation[2]!,
            );
            const error = multiply(previous.rotation, [
                -current.rotation[0]!,
                -current.rotation[1]!,
                -current.rotation[2]!,
                current.rotation[3]!,
            ]);
            const rotation = multiply(this.world.get(eid, VisualCorrection)!.rotation, error);
            correction.rotation.set(eid, rotation[0]!, rotation[1]!, rotation[2]!, rotation[3]!);
        }
        for (const old of this._history.keys()) if (old < tick) this._history.delete(old);
        this.outputs.discardBefore(tick);
        this._switches = this._switches.filter((change) => change.tick > tick);
    }
}

/** Contact identities include childIndex so compound-shape outputs remain distinct. */
export function contactOutputKey(
    kind: string,
    event: { shapeA: EntityRef; shapeB: EntityRef; childIndex: number },
): string {
    return `${kind}:${event.shapeA}:${event.shapeB}:${event.childIndex}`;
}

function multiply(a: readonly number[], b: readonly number[]): number[] {
    const [x, y, z, w] = a as [number, number, number, number];
    const [u, v, s, t] = b as [number, number, number, number];
    return [
        w * u + x * t + y * s - z * v,
        w * v - x * s + y * t + z * u,
        w * s + x * v - y * u + z * t,
        w * t - x * u - y * v - z * s,
    ];
}

function same(a: readonly number[] | undefined, b: readonly number[]): boolean {
    return a?.length === b.length && a.every((v, i) => v === b[i]);
}
