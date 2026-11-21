import { Body, BodyType } from "../../core/physics";
import { readRenderGlobalTransform, VisualCorrection } from "../../core/rendering";
import { GlobalTransform } from "../../core/transform";
import {
    component,
    DiagnosticsStore,
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
import { type ReplicationConfirmation, ReplicationReceiver, type ReplicationSchema } from ".";
import {
    type AcceptedIntent,
    decodeIntent,
    type InputRecord,
    IntentMessageKind,
    type IntentRecord,
    intentWatermark,
    type OperationRecord,
} from "./intent";
import {
    ClientInterpolation,
    InterpolationPlugin,
    InterpolationTarget,
    multiply,
} from "./interpolation";
import type { Connection } from "./link";
import { PredictionRecording } from "./recording-hooks";

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
    dependencies: [StandardPhysicsPlugin, InterpolationPlugin],
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
 * arrives). Paced steps drain to the latest available confirmation and replay once. This controller
 * owns World's time scale (0.8–1.5); a game's slow motion belongs to the server's simulation,
 * while World.pause remains local. Pacing adjusts the next step toward an initial lead up to two
 * ticks. One smoothed command-age sample per server tick raises the target for late inputs;
 * 32 samples with more than two ticks of arrival slack permit a one-tick reduction.
 * Duplicate feedback never counts. The target reserves min(8, maxAhead / 2) ticks for confirmation jitter. Delays beyond
 * the available prediction window cannot be hidden: prediction holds, including its interpolation
 * fraction, while frames continue, then resumes on confirmation. Explicit tick beyond the bound
 * throws before gameplay; World's failed-tick clock semantics apply.
 * Intent values and permissions belong to the game. */
export class ClientPrediction {
    readonly receiver: ReplicationReceiver;
    readonly interpolation: ClientInterpolation;
    readonly records = new Map<number, OperationRecord | AcceptedIntent>();
    /** Local tick history for replay, independent of operation acceptance. */
    readonly inputs = new Map<number, InputRecord>();
    readonly outputs = new FixedOutputs();
    /** Samples each original predicted fixed tick before gameplay, never during replay.
     * Return that tick's input; replay uses the retained copy. IntentSender.record throws for an
     * input of more than 510 values, and the throw leaves the tick. */
    sampleInput: (tick: number) => InputRecord | undefined = () => undefined;
    /** Routes non-replication packets drained at preFixed, for example to IntentSender.flush. */
    onMessages: (messages: Uint8Array[]) => void = () => {};
    private _history = new Map<number, WorldSnapshot>();
    private _entities = new Map<number, number>();
    private _interpolated: number[] = [];
    private _renderScratch = new Float32Array(12);
    private _correctionInverse = [0, 0, 0, 1];
    private _correctionError = [0, 0, 0, 1];
    private _correctionRotation = [0, 0, 0, 1];
    private _ordered: OperationRecord[] = [];
    private _active: OperationRecord[] = [];
    private _switches: { tick: number; entity: EntityRef; enabled: boolean }[] = [];
    private _remote: {
        entity: EntityRef;
        translation: readonly [number, number, number];
        rotation: readonly [number, number, number, number];
    }[] = [];
    private _types = new Map<number, number>();
    private _replaying = false;
    private _replayedTicks = 0;
    private _confirmedTick: number;
    private _resolvedId = 0;
    private _confirming = false;
    private _hasConfirmed = false;
    private _feedbackTick = -1;
    private _onTime = 0;
    private _lead: number;
    private readonly _leadLimit: number;
    /** Target ticks beyond the newest confirmation, below maxAhead's jitter reserve. */
    get lead(): number {
        return this._lead;
    }
    readonly maxAhead: number;
    readonly world: World;
    /** Selects the local avatar's client eid after confirmed rows are applied. Ownership is the game's data. */
    readonly avatar: (world: World, entities: ReadonlyMap<number, number>) => number | undefined;
    /** Runs again on replay. Borrows operations through tick not yet folded into the newest
     * confirmed state, ordered by tick then ID, and the latest local input at or before tick.
     * Do not retain or mutate the borrowed data. Apply operations only at their tick or derive
     * cumulative counters. A server refusal retires speculative operations through its watermark;
     * inputs are replayed independently of operation acceptance and never replaced by an echo. */
    readonly applyIntent: (
        world: World,
        tick: number,
        records: readonly OperationRecord[],
        input?: InputRecord,
    ) => void;
    /** confirmed=true observes every authoritative state at its tick, with current accepted records
     * and ECS rows. The solver reconciles only the latest state in a drain. confirmed=false runs
     * after each original or replayed tick; FixedOutputs deduplicates its one-off deliveries. */
    readonly observe: (world: World, outputs: FixedOutputs, confirmed: boolean) => void;
    constructor(
        world: World,
        schema: ReplicationSchema,
        connection: Connection,
        avatar: (world: World, entities: ReadonlyMap<number, number>) => number | undefined,
        applyIntent: (
            world: World,
            tick: number,
            records: readonly OperationRecord[],
            input?: InputRecord,
        ) => void,
        observe: (world: World, outputs: FixedOutputs, confirmed: boolean) => void = () => {},
        maxAhead = 32,
    ) {
        if (!Number.isSafeInteger(maxAhead) || maxAhead < 0)
            throw new Error("Invalid prediction bound");
        this.maxAhead = maxAhead;
        this._leadLimit = maxAhead - Math.min(8, maxAhead / 2);
        this._lead = Math.min(2, this._leadLimit);
        this._confirmedTick = world.time.fixedTick;
        if (
            schema.includes(Body) &&
            (!schema.includes(GlobalTransform) || !schema.includes(PredictionMotion))
        )
            throw new Error(
                "Body prediction requires GlobalTransform and PredictionMotion in the replication schema",
            );
        this.world = world;
        const diagnostics = world.resource(DiagnosticsStore);
        diagnostics.record("network/replayed_ticks", 0, undefined, "ticks");
        diagnostics.record("network/correction_translation", 0, undefined, "world units");
        diagnostics.record("network/correction_rotation", 0, undefined, "rad");
        this.interpolation = new ClientInterpolation(
            world,
            (eid) => this.predicted(eid) || eid === this.avatar(world, this.receiver.entities),
        );
        this.avatar = avatar;
        this.applyIntent = applyIntent;
        this.observe = observe;
        this.receiver = new ReplicationReceiver(
            world,
            schema,
            connection,
            () => {},
            (tick, _rows, intents, apply, preceding) =>
                this._confirm(tick, intents, apply, preceding),
        );
        this.receiver.onCommandAge = (tick, age) => this.pace(tick, age);
        for (const eid of world.query([Body]))
            this._types.set(eid, world.storage(Body).type.get(eid));
        const work = world.resource(PredictionWork);
        if (work.prepare) throw new Error("Only one ClientPrediction may drive a World");
        work.prepare = () => this._prepare();
        work.finish = () => this._finish();
        work.confirm = () => {
            const messages = this.receiver.flush("latest");
            for (const bytes of messages)
                if (
                    bytes.length >= 8 &&
                    new DataView(bytes.buffer, bytes.byteOffset, bytes.byteLength).getUint32(
                        0,
                        true,
                    ) === IntentMessageKind.Accepted
                )
                    this.accept(decodeIntent(bytes), intentWatermark(bytes));
            this.onMessages(messages);
            const lead = world.time.fixedTick + world.time.fixedAlpha - this._confirmedTick;
            const scale = Math.max(0.8, Math.min(1.5, 1 + (this._lead - lead) * 0.1));
            world.setTimeScale(world.time.scale + (scale - world.time.scale) * 0.05);
            const remaining = Math.max(
                0,
                this._confirmedTick + this.maxAhead - world.time.fixedTick,
            );
            const accumulated = world.time.fixedAlpha * Time.FIXED_DT + world.time.deltaTime;
            if (remaining === 0) world.discardOverstep(world.time.deltaTime);
            else {
                const excess = Math.floor(accumulated / Time.FIXED_DT) - remaining;
                if (excess > 0) world.discardOverstep(excess * Time.FIXED_DT);
            }
        };
        world.resource(PredictionRecording).attach?.(this);
        this._save();
    }
    /** Copies local inputs into tick history; repeated ticks replace samples.
     * Merges local or accepted operations by ID. The resolved-ID watermark retires only
     * speculative operations, including server-refused ones, never local input history. */
    accept(records: Iterable<IntentRecord | AcceptedIntent>, watermark = 0): void {
        this._resolvedId = Math.max(this._resolvedId, watermark);
        for (const record of records) {
            if (record.kind === "input") {
                if (record.tick > this._confirmedTick)
                    this.inputs.set(record.tick, { ...record, values: record.values.slice() });
                continue;
            }
            if (!("lateness" in record) && record.id <= this._resolvedId) continue;
            const previous = this.records.get(record.id);
            if (previous && "lateness" in previous && !("lateness" in record)) continue;
            this.records.set(record.id, record);
        }
        this._ordered.length = 0;
        for (const [id, record] of this.records) {
            if (
                ("lateness" in record && !this._confirming && record.tick <= this._confirmedTick) ||
                (!("lateness" in record) && id <= this._resolvedId)
            )
                this.records.delete(id);
            else this._ordered.push(record);
        }
        this._ordered.sort((a, b) => a.tick - b.tick || a.id - b.id);
    }
    /** Consumes one smoothed server-command-age sample per server tick. Positive age is late.
     * A two-tick arrival reserve grows promptly; 32 excess-slack samples relax one lead tick.
     * Duplicate and reordered samples have no effect. */
    pace(tick: number, age: number): void {
        if (tick <= this._feedbackTick || !Number.isFinite(age)) return;
        this._feedbackTick = tick;
        if (age > -1) {
            this._lead = Math.min(this._leadLimit, this._lead + Math.min(1, (age + 2) / 8));
            this._onTime = 0;
        } else if (age < -2) {
            if (++this._onTime >= 32) {
                this._lead = Math.max(Math.min(2, this._leadLimit), this._lead - 1);
                this._onTime = 0;
            }
        } else this._onTime = 0;
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
        if (!this._replaying) {
            const record = this.world.resource(PredictionRecording).sample;
            const sample = record ? record(this, tick) : this.sampleInput(tick);
            if (sample) {
                if (sample.tick !== tick)
                    throw new Error("Input sample must address the current tick");
                this.accept([sample]);
            }
        }
        let input: InputRecord | undefined;
        for (const sample of this.inputs.values())
            if (sample.tick <= tick && (!input || sample.tick > input.tick)) input = sample;
        this.applyIntent(this.world, tick, this._active, input);
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
    private _motion() {
        return new Map(
            [...this.world.query(motionTerms)].map((eid) => [
                eid,
                {
                    pose: this.world.get(eid, GlobalTransform)!,
                    motion: this.world.get(eid, PredictionMotion)!,
                    type: this.world.storage(Body).type.get(eid),
                },
            ]),
        );
    }
    private _applyConfirmation(
        tick: number,
        intents: readonly AcceptedIntent[],
        apply: () => void,
        before: ReturnType<ClientPrediction["_motion"]> | undefined,
    ): void {
        this.accept(intents);
        if (this.world.time.fixedTick !== tick) this.world.jumpToTick(tick);
        apply();
        this._entities = new Map(this.receiver.entities);
        this._hasConfirmed = true;
        const avatar = this.avatar(this.world, this.receiver.entities);
        if (avatar !== undefined && !this.world.has(avatar, PredictionTarget))
            this.world.add(avatar, PredictionTarget, { enabled: 1 });
        if (before) this._remote.length = 0;
        if (before)
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
        let count = 0;
        for (const eid of this.receiver.entities.values()) {
            if (eid === avatar || this.predicted(eid)) continue;
            if (!this.world.has(eid, InterpolationTarget)) this.world.add(eid, InterpolationTarget);
            this._interpolated[count++] = eid;
        }
        this._interpolated.length = count;
        this.interpolation.confirm(tick, this._interpolated);
        this.observe(this.world, this.outputs, true);
    }
    private _renderPose(eid: number) {
        const pose = this._renderScratch;
        readRenderGlobalTransform(this.world, eid, pose);
        return {
            translation: [pose[0]!, pose[1]!, pose[2]!],
            rotation: [pose[4]!, pose[5]!, pose[6]!, pose[7]!],
        };
    }
    private _confirm(
        tick: number,
        intents: readonly AcceptedIntent[],
        apply: () => void,
        preceding: readonly ReplicationConfirmation[],
    ): void {
        if (this._replaying) throw new Error("Prediction cannot receive during replay");
        const displayed = new Map<
            number,
            { translation: readonly number[]; rotation: readonly number[] }
        >();
        for (const eid of this.world.query([PredictionTarget, GlobalTransform]))
            if (this.predicted(eid)) displayed.set(eid, this._renderPose(eid));
        const head = Math.max(tick, this.world.time.fixedTick);
        const history = this._history.get(tick);
        if (history) {
            const entities = !this._hasConfirmed ? new Map(this.receiver.entities) : this._entities;
            this.world.restore(history);
            this.receiver.entities.clear();
            for (const [id, eid] of entities) this.receiver.entities.set(id, eid);
        } else if (this.world.time.fixedTick > tick)
            throw new Error(`Missing prediction tick ${tick}`);
        const before = this._motion();
        this._confirming = true;
        try {
            for (const state of preceding)
                this._applyConfirmation(state.tick, state.intents, state.apply, undefined);
            this._applyConfirmation(tick, intents, apply, before);
        } finally {
            this._confirming = false;
        }
        this._confirmedTick = tick;
        let held = -1;
        for (const key of this.inputs.keys()) if (key <= tick) held = Math.max(held, key);
        for (const key of this.inputs.keys()) if (key < held) this.inputs.delete(key);
        this.accept([]);
        this._save();
        this._replaying = true;
        try {
            while (this.world.time.fixedTick < head) {
                this.world.tick();
                this._replayedTicks++;
            }
        } finally {
            this._replaying = false;
        }
        const diagnostics = this.world.resource(DiagnosticsStore);
        diagnostics.record("network/replayed_ticks", this._replayedTicks);
        let translationError = 0;
        let rotationError = 0;
        for (const [eid, value] of displayed) {
            if (!this.world.has(eid, GlobalTransform) || !this.predicted(eid)) continue;
            const previous = value;
            const current = this._renderPose(eid);
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
            for (let i = 0; i < 4; i++)
                this._correctionInverse[i] = current.rotation[i]! * (i === 3 ? 1 : -1);
            multiply(previous.rotation, this._correctionInverse, this._correctionError);
            translationError = Math.max(
                translationError,
                Math.hypot(
                    previous.translation[0]! - current.translation[0]!,
                    previous.translation[1]! - current.translation[1]!,
                    previous.translation[2]! - current.translation[2]!,
                ),
            );
            rotationError = Math.max(
                rotationError,
                2 * Math.acos(Math.min(1, Math.abs(this._correctionError[3]!))),
            );
            this._correctionRotation[0] = correction.rotation.x.get(eid);
            this._correctionRotation[1] = correction.rotation.y.get(eid);
            this._correctionRotation[2] = correction.rotation.z.get(eid);
            this._correctionRotation[3] = correction.rotation.w.get(eid);
            const rotation = multiply(
                this._correctionRotation,
                this._correctionError,
                this._correctionRotation,
            );
            correction.rotation.set(eid, rotation[0]!, rotation[1]!, rotation[2]!, rotation[3]!);
        }
        diagnostics.record("network/correction_translation", translationError);
        diagnostics.record("network/correction_rotation", rotationError);
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

function same(a: readonly number[] | undefined, b: readonly number[]): boolean {
    return a?.length === b.length && a.every((v, i) => v === b[i]);
}
