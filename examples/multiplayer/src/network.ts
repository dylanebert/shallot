import {
    Body,
    Character,
    component,
    GlobalTransform,
    type Plugin,
    Shape,
    Transform,
    u32,
    type World,
} from "@dylanebert/shallot";
import {
    ClientPrediction,
    type Connection,
    type Game,
    type InputRecord,
    type IntentRecord,
    IntentSender,
    MultiplayerPlugin,
    PredictionMotion,
    PredictionPlugin,
    PredictionTarget,
    Replicate,
} from "@dylanebert/shallot/multiplayer";
import { Demo, Route } from "./demo";
import { PlayerMotion } from "./motion";
import { LocalPlayer, LocalPlayerInputSystem, Player, PlayerInput } from "./player";

const AVATAR = [
    Transform,
    GlobalTransform,
    Body,
    Shape,
    Character,
    Player,
    PlayerInput,
    PlayerMotion,
    PredictionMotion,
];
const PUBLIC = [Transform, GlobalTransform, Body, Shape, PredictionMotion];
const ConnectionSlot = component("ConnectionSlot", { slot: u32 });
/** Marks a client's own connection entity; disclosed only to that client, as lightyear's Controlled. */
const OwnSeat = component("OwnSeat", {});
export const SCHEMA = [...AVATAR, ConnectionSlot, OwnSeat];
const SLOTS = [0, 1] as const;
const Connections: Plugin = { name: "GameConnections", components: [ConnectionSlot, OwnSeat] };
export const GAME_PLUGINS = [Demo, PredictionPlugin, MultiplayerPlugin, Connections];
export function scene(world: World) {
    const result = world.resource(Route).entities;
    if (!result) throw new Error("Route is not initialized");
    return result;
}
export function validIntent(record: IntentRecord) {
    const v = record.values;
    return (
        record.kind === "input" &&
        v.length === 6 &&
        v.every(Number.isFinite) &&
        Math.abs(v[0]!) <= 1 &&
        Math.abs(v[1]!) <= 1 &&
        Math.abs(v[2]!) <= 1e6 &&
        Math.abs(v[3]!) < Math.PI / 2 &&
        (v[4] === 0 || v[4] === 1) &&
        Number.isSafeInteger(v[5]) &&
        v[5]! >= 0 &&
        v[5]! <= 0xffffffff
    );
}
export function applyIntent(
    world: World,
    player: number,
    tick: number,
    _records: readonly IntentRecord[],
    latest?: InputRecord,
) {
    if (!latest || latest.tick > tick || !validIntent(latest)) return;
    if (!world.has(player, PlayerInput)) world.add(player, PlayerInput);
    const input = world.storage(PlayerInput),
        v = latest.values;
    input.move.set(player, v[0]!, v[1]!);
    input.yaw.set(player, v[2]!);
    input.pitch.set(player, v[3]!);
    input.sprint.set(player, v[4]!);
    input.jumpPresses.set(player, v[5]!);
}
export const GAME: Game = {
    plugins: GAME_PLUGINS,
    schema: SCHEMA,
    maxClients: 2,
    onCreate(world) {
        for (const eid of world.query([Body])) world.add(eid, Replicate);
        for (const [slot, { player }] of scene(world).players.entries())
            world.storage(Replicate).rule.set(player, slot + 1);
    },
    onJoin(world, eid) {
        const occupied = [...world.query([ConnectionSlot])].map(
            (eid) => world.get(eid, ConnectionSlot)!.slot,
        );
        const slot = SLOTS.find((slot) => !occupied.includes(slot))!;
        world.add(eid, ConnectionSlot, { slot });
        world.add(eid, OwnSeat);
        world.add(eid, Replicate, { rule: SLOTS.length + 1 + slot });
    },
    rules(world, eid) {
        const slot = world.get(eid, ConnectionSlot)!.slot;
        return [
            new Map([[eid, AVATAR]]),
            ...SLOTS.map((owner) => new Map([[eid, owner === slot ? AVATAR : PUBLIC]])),
            ...SLOTS.map(
                (owner) =>
                    new Map([[eid, owner === slot ? [ConnectionSlot, OwnSeat] : [ConnectionSlot]]]),
            ),
        ];
    },
    onLeave(world, eid) {
        const slot = world.get(eid, ConnectionSlot)?.slot;
        if (slot === undefined) return;
        const player = scene(world).players[slot]!.player;
        if (world.has(player, PlayerInput)) world.storage(PlayerInput).move.set(player, 0, 0);
    },
    validateIntent: (_world, _eid, record) => validIntent(record),
    applyIntent(world, eid, tick, records, input) {
        const slot = world.get(eid, ConnectionSlot)!.slot;
        applyIntent(world, scene(world).players[slot]!.player, tick, records, input);
    },
    onClient(world, connection) {
        const client = new GameClient(world, connection);
        world.resource(ClientSession).client = client;
        world.addSystem({
            name: "select-prediction",
            group: "setup",
            update: () => client.selectPrediction(),
        });
    },
};
export class GameClient {
    readonly prediction: ClientPrediction;
    readonly intent: IntentSender;
    readonly world: World;
    private _values = [0, 0, 0, 0, 0, 0];
    constructor(world: World, connection: Connection) {
        this.world = world;
        this.intent = new IntentSender(connection);
        // The lift's trajectory is tick-derived gameplay, so the rider predicts it too.
        world.add(scene(world).lift, PredictionTarget);
        this.prediction = new ClientPrediction(
            world,
            SCHEMA,
            connection,
            () => (this.slot < 0 ? undefined : scene(world).players[this.slot]!.player),
            (_world, tick, records, input) => {
                if (this.slot >= 0)
                    applyIntent(
                        world,
                        scene(world).players[this.slot]!.player,
                        tick,
                        records,
                        input,
                    );
            },
        );
        this.prediction.sampleInput = (tick) => {
            const input = this.intent.record(tick, "input", this._values);
            this.intent.flush();
            return input;
        };
        // Both ends compose the same route; wire IDs follow its selected entity order.
        const selected = [...world.query([Body])];
        for (const [index, eid] of selected.entries()) {
            this.prediction.receiver.entities.set(index + 1, eid);
        }
        this.prediction.onMessages = (messages) => {
            this.intent.flush(messages);
        };
    }
    /** This client's slot, from the connection entity disclosed with OwnSeat; -1 before it arrives. */
    get slot() {
        const [seat] = this.world.query([OwnSeat, ConnectionSlot]);
        return seat === undefined ? -1 : this.world.get(seat, ConnectionSlot)!.slot;
    }
    record(values: number[]) {
        this._values = values.slice();
    }
    selectPrediction() {
        if (this.slot >= 0) {
            const { player } = scene(this.world).players[this.slot]!;
            const box = scene(this.world).box;
            if (this.world.has(player, GlobalTransform) && this.world.has(box, GlobalTransform)) {
                const p = this.world.get(player, GlobalTransform)!.translation;
                const b = this.world.get(box, GlobalTransform)!.translation;
                const near = Math.hypot(p[0]! - b[0]!, p[1]! - b[1]!, p[2]! - b[2]!) < 4;
                if (this.prediction.predicted(box) !== near) this.prediction.switchBody(box, near);
            }
        }
    }
    step(delta: number) {
        this.world.step(delta);
    }
}
export const ClientControls: Plugin = {
    name: "NetworkControls",
    recovery: "stateless",
    dependencies: GAME_PLUGINS,
    systems: [
        {
            name: "send-local-input",
            group: "setup",
            after: [LocalPlayerInputSystem],
            update(world) {
                const client = world.resource(ClientSession).client;
                if (!client || client.slot < 0) return;
                const { player } = scene(world).players[client.slot]!;
                if (!world.has(player, LocalPlayer)) world.add(player, LocalPlayer);
                const input = world.get(player, LocalPlayer);
                if (input)
                    client.record([
                        ...input.move,
                        input.viewYaw,
                        input.viewPitch,
                        input.sprint,
                        input.jumpPresses,
                    ]);
            },
        },
    ],
};
export const ClientSession = { create: (): { client?: GameClient } => ({}) };
