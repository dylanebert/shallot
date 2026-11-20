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
    RoomBot,
    RoomBots,
    RoomLeader,
    type ThinClient,
} from "@dylanebert/shallot/multiplayer";
import { Demo, Route } from "./demo";
import { PlayerMotion } from "./motion";
import { LocalPlayer, LocalPlayerInputSystem, Player, PlayerInput } from "./player";

export const ConnectionSlot = component("ConnectionSlot", { slot: u32 });
const PARTY = [ConnectionSlot, RoomLeader, RoomBot];
export const SCHEMA = [
    Transform,
    GlobalTransform,
    Body,
    Shape,
    Character,
    Player,
    PlayerInput,
    PlayerMotion,
    PredictionMotion,
    ...PARTY,
];
const PUBLIC = [Transform, GlobalTransform, Body, Shape, PredictionMotion];
const Connections: Plugin = { name: "GameConnections", components: PARTY };
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
const SLOTS = [0, 1, 2, 3] as const;

export function thinBot(seed: number, slot?: number): ThinClient {
    return {
        seed,
        admit(world, eid) {
            if (slot === undefined) return true;
            for (const peer of world.query([ConnectionSlot]))
                if (world.get(peer, ConnectionSlot)!.slot === slot) return false;
            world.add(eid, ConnectionSlot, { slot });
            return true;
        },
        input(_world, sender, tick) {
            const direction = Math.floor(tick / 90) % 2 ? 0.35 : -0.35;
            sender.record(tick + 1, "input", [direction, 0, 0, 0, 0, 0]);
        },
    };
}

const ASSIGNMENT = 100;
function assignment(world: World, slot: number) {
    let roster = 0;
    for (const peer of world.query([ConnectionSlot]))
        roster |= 1 << world.get(peer, ConnectionSlot)!.slot;
    const words = new Uint32Array([ASSIGNMENT, slot, roster]);
    return new Uint8Array(words.buffer);
}
export const GAME: Game = {
    plugins: GAME_PLUGINS,
    schema: SCHEMA,
    maxClients: SLOTS.length,
    reconnectMs: 10_000,
    onCreate(world) {
        for (const eid of world.query([Body])) world.add(eid, Replicate);
        for (const [slot, { player }] of scene(world).players.entries())
            world.storage(Replicate).rule.set(player, slot + 1);
    },
    onJoin(world, eid) {
        const occupied = [...world.query([ConnectionSlot])].map(
            (eid) => world.get(eid, ConnectionSlot)!.slot,
        );
        const slot =
            world.get(eid, ConnectionSlot)?.slot ?? SLOTS.find((slot) => !occupied.includes(slot))!;
        if (!world.has(eid, ConnectionSlot)) world.add(eid, ConnectionSlot, { slot });
        world.add(eid, Replicate, { rule: SLOTS.length + 1 });
        world.storage(Body).isEnabled.set(scene(world).players[slot]!.player, 1);
    },
    rules(world, eid) {
        const slot = world.get(eid, ConnectionSlot)!.slot;
        return [
            new Map([[eid, SCHEMA]]),
            ...SLOTS.map((owner) => new Map([[eid, owner === slot ? SCHEMA : PUBLIC]])),
            new Map([[eid, PARTY]]),
        ];
    },
    onLeave(world, eid) {
        const slot = world.get(eid, ConnectionSlot)?.slot;
        if (slot === undefined) return;
        const player = scene(world).players[slot]!.player;
        world.storage(Body).isEnabled.set(player, 0);
        world.remove(player, PlayerInput);
    },
    validateIntent(world, eid, record) {
        if (record.kind === "input") return validIntent(record);
        return (
            record.values.length === 2 &&
            (record.values[0] === 0 || record.values[0] === 1) &&
            SLOTS.some((slot) => slot === record.values[1]) &&
            !!world.resource(RoomBots).controller?.canManage(eid)
        );
    },
    applyIntent(world, eid, tick, records, input) {
        const bots = world.resource(RoomBots).controller;
        if (bots) {
            for (const command of records) {
                if (command.tick !== tick) continue;
                const [remove, slot] = command.values;
                if (!remove) {
                    void bots.add(eid, thinBot(tick, slot)).catch(console.error);
                } else {
                    const target = [...world.query([RoomBot, ConnectionSlot])].find(
                        (bot) => world.get(bot, ConnectionSlot)!.slot === slot,
                    );
                    if (target !== undefined) bots.remove(eid, target);
                }
            }
        }
        const slot = world.get(eid, ConnectionSlot)!.slot;
        applyIntent(world, scene(world).players[slot]!.player, tick, records, input);
    },
    onSend(world, eid, connection) {
        connection.send(assignment(world, world.get(eid, ConnectionSlot)!.slot));
    },
    onClient(world, connection) {
        const client = new GameClient(world, connection);
        world.resource(ClientSession).client = client;
    },
};
export class GameClient {
    slot = -1;
    roster = 0;
    readonly prediction: ClientPrediction;
    readonly intent: IntentSender;
    readonly connection: Connection;
    private _values = [0, 0, 0, 0, 0, 0];
    constructor(world: World, connection: Connection) {
        this.connection = connection;
        const routed: Connection = {
            endpoint: connection.endpoint,
            get generation() {
                return connection.generation;
            },
            get closed() {
                return connection.closed;
            },
            close: () => connection.close(),
            send: (bytes) => connection.send(bytes),
            receive: () =>
                connection.receive().filter((bytes) => {
                    if (bytes.length !== 12) return true;
                    const view = new DataView(bytes.buffer, bytes.byteOffset, bytes.byteLength);
                    if (view.getUint32(0, true) !== ASSIGNMENT) return true;
                    this.slot = view.getUint32(4, true);
                    this.roster = view.getUint32(8, true);
                    return false;
                }),
        };
        this.intent = new IntentSender(routed);
        // The lift's trajectory is tick-derived gameplay, so the rider predicts it too.
        world.add(scene(world).lift, PredictionTarget);
        this.prediction = new ClientPrediction(
            world,
            SCHEMA,
            routed,
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
    record(values: number[]) {
        this._values = values.slice();
    }
    requestBot(tick: number, slot: number, remove: boolean) {
        this.intent.record(tick, "operation", [Number(remove), slot]);
        this.intent.flush();
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
