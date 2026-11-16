import {
    Body,
    Character,
    createApp,
    GlobalTransform,
    type Plugin,
    Shape,
    Transform,
    type World,
} from "@dylanebert/shallot";
import {
    ClientPrediction,
    type Connection,
    IntentReceiver,
    type IntentRecord,
    IntentSender,
    MultiplayerPlugin,
    PredictionMotion,
    PredictionPlugin,
    PredictionTarget,
    Replicate,
    ReplicationSender,
} from "@dylanebert/shallot/multiplayer";
import { Demo, Route } from "./demo";
import { PlayerMotion } from "./motion";
import { LocalPlayer, LocalPlayerInputSystem, Player, PlayerInput } from "./player";

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
];
const PUBLIC = [Transform, GlobalTransform, Body, Shape, PredictionMotion];
export const GAME_PLUGINS = [Demo, PredictionPlugin, MultiplayerPlugin];
export async function openGame() {
    return createApp({ defaults: false, plugins: GAME_PLUGINS });
}
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
    records: readonly IntentRecord[],
) {
    let latest: IntentRecord | undefined;
    for (const record of records) {
        if (record.tick > tick || !validIntent(record)) continue;
        if (
            !latest ||
            record.tick > latest.tick ||
            (record.tick === latest.tick && record.id > latest.id)
        )
            latest = record;
    }
    if (!latest) return;
    if (!world.has(player, PlayerInput)) world.add(player, PlayerInput);
    const input = world.storage(PlayerInput),
        v = latest.values;
    input.move.set(player, v[0]!, v[1]!);
    input.yaw.set(player, v[2]!);
    input.pitch.set(player, v[3]!);
    input.sprint.set(player, v[4]!);
    input.jumpPresses.set(player, v[5]!);
}
const ASSIGNMENT = 100;
function assignment(slot: number) {
    const words = new Uint32Array([ASSIGNMENT, slot]);
    return new Uint8Array(words.buffer);
}
export class GameServer {
    readonly peers = new Map<number, { sender: ReplicationSender; intent: IntentReceiver }>();
    readonly world: World;
    constructor(world: World) {
        this.world = world;
        for (const eid of world.query([Body])) world.add(eid, Replicate);
        for (const [slot, { player }] of scene(world).players.entries()) {
            world.storage(Replicate).rule.set(player, slot + 1);
        }
    }
    connect(connection: Connection) {
        const slot = [0, 1].find((slot) => !this.peers.has(slot));
        if (slot === undefined) throw new Error("Two players already connected");
        const rules = [
            new Map([[slot, SCHEMA]]),
            ...[0, 1].map((owner) => new Map([[slot, owner === slot ? SCHEMA : PUBLIC]])),
        ];
        const sender = new ReplicationSender(this.world, SCHEMA, rules, slot, connection);
        const peer = { sender, intent: new IntentReceiver(connection) };
        this.peers.set(slot, peer);
        connection.send(assignment(slot));
        sender.record(this.world.time.fixedTick);
        sender.flush();
        return slot;
    }
    disconnect(slot: number) {
        this.peers.delete(slot);
        const player = scene(this.world).players[slot]!.player;
        if (this.world.has(player, PlayerInput))
            this.world.storage(PlayerInput).move.set(player, 0, 0);
    }
    tick() {
        const tick = this.world.time.fixedTick + 1;
        for (const [slot, peer] of this.peers) {
            peer.intent.receive(peer.sender.flush(), tick);
            applyIntent(this.world, scene(this.world).players[slot]!.player, tick, [
                ...peer.intent.records.values(),
            ]);
        }
        this.world.tick();
        for (const [slot, peer] of this.peers) {
            peer.sender.connection.send(assignment(slot));
            peer.sender.record(tick, [...peer.intent.records.values()]);
            peer.sender.flush();
        }
    }
}
export class GameClient {
    slot = -1;
    readonly prediction: ClientPrediction;
    readonly intent: IntentSender;
    readonly world: World;
    constructor(world: World, connection: Connection) {
        this.world = world;
        const routed: Connection = {
            send: (bytes) => connection.send(bytes),
            receive: () =>
                connection.receive().filter((bytes) => {
                    if (bytes.length !== 8) return true;
                    const view = new DataView(bytes.buffer, bytes.byteOffset, bytes.byteLength);
                    if (view.getUint32(0, true) !== ASSIGNMENT) return true;
                    this.slot = view.getUint32(4, true);
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
            (_world, tick, records) => {
                if (this.slot >= 0)
                    applyIntent(world, scene(world).players[this.slot]!.player, tick, records);
            },
        );
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
        const record = this.intent.record(this.world.time.fixedTick + 1, "input", values);
        this.prediction.accept([record]);
        this.intent.flush();
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
        this.selectPrediction();
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
                client.selectPrediction();
                const { player } = scene(world).players[client.slot]!;
                if (!world.has(player, LocalPlayer)) world.add(player, LocalPlayer);
                const input = world.get(player, PlayerInput);
                if (input)
                    client.record([
                        ...input.move,
                        input.yaw,
                        input.pitch,
                        input.sprint,
                        input.jumpPresses,
                    ]);
            },
        },
    ],
};
export const ClientSession = { create: (): { client?: GameClient } => ({}) };
