import pkg from "../../../package.json" with { type: "json" };
import { createApp, type Plugin, Time, type World } from "../../engine";
import { entity, fields } from "../../engine/ecs";
import { MultiplayerPlugin, Replicate, type ReplicationReceiver } from ".";
import { transportClock } from "./diagnostics";
import type { Game } from "./game";
import { clientStartup, ServerSession, server } from "./game";
import type { InputRecord } from "./intent";
import type { Connection } from "./link";
import { type ClientPrediction, PredictionTarget } from "./prediction";
import { PredictionRecording } from "./recording-hooks";

export interface RecordingCall {
    kind: string;
    tick: number;
    args: unknown[];
    calls: RecordingCall[];
    result?: unknown;
}
/** Diagnosis only: code and incoming bytes are rerun, never restored as game state.
 * Record from World creation. Retains bytes and calls until finish; not a bounded history. */
export interface SessionRecording {
    engineVersion: string;
    gameVersion: string;
    /** Engine identity and build identifier; browsers expose their host build via userAgent. */
    javascript: { engine: string; version: string };
    role: "server" | "client";
    calls: RecordingCall[];
}
export interface RecordingOptions {
    gameVersion: string;
    /** Called once before disposal. The caller owns persistence, for example JSON serialization. */
    write(recording: SessionRecording): void;
}

const ActiveRecording = { create: (): { value?: SessionRecording } => ({}) };

function closeArgs(id: number, code?: number, reason?: string): unknown[] {
    if (reason !== undefined) return [id, code ?? null, reason];
    return code === undefined ? [id] : [id, code];
}

/** Borrows the opt-in recording accumulated so far; serialize it before the next step. */
export function sessionRecording(world: World): SessionRecording | undefined {
    return world.resource(ActiveRecording).value;
}

function javascriptRuntime(): SessionRecording["javascript"] {
    if (typeof Bun !== "undefined")
        return { engine: "JavaScriptCore", version: process.versions.webkit! };
    if (typeof process !== "undefined" && process.versions?.v8)
        return { engine: "V8", version: process.versions.v8 };
    const agent = navigator.userAgent;
    const engine = /Chrome\//.test(agent)
        ? "V8"
        : /Firefox\//.test(agent)
          ? "SpiderMonkey"
          : "JavaScriptCore";
    return { engine, version: agent };
}

/** Hashes schema fields in key/component/field order, including membership and entity keys.
 * Covers replication membership and local prediction, not presentation-only schema rows.
 * Each key is a run-independent identity for an eid, so an entity only one run creates
 * shifts no other row. Unkeyed rows, predicted entities without a wire ID, hash as an unordered
 * multiset, so no allocation order reaches the hash; two of them exchanging states hash equal.
 * Entity reference fields hash the referenced key, or a server's eid; on a client, a live
 * entity without a key hashes as one unkeyed token, distinct from 0 for none. */
function hashReplicatedRows(
    world: World,
    game: Game,
    keys: ReadonlyMap<number, number[]>,
    unkeyed: readonly number[],
    server: boolean,
): string {
    const bytes = new Uint8Array(8);
    const view = new DataView(bytes.buffer);
    let hash = 0;
    const number = (value: number) => {
        view.setFloat64(0, value, true);
        for (const byte of bytes) hash = Math.imul(hash ^ byte, 16777619) >>> 0;
    };
    const key = (eid: number) => {
        const values = keys.get(eid) ?? (server && eid ? [0, eid] : undefined);
        if (values) {
            number(values.length);
            for (const value of values) number(value);
        } else number(eid ? -1 : 0);
    };
    const row = (eid: number) => {
        for (const [index, component] of game.schema.entries()) {
            number(index);
            number(Number(world.has(eid, component)));
            if (!world.has(eid, component)) continue;
            const values = world.get(eid, component)! as Record<string, string | number | number[]>;
            for (const { name, field } of fields(component)) {
                const value = values[name]!;
                if (field === entity) key(value as number);
                else if (typeof value === "number") number(value);
                else if (typeof value === "string") {
                    number(value.length);
                    for (let i = 0; i < value.length; i++) number(value.charCodeAt(i));
                } else for (const item of value) number(item);
            }
        }
    };
    const order = (a: number[], b: number[]) => {
        for (let i = 0; i < Math.min(a.length, b.length); i++)
            if (a[i] !== b[i]) return a[i]! - b[i]!;
        return a.length - b.length;
    };
    const rows = unkeyed
        .map((eid) => {
            hash = 2166136261;
            row(eid);
            return hash;
        })
        .sort((a, b) => a - b);
    hash = 2166136261;
    for (const [eid, values] of [...keys].sort((a, b) => order(a[1], b[1]))) {
        number(values.length);
        for (const value of values) number(value);
        row(eid);
    }
    number(rows.length);
    for (const value of rows) number(value);
    return hash.toString(16).padStart(8, "0");
}

/** @internal Shared live recorder and playback verifier. Calls form a tree so nested game calls
 * are checked in place, while calls made by the external driver can be stepped independently. */
export class SessionRecorder {
    readonly recording: SessionRecording;
    private _world!: World;
    private _active = true;
    private _prediction?: ClientPrediction;
    private readonly _receivers: ReplicationReceiver[] = [];
    private _nextConnection = 0;
    private readonly _connections = new Map<number, Connection>();
    private readonly _identities = new WeakMap<Connection, number>();
    private readonly _wrapped = new WeakMap<Connection, Connection>();
    private _stack: { calls: RecordingCall[]; cursor: number }[];
    private readonly _game: Game;
    private readonly _options: RecordingOptions;
    private readonly _playback?: SessionRecording;
    constructor(
        game: Game,
        role: SessionRecording["role"],
        options: RecordingOptions,
        playback?: SessionRecording,
    ) {
        this._game = game;
        this._options = options;
        this._playback = playback;
        this.recording = playback ?? {
            engineVersion: pkg.version,
            gameVersion: options.gameVersion,
            javascript: javascriptRuntime(),
            role,
            calls: [],
        };
        this._stack = [{ calls: this.recording.calls, cursor: 0 }];
    }
    private fail(message: string, tick = this._world.time.fixedTick): never {
        throw new Error(`Replay differs at fixed tick ${tick}: ${message}`);
    }
    private invoke<T>(kind: string, args: unknown[], run: () => T, substitute = false): T {
        if (!this._active) return run();
        const parent = this._stack.at(-1)!;
        let call: RecordingCall;
        if (this._playback) {
            const expected = parent.calls[parent.cursor++];
            if (
                !expected ||
                expected.kind !== kind ||
                expected.tick !== this._world.time.fixedTick ||
                JSON.stringify(expected.args) !== JSON.stringify(args)
            )
                this.fail(
                    `expected ${expected?.kind ?? "end of calls"}(${JSON.stringify(expected?.args)}) at ${expected?.tick}, got ${kind}(${JSON.stringify(args)})`,
                    Math.min(
                        expected?.tick ?? this._world.time.fixedTick,
                        this._world.time.fixedTick,
                    ),
                );
            call = expected;
        } else {
            call = {
                kind,
                tick: this._world.time.fixedTick,
                args: structuredClone(args),
                calls: [],
            };
            parent.calls.push(call);
        }
        const frame = { calls: call.calls, cursor: 0 };
        this._stack.push(frame);
        try {
            const value = substitute && this._playback ? (call.result as T) : run();
            if (this._playback) {
                if (frame.cursor !== frame.calls.length)
                    this.fail(
                        `missing ${frame.calls[frame.cursor]!.kind}`,
                        frame.calls[frame.cursor]!.tick,
                    );
                if (
                    !substitute &&
                    JSON.stringify(value ?? null) !== JSON.stringify(call.result ?? null)
                )
                    this.fail(`${kind} result differs`);
            } else call.result = structuredClone(value ?? null);
            return value;
        } finally {
            this._stack.pop();
        }
    }
    readonly plugin: Plugin = {
        name: "SessionRecording",
        recovery: "stateless",
        initialize: (world) => {
            this._world = world;
            world.resource(ActiveRecording).value = this.recording;
            world.onDispose(() => this.finish());
            const tick = world.tick.bind(world);
            const step = world.step.bind(world);
            world.tick = () => this.invoke("tick", [], tick);
            world.step = (delta = Time.DEFAULT_DT) =>
                this.invoke("step", [delta], () => step(delta));
            for (const kind of [
                "pause",
                "resume",
                "setTimeScale",
                "discardOverstep",
                "jumpToTick",
            ] as const) {
                const control = world[kind].bind(world);
                world[kind] = (value?: number) =>
                    this.invoke(kind, value === undefined ? [] : [value], () => control(value!));
            }
            const hooks = world.resource(PredictionRecording);
            hooks.receiver = (receiver) => this._receivers.push(receiver);
            hooks.attach = (prediction) => {
                this._prediction = prediction;
                const accept = prediction.accept.bind(prediction);
                const pace = prediction.pace.bind(prediction);
                const switchBody = prediction.switchBody.bind(prediction);
                prediction.accept = (records, watermark = 0) => {
                    const values = [...records];
                    this.invoke("accept", [values, watermark], () => accept(values, watermark));
                };
                prediction.pace = (tick, age) =>
                    this.invoke("pace", [tick, age], () => pace(tick, age));
                prediction.switchBody = (eid, enabled) =>
                    this.invoke("switchBody", [eid, enabled], () => switchBody(eid, enabled));
            };
            hooks.sample = (prediction, tick) => {
                const sample = () => prediction.sampleInput(tick);
                // Run the game callback, checking its nested prediction/receive calls, but replace
                // the device sample's return value with the recorded tick-addressed input.
                if (!this._playback) return this.invoke("sample", [tick], sample);
                const expected = this._stack.at(-1)!.calls[this._stack.at(-1)!.cursor];
                this.invoke("sample", [tick], () => {
                    sample();
                    return expected?.result;
                });
                return (expected?.result ?? undefined) as InputRecord | undefined;
            };
        },
        systems: [
            {
                name: "hash-recorded-tick",
                group: "postFixed",
                boundary: "after",
                update: () =>
                    this.invoke("hash", [], () => {
                        const world = this._world;
                        const { keys, unkeyed } = this.keys();
                        return hashReplicatedRows(
                            world,
                            this._game,
                            keys,
                            unkeyed,
                            this.recording.role === "server",
                        );
                    }),
            },
        ],
    };
    /** A server's eids are deterministic in its replay, since replay composes the same server.
     * A client keys replicated entities by receiver and wire ID; predicted entities without
     * one stay unkeyed, so neither depends on what else the tab allocated. */
    private keys(): { keys: Map<number, number[]>; unkeyed: number[] } {
        const world = this._world;
        const keys = new Map<number, number[]>();
        if (this.recording.role === "server") {
            for (const eid of world.query([Replicate])) keys.set(eid, [0, eid]);
            return { keys, unkeyed: [] };
        }
        this._receivers.forEach((receiver, index) => {
            for (const [id, eid] of receiver.entities) keys.set(eid, [1, index, id]);
        });
        const unkeyed = [...world.query([PredictionTarget])].filter(
            (eid) => !keys.has(eid) && this._prediction?.predicted(eid),
        );
        return { keys, unkeyed };
    }
    connection(connection: Connection): Connection {
        const existing = this._wrapped.get(connection);
        if (existing) return existing;
        const id = this._nextConnection++;
        const clock =
            (connection as Connection & { [transportClock]?: () => number })[transportClock] ??
            (() => performance.now());
        let sending = 0;
        let closing = false;
        let close: Connection["close"] = (code, reason) => connection.close(code, reason);
        let lastClosed: boolean | undefined;
        let send = (bytes: Uint8Array) => connection.send(bytes);
        const playback = !!this._playback;
        const thisRecorder = this;
        const wrapped: Connection & { [transportClock]: () => number } = {
            endpoint: playback ? undefined : connection.endpoint,
            onError: connection.onError && ((error) => connection.onError?.(error)),
            get generation() {
                if (playback) {
                    const parent = thisRecorder._stack.at(-1)!;
                    if (parent.calls[parent.cursor]?.kind !== "generation") return undefined;
                } else if (connection.generation === undefined) return undefined;
                return thisRecorder.invoke("generation", [id], () => connection.generation, true);
            },
            [transportClock]: () => (playback ? this._world.time.realElapsed * 1000 : clock()),
            get closed() {
                if (sending) return playback ? lastClosed : connection.closed;
                lastClosed = status();
                return lastClosed;
            },
            get close() {
                const current = close;
                return (code?: number, reason?: string) => {
                    if (closing) return current(code, reason);
                    closing = true;
                    try {
                        thisRecorder.invoke("close", closeArgs(id, code, reason), () =>
                            current(code, reason),
                        );
                    } finally {
                        closing = false;
                    }
                };
            },
            set close(value) {
                close = value;
            },
            // Diagnostic send-side status polls are not incoming observations: replay does not
            // reproduce the external sender's queues or transmit bytes to another World.
            get send() {
                const current = send;
                return (bytes: Uint8Array) => {
                    sending++;
                    try {
                        current(bytes);
                    } finally {
                        sending--;
                    }
                };
            },
            set send(value) {
                send = value;
            },
            receive: () => {
                let messages: Uint8Array[] = [];
                const result = this.invoke(
                    "receive",
                    [id],
                    () => {
                        messages = connection.receive();
                        return messages.map((bytes) => [...bytes]);
                    },
                    true,
                );
                return playback ? result.map((bytes) => new Uint8Array(bytes)) : messages;
            },
        };
        const status = () =>
            this.invoke("closed", [id], () => connection.closed ?? null, true) ?? undefined;
        this._wrapped.set(connection, wrapped);
        this._wrapped.set(wrapped, wrapped);
        this._identities.set(wrapped, id);
        this._connections.set(id, wrapped);
        return wrapped;
    }
    attachServer(): void {
        const authority = this._world.resource(ServerSession).server!;
        const connect = authority.connect.bind(authority);
        const disconnect = authority.disconnect.bind(authority);
        const hold = authority.hold.bind(authority);
        const resume = authority.resume.bind(authority);
        // The recording addresses the reconstructed entity, not a process-local ClientId.
        authority.hold = (id) => {
            const eid = authority.entity(id);
            if (eid !== undefined) this.invoke("holdSeat", [eid], () => hold(id));
        };
        authority.resume = (id) => {
            const eid = authority.entity(id);
            if (eid === undefined) return resume(id);
            this.invoke("resumeSeat", [eid], () => resume(id));
        };
        authority.connect = (connection, kind = "human") => {
            const wrapped = this._identities.has(connection)
                ? connection
                : this.connection(connection);
            const id = this._identities.get(wrapped)!;
            return this.invoke("connect", kind === "bot" ? [id, kind] : [id], () =>
                connect(wrapped, kind),
            );
        };
        authority.disconnect = (id, code, reason) => {
            const eid = authority.entity(id);
            if (eid !== undefined)
                this.invoke("disconnect", closeArgs(eid, code, reason), () =>
                    disconnect(id, code, reason),
                );
        };
    }
    finish(): void {
        if (!this._active) return;
        this._active = false;
        if (!this._playback) this._options.write(this.recording);
    }
    run(): void {
        const root = this._stack[0]!;
        const empty: Connection = { close() {}, send() {}, receive: () => [] };
        while (root.cursor < root.calls.length) {
            const call = root.calls[root.cursor]!;
            const args = call.args;
            switch (call.kind) {
                case "step":
                    this._world.step(args[0] as number);
                    break;
                case "tick":
                    this._world.tick();
                    break;
                case "pause":
                case "resume":
                case "setTimeScale":
                case "discardOverstep":
                case "jumpToTick":
                    this._world[call.kind](args[0] as number);
                    break;
                case "connect": {
                    const authority = this._world.resource(ServerSession).server!;
                    authority.connect(
                        this.connection({ ...empty }),
                        args[1] === "bot" ? "bot" : "human",
                    );
                    break;
                }
                case "disconnect":
                case "holdSeat":
                case "resumeSeat": {
                    const authority = this._world.resource(ServerSession).server!;
                    const id = authority.peers.get(args[0] as number)?.clientId;
                    if (id === undefined) throw new Error("Recorded seat is missing");
                    if (call.kind === "disconnect")
                        authority.disconnect(
                            id,
                            (args[1] ?? undefined) as number | undefined,
                            args[2] as string | undefined,
                        );
                    else if (call.kind === "holdSeat") authority.hold(id);
                    else authority.resume(id);
                    break;
                }
                case "generation":
                    void this._connections.get(args[0] as number)!.generation;
                    break;
                case "accept":
                    this._prediction!.accept(
                        args[0] as Parameters<ClientPrediction["accept"]>[0],
                        args[1] as number,
                    );
                    break;
                case "switchBody":
                    this._prediction!.switchBody(args[0] as number, args[1] as boolean);
                    break;
                case "pace":
                    this._prediction!.pace(args[0] as number, args[1] as number);
                    break;
                case "closed":
                    void this._connections.get(args[0] as number)!.closed;
                    break;
                case "receive":
                    this._connections.get(args[0] as number)!.receive();
                    break;
                case "close":
                    this._connections
                        .get(args[0] as number)!
                        .close(
                            (args[1] ?? undefined) as number | undefined,
                            args[2] as string | undefined,
                        );
                    break;
                default:
                    this.fail(`unstepped ${call.kind}`);
            }
        }
    }
}

/** Rebuilds the same game's role without a socket or loading recorded state.
 * Refuses either version mismatch before building the World; throws at the first differing tick
 * or call. Game plugins and startup hooks run normally, including decoding and validation. */
export async function replaySession(
    game: Game,
    gameVersion: string,
    recording: SessionRecording,
): Promise<void> {
    if (recording.engineVersion !== pkg.version || recording.gameVersion !== gameVersion)
        throw new Error("Replay version mismatch");
    const recorder = new SessionRecorder(
        game,
        recording.role,
        { gameVersion, write() {} },
        recording,
    );
    const connection =
        recording.role === "client"
            ? recorder.connection({ close() {}, send() {}, receive: () => [] })
            : undefined;
    // A server has no pre-existing client endpoint; numbering starts with its first admission.
    const plugins: Plugin[] =
        recording.role === "server"
            ? [recorder.plugin, server(game)]
            : [
                  recorder.plugin,
                  {
                      name: "ReplayClient",
                      dependencies: [...game.plugins, MultiplayerPlugin],
                      systems: [clientStartup(game, () => connection!)],
                  },
              ];
    const app = await createApp({ defaults: false, plugins });
    try {
        if (recording.role === "server") recorder.attachServer();
        recorder.run();
    } finally {
        recorder.finish();
        app.dispose();
    }
}
