import {
    type App,
    component,
    DiagnosticsStore,
    type Plugin,
    type System,
    Time,
    type World,
} from "../../engine";
import {
    MultiplayerPlugin,
    ReplicationMessageKind,
    type ReplicationRules,
    type ReplicationSchema,
    ReplicationSender,
} from ".";
import { diagnosticKind, observeNetwork, transportClock } from "./diagnostics";
import {
    decodeInputs,
    decodeIntent,
    type InputRecord,
    IntentMessageKind,
    IntentReceiver,
    type IntentRecord,
    type IntentSender,
    type OperationRecord,
} from "./intent";
import { type Connection, type LinkFaults, seededLink } from "./link";
import { type RecordingOptions, SessionRecorder, type SessionRecording } from "./recording";
import { createServerApp, roomHost } from "./room-host";
import { LEAVE_CLOSE_CODE, WebSocketConnection } from "./websocket";

/** A live server-side connection entity. Its transport and histories live outside ECS storage. */
export const ClientOf = component("ClientOf", {});

/** Server-owned marker on the leading seat. Include it in the schema and disclose it on
 * seat entities through Replicate to let clients observe the leader. */
export const RoomLeader = component("RoomLeader", {});

/** Server-owned kind marker. Disclose on seat entities through Replicate. */
export const RoomBot = component("RoomBot", {});

/** Shared protocol and game policy. Connection arguments are server eids, not client-supplied IDs.
 * Hooks borrow runtime handles; explicit disconnect calls onLeave before despawning.
 * Direct ECS removal may already have removed the entity's other components.
 * onClient installs game-specific prediction or routing at the client's first stepped preFixed
 * boundary, after app warm-up, where prediction may capture its initial World snapshot.
 * Replay composes only the game's plugins and onClient, not the tab's controls or presentation;
 * a client's hash keys entities by wire ID, so entities only the tab creates change no tick hash.
 * Predicted entities without a wire ID hash as an unordered set, and a reference to one only as
 * present. */
export interface Game {
    plugins: readonly Plugin[];
    schema: ReplicationSchema;
    maxClients: number;
    /** Milliseconds to hold a disconnected seat. Omit for immediate leave. */
    reconnectMs?: number;
    rules(world: World, connection: number): ReplicationRules;
    onCreate?(world: World): void;
    onJoin?(world: World, connection: number, transport: Connection): void;
    onLeave?(world: World, connection: number): void;
    validateIntent(world: World, connection: number, record: IntentRecord): boolean;
    /** Borrows accepted operations and this tick's held input before gameplay.
     * Input keeps its requested tick, which may precede tick after loss or late arrival. */
    applyIntent(
        world: World,
        connection: number,
        tick: number,
        records: readonly OperationRecord[],
        input?: InputRecord,
    ): void;
    onSend?(world: World, connection: number, transport: Connection): void;
    onClient?(world: World, transport: Connection): void;
}

interface Peer {
    diagnostics: ReturnType<typeof observeNetwork>;
    connection: Connection;
    sender: ReplicationSender;
    intent: IntentReceiver;
    held?: boolean;
    /** Tick of admission or the latest resume. */
    since: number;
    kind: "human" | "bot";
}

/** Server runtime keyed by connection entity. Removing ClientOf closes the transport and releases
 * queued bytes, its runtime handle and histories. World disposal also closes every transport.
 * Use the server role plugin; tick the World normally, including through World.step. */
class GameServer {
    private readonly _peers = new Map<number, Peer>();
    get peers(): ReadonlyMap<number, Peer> {
        return this._peers;
    }
    /** Earliest admitted human seat, including seats held for rejoin. */
    get leader(): number | undefined {
        for (const [eid, peer] of this._peers) if (peer.kind === "human") return eid;
        return undefined;
    }
    readonly world: World;
    readonly game: Game;
    private readonly _diagnostics: DiagnosticsStore;
    constructor(world: World, game: Game) {
        this.world = world;
        this.game = game;
        if (!Number.isSafeInteger(game.maxClients) || game.maxClients < 1)
            throw new Error("maxClients must be a positive integer");
        const diagnostics = world.resource(DiagnosticsStore);
        this._diagnostics = diagnostics;
        const unobserve = world.observeMembership(ClientOf, (eid, present) => {
            if (!present) this.disconnect(eid);
        });
        world.onDispose(() => {
            unobserve();
            for (const [eid, peer] of this._peers) {
                this._peers.delete(eid);
                peer.diagnostics.remove();
                peer.connection.close();
            }
            diagnostics.record("network/peer_count", 0);
        });
        diagnostics.record("network/peer_count", 0);
        game.onCreate?.(world);
    }
    /** Refuses admission at the peer limit without spawning an entity or calling onJoin. */
    connect(
        connection: Connection,
        kind: Peer["kind"] = "human",
        admit?: ThinClient["admit"],
    ): number | undefined {
        if (this.peers.size >= this.game.maxClients) return undefined;
        const eid = this.world.create();
        this.world.add(eid, ClientOf);
        const diagnostics = observeNetwork(this.world, connection, `network/connections/${eid}/`);
        try {
            if (kind === "bot" && this.game.schema.includes(RoomBot)) this.world.add(eid, RoomBot);
            if (admit && !admit(this.world, eid)) {
                diagnostics.remove();
                this.world.destroy(eid);
                return undefined;
            }
            this.game.onJoin?.(this.world, eid, connection);
            const sender = new ReplicationSender(
                this.world,
                this.game.schema,
                this.game.rules(this.world, eid),
                eid,
                connection,
            );
            this._peers.set(eid, {
                connection,
                sender,
                intent: new IntentReceiver(connection),
                diagnostics,
                since: this.world.time.fixedTick,
                kind,
            });
            if (this.game.schema.includes(RoomLeader) && this.leader === eid)
                this.world.add(eid, RoomLeader);
            this._diagnostics.record("network/peer_count", this._peers.size);
            sender.record(this.world.time.fixedTick);
            sender.flush();
            return eid;
        } catch (error) {
            diagnostics.remove();
            try {
                this.disconnect(eid);
            } finally {
                connection.close();
                this.world.destroy(eid);
            }
            throw error;
        }
    }
    /** Idempotent. Explicit leave runs onLeave before despawning; transport closure and history
     * release still run if the hook throws. */
    disconnect(eid: number): void {
        const peer = this._peers.get(eid);
        if (!peer) return;
        const wasLeader = this.leader === eid;
        this._peers.delete(eid);
        peer.diagnostics.remove();
        this._diagnostics.record("network/peer_count", this._peers.size);
        try {
            if (wasLeader && this.game.schema.includes(RoomLeader)) {
                if (this.world.has(eid, RoomLeader)) this.world.remove(eid, RoomLeader);
                const successor = this.leader;
                if (successor !== undefined) this.world.add(successor, RoomLeader);
            }
            this.game.onLeave?.(this.world, eid);
        } finally {
            peer.connection.close();
            this.world.destroy(eid);
        }
    }
    /** @internal Keep the entity and intent history, but stop stepping this peer. */
    hold(eid: number): void {
        const peer = this._peers.get(eid);
        if (peer) peer.held = true;
    }
    /** @internal Start a fresh replication history for a held peer. */
    resume(eid: number): void {
        const peer = this._peers.get(eid);
        if (!peer?.held) throw new Error("Seat is not held");
        peer.sender = new ReplicationSender(
            this.world,
            this.game.schema,
            this.game.rules(this.world, eid),
            eid,
            peer.connection,
        );
        peer.since = this.world.time.fixedTick;
        peer.sender.record(peer.since);
        peer.sender.flush();
        peer.held = false;
    }
    /** @internal */
    receive(): void {
        const tick = this.world.time.fixedTick;
        for (const [eid, peer] of this.peers) {
            if (peer.held) continue;
            if (peer.connection.closed) {
                this.disconnect(eid);
                continue;
            }
            const messages = peer.sender.flush();
            peer.intent.prune(peer.sender.acknowledged);
            peer.intent.receive(messages, tick, (record) =>
                this.game.validateIntent(this.world, eid, record),
            );
            const input = peer.intent.inputAt(tick);
            this.game.applyIntent(
                this.world,
                eid,
                tick,
                [...peer.intent.records.values()],
                input ? { ...input, kind: "input" } : undefined,
            );
        }
    }
    /** @internal */
    send(): void {
        for (const [eid, peer] of this.peers) {
            if (peer.held) continue;
            this.game.onSend?.(this.world, eid, peer.connection);
            peer.sender.record(
                this.world.time.fixedTick,
                [...peer.intent.records.values()],
                peer.intent.commandAge(this.world.time.fixedTick),
            );
            peer.sender.flush();
        }
    }
}

/** @internal Whether GameServer.receive consumes these client bytes without throwing: a ping or
 * pong, an acknowledgement, operation records or tick inputs. */
export function isClientMessage(bytes: Uint8Array): boolean {
    if (diagnosticKind(bytes)) return true;
    if (bytes.length < 4) return false;
    const kind = new DataView(bytes.buffer, bytes.byteOffset, bytes.byteLength).getUint32(0, true);
    try {
        if (kind === ReplicationMessageKind.Acknowledgement) return bytes.length === 12;
        if (kind === IntentMessageKind.Inputs) decodeInputs(bytes);
        else if (kind === IntentMessageKind.Records) decodeIntent(bytes);
        else return false;
        return true;
    } catch {
        return false;
    }
}

/** The server role's runtime; absent in client-only Worlds. */
export const ServerSession = { create: (): { server?: GameServer } => ({}) };

/** Composes authority into an app. The caller owns the app, its clock and its disposal. */
export function server(game: Game): Plugin {
    return {
        name: "MultiplayerServer",
        dependencies: [...game.plugins, MultiplayerPlugin],
        components: [ClientOf, RoomLeader, RoomBot],
        recovery: "stateless",
        initialize(world) {
            world.resource(ServerSession).server = new GameServer(world, game);
        },
        dispose(world) {
            if (world.signal.aborted) return;
            const runtime = world.resource(ServerSession).server;
            if (runtime) for (const eid of runtime.peers.keys()) runtime.disconnect(eid);
        },
        systems: [
            {
                name: "receive-client-intents",
                group: "fixed",
                first: true,
                update(world) {
                    world.resource(ServerSession).server!.receive();
                },
            },
            {
                name: "send-server-state",
                group: "postFixed",
                last: true,
                update(world) {
                    world.resource(ServerSession).server!.send();
                },
            },
        ],
    };
}

/** @internal Shared first-boundary startup for composed client roles. */
export function clientStartup(game: Game, connection: (world: World) => Connection): System {
    return {
        name: "start-game-client",
        group: "preFixed",
        first: true,
        setup(world) {
            observeNetwork(world, connection(world));
            game.onClient?.(world, connection(world));
        },
        update(world) {
            observeNetwork(world, connection(world)).poll();
        },
    };
}

/** Connects a caller-owned client app. Initialization awaits the socket; disposal closes it.
 * An admitted reconnectable seat retries on close, from 50 ms up to 1 second between attempts.
 * Initial failure or a refused token is reported to onError. */
export function client(
    game: Game,
    options: {
        url: string;
        onError?: (error: Error) => void;
        /** Receives the server-selected room code on admission to serveRooms. */
        onRoom?: (code: string) => void;
        recording?: RecordingOptions;
        /** Schedule a retry in milliseconds, returning its cancellation function. */
        retryAfter?: (delay: number, retry: () => void) => () => void;
    },
): Plugin {
    const recorder = options.recording && new SessionRecorder(game, "client", options.recording);
    const Transport = { create: (): { connection?: Connection } => ({}) };
    return {
        name: "MultiplayerClient",
        dependencies: [...(recorder ? [recorder.plugin] : []), ...game.plugins],
        async initialize(world) {
            let url = options.url;
            if (options.onRoom) {
                const endpoint = new URL(url, globalThis.location?.href);
                endpoint.searchParams.set("announceRoom", "1");
                url = endpoint.href;
            }
            const socket = new WebSocket(url);
            let seat: string | undefined;
            let token: string | undefined;
            let generation = 0;
            let delay = 50;
            let cancel: (() => void) | undefined;
            let disposed = false;
            let failed = false;
            const failure = () =>
                new Error(`No server answered at ${options.url}. The connection failed or closed.`);
            const refusal = (code: number, reason: string) =>
                code === 1006
                    ? failure()
                    : new Error(
                          `${options.url} closed the connection (${code}${reason ? `: ${reason}` : ""}).`,
                      );
            const schedule =
                options.retryAfter ??
                ((ms: number, retry: () => void) => {
                    const timer = setTimeout(retry, ms);
                    return () => clearTimeout(timer);
                });
            function attach(socket: WebSocket) {
                socket.addEventListener("close", ({ code, reason }) => {
                    if (disposed || world.signal.aborted) return;
                    if (!token || code === 1008) {
                        failed = true;
                        options.onError?.(refusal(code, reason));
                        return;
                    }
                    cancel = schedule(delay, () => {
                        if (disposed) return;
                        const url = new URL(socket.url);
                        url.searchParams.delete("creator");
                        url.searchParams.set("resume", token!);
                        url.searchParams.set("seat", seat!);
                        current = attach(new WebSocket(url));
                        generation++;
                    });
                    delay = Math.min(1000, delay * 2);
                });
                return new WebSocketConnection(socket, (message) => {
                    const room = options.onRoom && /^room:([0-9A-HJKMNP-TV-Z]{10})$/.exec(message);
                    if (room) {
                        options.onRoom?.(room[1]!);
                        return true;
                    }
                    if (!game.reconnectMs) return false;
                    const match = /^resume:([0-9a-f-]{36}):([0-9a-f-]{36})$/.exec(message);
                    if (!match) return false;
                    [, seat, token] = match;
                    delay = 50;
                    return true;
                });
            }
            let current = attach(socket);
            const connection: Connection = {
                endpoint: socket.url,
                get generation() {
                    return generation || undefined;
                },
                get closed() {
                    return disposed || failed || (!token && current.closed);
                },
                close() {
                    disposed = true;
                    cancel?.();
                    current.close(game.reconnectMs ? LEAVE_CLOSE_CODE : undefined);
                },
                send(bytes) {
                    if (!token || (current.socket.readyState === 1 && !current.closed))
                        current.send(bytes);
                },
                receive() {
                    return current.receive();
                },
            };
            world.onDispose(() => connection.close());
            await new Promise<void>((resolve, reject) => {
                socket.addEventListener("open", () => resolve(), { once: true });
                socket.addEventListener("error", () => reject(failure()), { once: true });
                socket.addEventListener("close", () => reject(failure()), { once: true });
            });
            world.resource(Transport).connection = recorder?.connection(connection) ?? connection;
        },
        systems: [clientStartup(game, (world) => world.resource(Transport).connection!)],
    };
}

interface LocalOptions {
    seed: number;
    faults: LinkFaults;
}

export interface ThinClient {
    seed: number;
    /** Server-side admission before onJoin; may set game-owned seat data or refuse admission. */
    admit?(world: World, eid: number): boolean;
    /** IntentSender.record throws on invalid text or a record that cannot fit one upstream
     * packet, and the throw leaves the room's tick. */
    input(world: World, sender: IntentSender, tick: number): void;
}

function localHost(game: Game, options: LocalOptions, pinned?: App, recorder?: SessionRecorder) {
    type Endpoint = Connection & { credentials?: { id: string; token: string } };
    let clock = 0;
    let accumulator = 0;
    let seed = options.seed;
    const links = new Set<ReturnType<typeof seededLink>>();
    const host = roomHost<Endpoint>(
        game,
        {
            reservationNow: () => clock,
            reconnectNow: () => clock,
            botFaults: options.faults,
            sendToken(connection, id, token) {
                connection.credentials = { id, token };
            },
        },
        pinned,
        recorder,
    );
    function stream(seed: number) {
        const link = seededLink(seed, options.faults);
        link.advance(clock);
        links.add(link);
        const endpoint: Endpoint = link.a;
        const close = endpoint.close;
        endpoint.close = () => {
            links.delete(link);
            close();
        };
        return { endpoint, client: link.b };
    }
    async function join(code?: string, creator?: string, clientRecorder?: SessionRecorder) {
        const clientSeed = seed++;
        let current = stream(clientSeed);
        const result = host.connect(current.endpoint, {
            intent: code === undefined ? "joinOrCreate" : "join",
            code,
            creator,
        });
        await result?.ready;
        if (!result?.room.app || current.endpoint.closed) throw new Error("Local client refused");
        const room = result.room.code;
        let cut = false;
        let generation = 0;
        let left = false;
        const leave = () => {
            if (left) return;
            left = true;
            try {
                host.leave(current.endpoint, false);
            } finally {
                current.endpoint.close();
            }
        };
        const connection: Connection & { [transportClock]: () => number } = {
            [transportClock]: () => clock,
            get generation() {
                return generation || undefined;
            },
            get closed() {
                return disposed || left || (!cut && !!current.client.closed);
            },
            close: leave,
            send(bytes) {
                current.client.send(bytes);
            },
            receive() {
                return current.client.receive();
            },
        };
        const plugin: Plugin = {
            name: "MultiplayerLocalClient",
            dependencies: [...(clientRecorder ? [clientRecorder.plugin] : []), ...game.plugins],
            initialize(world) {
                world.onDispose(pinned ? dispose : leave);
            },
            systems: [
                clientStartup(game, () => clientRecorder?.connection(connection) ?? connection),
            ],
        };
        return {
            code: room,
            plugin,
            leave,
            /** Adds a thin client to this room. Its handle removes the bot and frees its seat. */
            addBot: async (thin: ThinClient) => {
                if (left || current.endpoint.closed)
                    throw new Error("Local client is disconnected");
                const bot = await host.addBot(result.room, thin);
                if (!bot) throw new Error("Thin client refused");
                return bot;
            },
            /** Drops this stream now. The room's reconnect window runs on pumped time. */
            cut() {
                if (left || current.endpoint.closed) return;
                cut = !!current.endpoint.credentials;
                current.endpoint.close();
                host.leave(current.endpoint);
            },
            /** Starts a fresh stream on the same seat after cut() or a host close, or returns
             * false if it cannot resume. */
            resume() {
                const credentials = current.endpoint.credentials;
                if (!(cut || current.endpoint.closed) || left || !credentials) return false;
                current = stream(clientSeed);
                cut = false;
                generation++;
                host.connect(current.endpoint, {
                    intent: "join",
                    code: room,
                    seat: credentials.id,
                    resume: credentials.token,
                });
                return !current.endpoint.closed;
            },
        };
    }
    let disposed = false;
    function dispose() {
        if (disposed) return;
        disposed = true;
        host.stop();
    }
    return {
        join,
        create: host.reserve,
        list: host.list,
        faults: options.faults,
        pump(delta: number) {
            if (disposed) return;
            clock += delta * 1000;
            host.sweep();
            for (const link of links) link.advance(clock);
            host.advanceBots(clock);
            accumulator += Math.min(delta, Time.MAX_FIXED_STEPS * Time.FIXED_DT);
            while (accumulator >= Time.FIXED_DT) {
                host.tick();
                accumulator -= Time.FIXED_DT;
            }
        },
        dispose,
    };
}

type LegacyLocalOptions = LocalOptions & {
    recording?: (role: "server" | "client") => RecordingOptions | undefined;
    thinClients?: readonly ThinClient[];
};

type LocalHost = ReturnType<typeof localHost>;
type LocalRooms = Omit<LocalHost, "join"> & {
    join(code?: string, creator?: string): ReturnType<LocalHost["join"]>;
};

/** Same-process rooms over seeded links. join() quick-joins; join(code) claims or enters a room.
 * create() returns a private code valid for 60 seconds until claimed (undefined at capacity).
 * Pass its creator credential as join(code, creator) to take the first seat during the first
 * 30 seconds. list() lists public rooms.
 * Build a caller-owned client app with the joined seat's plugin; disposing it leaves the seat.
 * Leave explicitly if the build fails. Dispose the host to release all rooms and bots.
 * Pump frame deltas in seconds; delays and reconnect windows are milliseconds of pumped time.
 * After cut(), call resume() explicitly; without a reconnect window the drop is permanent. */
export function local(game: Game, options: LocalOptions & { rooms: true }): Promise<LocalRooms>;
/** Same-process single-room shortcut. Dispose if the client app fails to build; a built client
 * World owns session disposal. Thin clients receive and acknowledge without ticking their Worlds. */
export function local(
    game: Game,
    options: LegacyLocalOptions,
): Promise<{
    plugin: Plugin;
    /** Borrows the opt-in histories; serialize before pumping or stepping again. */
    recordings(): SessionRecording[];
    faults: LinkFaults;
    pump(delta: number): void;
    dispose(): void;
}>;
export async function local(game: Game, options: LegacyLocalOptions & { rooms?: true }) {
    if (options.rooms) return localHost(game, options);
    const serverOptions = options.recording?.("server");
    const clientOptions = options.recording?.("client");
    const serverRecorder = serverOptions && new SessionRecorder(game, "server", serverOptions);
    const clientRecorder = clientOptions && new SessionRecorder(game, "client", clientOptions);
    const app = await createServerApp(game, serverRecorder);
    let host: LocalHost | undefined;
    try {
        serverRecorder?.attachServer();
        host = localHost(game, options, app, serverRecorder);
        const seat = await host.join(undefined, undefined, clientRecorder);
        for (const thin of options.thinClients ?? []) await seat.addBot(thin);
        return {
            plugin: seat.plugin,
            recordings: () =>
                [serverRecorder?.recording, clientRecorder?.recording].filter(
                    (value) => value !== undefined,
                ),
            faults: host.faults,
            pump: host.pump,
            dispose: host.dispose,
        };
    } catch (error) {
        if (host) host.dispose();
        else app.dispose();
        throw error;
    }
}
