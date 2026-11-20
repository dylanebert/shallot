import {
    type App,
    component,
    createApp,
    DiagnosticsStore,
    type Plugin,
    type System,
    Time,
    type World,
} from "../../engine";
import {
    MultiplayerPlugin,
    ReplicationReceiver,
    type ReplicationRules,
    type ReplicationSchema,
    ReplicationSender,
} from ".";
import { observeNetwork } from "./diagnostics";
import {
    type InputRecord,
    IntentReceiver,
    type IntentRecord,
    IntentSender,
    type OperationRecord,
} from "./intent";
import { type Connection, type LinkFaults, seededLink } from "./link";
import { type RecordingOptions, SessionRecorder } from "./recording";
import { WebSocketConnection } from "./websocket";

/** A live server-side connection entity. Its transport and histories live outside ECS storage. */
export const ClientOf = component("ClientOf", {});

/** Shared protocol and game policy. Connection arguments are server eids, not client-supplied IDs.
 * Hooks borrow runtime handles; explicit disconnect calls onLeave before despawning.
 * Direct ECS removal may already have removed the entity's other components.
 * onClient installs game-specific prediction or routing at the client's first stepped preFixed
 * boundary, after app warm-up, where prediction may capture its initial World snapshot.
 * Replay composes only the game's plugins and onClient, not the tab's controls or presentation. */
export interface Game {
    plugins: readonly Plugin[];
    schema: ReplicationSchema;
    maxClients: number;
    /** Milliseconds to hold a disconnected socket seat. Omit for immediate leave. */
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
}

/** Server runtime keyed by connection entity. Removing ClientOf closes the transport and releases
 * queued bytes, its runtime handle and histories. World disposal also closes every transport.
 * Use the server role plugin; tick the World normally, including through World.step. */
class GameServer {
    private readonly _peers = new Map<number, Peer>();
    get peers(): ReadonlyMap<number, Peer> {
        return this._peers;
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
    connect(connection: Connection): number | undefined {
        if (this.peers.size >= this.game.maxClients) return undefined;
        const eid = this.world.create();
        this.world.add(eid, ClientOf);
        const diagnostics = observeNetwork(this.world, connection, `network/connections/${eid}/`);
        try {
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
            });
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
        this._peers.delete(eid);
        peer.diagnostics.remove();
        this._diagnostics.record("network/peer_count", this._peers.size);
        try {
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
        peer.sender.record(this.world.time.fixedTick);
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

/** The server role's runtime; absent in client-only Worlds. */
export const ServerSession = { create: (): { server?: GameServer } => ({}) };

/** Composes authority into an app. The caller owns the app, its clock and its disposal. */
export function server(game: Game): Plugin {
    return {
        name: "MultiplayerServer",
        dependencies: [...game.plugins, MultiplayerPlugin],
        components: [ClientOf],
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
            const socket = new WebSocket(options.url);
            let seat: string | undefined;
            let token: string | undefined;
            let generation = 0;
            let delay = 50;
            let cancel: (() => void) | undefined;
            let disposed = false;
            let failed = false;
            const failure = () =>
                new Error(`No server answered at ${options.url}. The connection failed or closed.`);
            const schedule =
                options.retryAfter ??
                ((ms: number, retry: () => void) => {
                    const timer = setTimeout(retry, ms);
                    return () => clearTimeout(timer);
                });
            function attach(socket: WebSocket) {
                socket.addEventListener("close", ({ code }) => {
                    if (disposed || world.signal.aborted) return;
                    if (!token || code === 1008) {
                        failed = true;
                        options.onError?.(failure());
                        return;
                    }
                    cancel = schedule(delay, () => {
                        if (disposed) return;
                        const url = new URL(socket.url);
                        url.searchParams.set("resume", token!);
                        url.searchParams.set("seat", seat!);
                        current = attach(new WebSocket(url));
                        generation++;
                    });
                    delay = Math.min(1000, delay * 2);
                });
                return new WebSocketConnection(socket, (message) => {
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
                    current.close();
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

/** Same-process shortcut over seeded links, not socket or process-isolation evidence.
 * Prepares child apps before returning a plugin for the caller's client app: app builds serialize,
 * so child apps cannot be built inside a plugin initializer. Dispose the session if the client
 * app cannot be built; a built client World also owns disposal. Delay values are milliseconds.
 * The caller pumps frame deltas in seconds; the server catches up at most four fixed ticks.
 * Thin clients receive and acknowledge state without ticking, predicting or rendering it. */
export async function local(
    game: Game,
    options: {
        seed: number;
        faults: LinkFaults;
        recording?: (role: "server" | "client") => RecordingOptions | undefined;
        thinClients?: readonly {
            seed: number;
            input(world: World, sender: IntentSender, tick: number): void;
        }[];
    },
) {
    const apps: App[] = [];
    const serverOptions = options.recording?.("server");
    const clientOptions = options.recording?.("client");
    const serverRecorder = serverOptions && new SessionRecorder(game, "server", serverOptions);
    const clientRecorder = clientOptions && new SessionRecorder(game, "client", clientOptions);
    try {
        const app = await createApp({
            defaults: false,
            plugins: [
                ...(serverRecorder ? [serverRecorder.plugin] : []),
                ...game.plugins,
                MultiplayerPlugin,
                server(game),
            ],
        });
        apps.push(app);
        serverRecorder?.attachServer();
        const authority = app.world.resource(ServerSession).server!;
        const link = seededLink(options.seed, options.faults);
        if (authority.connect(link.a) === undefined) throw new Error("Local client refused");
        const thins: {
            app: App;
            link: ReturnType<typeof seededLink>;
            input: IntentSender;
            receiver: ReplicationReceiver;
            run: (world: World, sender: IntentSender, tick: number) => void;
        }[] = [];
        for (const thin of options.thinClients ?? []) {
            const app = await createApp({ defaults: false, plugins: [...game.plugins] });
            apps.push(app);
            const link = seededLink(thin.seed, options.faults);
            observeNetwork(app.world, link.b);
            if (authority.connect(link.a) === undefined) throw new Error("Thin client refused");
            thins.push({
                app,
                link,
                input: new IntentSender(link.b),
                receiver: new ReplicationReceiver(app.world, game.schema, link.b),
                run: thin.input,
            });
        }
        let clock = 0,
            accumulator = 0,
            disposed = false;
        const dispose = () => {
            if (disposed) return;
            disposed = true;
            for (const app of apps.reverse()) app.dispose();
        };
        const plugin: Plugin = {
            name: "MultiplayerLocalClient",
            dependencies: [...(clientRecorder ? [clientRecorder.plugin] : []), ...game.plugins],
            initialize(world) {
                world.onDispose(dispose);
            },
            systems: [clientStartup(game, () => clientRecorder?.connection(link.b) ?? link.b)],
        };
        return {
            plugin,
            /** Borrows both opt-in histories; serialize before pumping or stepping again. */
            recordings: () =>
                [serverRecorder?.recording, clientRecorder?.recording].filter(
                    (value) => value !== undefined,
                ),
            faults: options.faults,
            pump(delta: number) {
                if (disposed) return;
                clock += delta * 1000;
                link.advance(clock);
                for (const thin of thins) {
                    thin.link.advance(clock);
                    thin.input.flush(thin.receiver.flush());
                }
                accumulator += Math.min(delta, Time.MAX_FIXED_STEPS * Time.FIXED_DT);
                while (accumulator >= Time.FIXED_DT) {
                    for (const thin of thins) {
                        thin.run(thin.app.world, thin.input, authority.world.time.fixedTick);
                        thin.input.flush();
                    }
                    authority.world.tick();
                    accumulator -= Time.FIXED_DT;
                }
            },
            dispose,
        };
    } catch (error) {
        for (const app of apps.reverse()) app.dispose();
        throw error;
    }
}
