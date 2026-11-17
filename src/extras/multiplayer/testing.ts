import { type App, createApp, type Plugin, Time, type World } from "../../engine";
import { MultiplayerPlugin, ReplicationReceiver } from ".";
import { observeNetwork } from "./diagnostics";
import { clientStartup, type Game, ServerSession, server as serverRole } from "./game";
import { IntentSender } from "./intent";
import { type Connection, type LinkFaults, seededLink } from "./link";
import type { PeerProxy } from "./proxy";
import { type RecordingOptions, SessionRecorder } from "./recording";
import { WebSocketConnection } from "./websocket";

export type { PeerProxy, StreamFaults } from "./proxy";

/** Composes caller-stepped test Worlds. Link faults use milliseconds; each transport round advances
 * their shared clock by one fixed tick. Socket mode uses real loopback IO. proxy opts into
 * per-peer TCP stream controls, initially disabled; it requires WebSocket transport.
 * Proxy delay and bandwidth use wall time, not stepped time.
 * Seed precedence: options.seed, then SHALLOT_MULTIPLAYER_SEED (decimal u32), then 1.
 * The environment is read only when options.seed is omitted. The seed and its source print at
 * creation; failures include the seed. Client i uses seed + i.
 * Clients run the game's onClient at their first stepped boundary. Thin clients only receive and
 * acknowledge state: their Worlds do not simulate and onClient is not called.
 * Dispose the session, not just its Worlds. Disposal closes all owned peers and fails if any World,
 * connection entity, transport or loopback listener/socket survives cleanup. Every endpoint and
 * proxy listener's release is observed through a refused TCP connection. This is an owned-resource check,
 * not a process-wide heap or handle leak detector. */
export async function createSession(
    game: Game,
    options: {
        seed?: number;
        clients: number;
        thinClients?: number;
        transport?: "link" | "websocket";
        faults?: LinkFaults;
        proxy?: boolean;
        recording?: (role: "server" | `client[${number}]`) => RecordingOptions | undefined;
    },
) {
    const { clients, thinClients = 0, transport = "link" } = options;
    const environment =
        options.seed === undefined && typeof process !== "undefined"
            ? process.env.SHALLOT_MULTIPLAYER_SEED
            : undefined;
    if (environment !== undefined && !/^\d+$/.test(environment))
        throw new Error("SHALLOT_MULTIPLAYER_SEED must be a decimal u32");
    const seed = options.seed ?? (environment === undefined ? 1 : Number(environment));
    const source =
        options.seed !== undefined
            ? "options"
            : environment !== undefined
              ? "SHALLOT_MULTIPLAYER_SEED"
              : "default";
    for (const count of [clients, thinClients])
        if (!Number.isSafeInteger(count) || count < 0)
            throw new Error("Session client counts must be nonnegative integers");
    if (clients + thinClients > game.maxClients)
        throw new Error(`Session exceeds game's peer limit seed=${seed}`);
    if (!Number.isSafeInteger(seed) || seed < 0 || seed > 0xffffffff)
        throw new Error("Session seed must be a u32");
    if (transport !== "websocket" && options.proxy)
        throw new Error("TCP proxies require WebSocket transport");
    if (transport === "websocket" && options.faults)
        throw new Error("Link faults do not impair WebSockets");
    console.info(`multiplayer session seed=${seed} source=${source} transport=${transport}`);
    const apps: App[] = [];
    const recordings: SessionRecorder[] = [];
    const links: ReturnType<typeof seededLink>[] = [];
    const sockets: WebSocket[] = [];
    let endpoint: ReturnType<typeof import("./bun").serveWebSocket> | undefined;
    const proxies: Awaited<ReturnType<typeof import("./proxy").createProxy>>[] = [];
    let clock = 0;
    let disposed = false;
    const peers: {
        name: string;
        world: World;
        connection?: Connection;
        proxy?: PeerProxy;
        input?: IntentSender;
        receiver?: ReplicationReceiver;
        dispose(): void;
    }[] = [];
    type Peer = (typeof peers)[number];
    const describe = (peer: Peer) =>
        `${peer.name} tick=${peer.world.time.fixedTick} connection=${peer.connection ? (peer.connection.closed ? "closed" : "open") : "authority"} disposed=${peer.world.signal.aborted} seed=${seed}`;
    const failure = (peer: Peer, message: string) => new Error(`${message}: ${describe(peer)}`);
    const io = async () => {
        if (transport === "websocket") await new Promise((resolve) => setTimeout(resolve, 0));
    };
    try {
        const recordingOptions = options.recording?.("server");
        const recorder = recordingOptions && new SessionRecorder(game, "server", recordingOptions);
        if (recorder) recordings.push(recorder);
        const app = await createApp({
            defaults: false,
            plugins: [...(recorder ? [recorder.plugin] : []), serverRole(game)],
        });
        recorder?.attachServer();
        apps.push(app);
        const authority = app.world.resource(ServerSession).server!;
        const server: Peer = { name: "server", world: app.world, dispose: () => app.dispose() };
        peers.push(server);
        if (transport === "websocket") {
            const { serveWebSocket } = await import("./bun");
            const entities = new WeakMap<Connection, number>();
            endpoint = serveWebSocket({
                hostname: "127.0.0.1",
                port: 0,
                connect(connection) {
                    const eid = authority.connect(connection);
                    if (eid === undefined) connection.close();
                    else entities.set(connection, eid);
                },
                disconnect(connection) {
                    const eid = entities.get(connection);
                    if (
                        eid !== undefined &&
                        authority.peers.get(eid)?.connection ===
                            (recorder?.connection(connection) ?? connection)
                    )
                        authority.disconnect(eid);
                    entities.delete(connection);
                },
            });
        }
        const regular: Peer[] = [],
            thins: Peer[] = [];
        for (let i = 0; i < clients + thinClients; i++) {
            let connection: Connection;
            let proxy: (typeof proxies)[number] | undefined;
            if (transport === "link") {
                const link = seededLink(
                    (seed + i) >>> 0,
                    options.faults ?? {
                        minDelay: 0,
                        maxDelay: 0,
                        loss: 0,
                        duplication: 0,
                    },
                );
                links.push(link);
                connection = link.b;
                if (authority.connect(link.a) === undefined)
                    throw new Error(`client[${i}] refused seed=${seed}`);
            } else {
                if (options.proxy) {
                    const { createProxy } = await import("./proxy");
                    proxy = await createProxy(endpoint!.port!, (seed + i) >>> 0);
                    proxies.push(proxy);
                }
                const socket = new WebSocket(`ws://127.0.0.1:${proxy?.port ?? endpoint!.port}`);
                sockets.push(socket);
                connection = new WebSocketConnection(socket);
                await new Promise<void>((resolve, reject) => {
                    socket.addEventListener("open", () => resolve(), { once: true });
                    socket.addEventListener(
                        "error",
                        () => reject(new Error(`client[${i}] socket error seed=${seed}`)),
                        { once: true },
                    );
                    socket.addEventListener(
                        "close",
                        () => reject(new Error(`client[${i}] socket closed seed=${seed}`)),
                        { once: true },
                    );
                });
                if (connection.closed) throw new Error(`client[${i}] refused seed=${seed}`);
            }
            const thin = i >= clients;
            const recordingOptions = thin ? undefined : options.recording?.(`client[${i}]`);
            const clientRecorder =
                recordingOptions && new SessionRecorder(game, "client", recordingOptions);
            if (clientRecorder) {
                recordings.push(clientRecorder);
                connection = clientRecorder.connection(connection);
            }
            const startup: Plugin = {
                name: "SessionClient",
                dependencies: [...game.plugins, MultiplayerPlugin],
                initialize(world) {
                    world.onDispose(() => connection.close());
                },
                systems: thin ? [] : [clientStartup(game, () => connection)],
            };
            const app = await createApp({
                defaults: false,
                plugins: [...(clientRecorder ? [clientRecorder.plugin] : []), startup],
            });
            apps.push(app);
            const peer: Peer = {
                name: `${thin ? "thin" : "client"}[${thin ? i - clients : i}]`,
                world: app.world,
                connection,
                dispose: () => app.dispose(),
            };
            if (thin) {
                observeNetwork(app.world, connection);
                peer.input = new IntentSender(connection);
                peer.receiver = new ReplicationReceiver(app.world, game.schema, connection);
            }
            if (proxy) peer.proxy = proxy.controls;
            peers.push(peer);
            (thin ? thins : regular).push(peer);
        }
        const active = (peer: Peer) => {
            if (disposed) throw failure(peer, "Session disposed");
            if (!peers.includes(peer)) throw new Error(`Foreign session peer seed=${seed}`);
            if (peer.world.signal.aborted) throw failure(peer, "Peer disposed");
        };
        const count = (ticks: number) => {
            if (!Number.isSafeInteger(ticks) || ticks < 0)
                throw new Error("Tick budget must be a nonnegative integer");
        };
        const step = async (peer: Peer, ticks = 1) => {
            active(peer);
            count(ticks);
            for (let i = 0; i < ticks; i++) {
                try {
                    if (thins.includes(peer)) peer.input!.flush(peer.receiver!.flush());
                    else if (peer === server) peer.world.tick();
                    else peer.world.step(Time.FIXED_DT);
                    await io();
                    for (const proxy of proxies) proxy.check();
                } catch (error) {
                    throw failure(peer, `Step failed (${String(error)})`);
                }
            }
        };
        const advance = async (ticks = 1) => {
            active(server);
            count(ticks);
            for (let i = 0; i < ticks; i++) {
                clock += Time.FIXED_DT * 1000;
                for (const link of links) link.advance(clock);
                for (const peer of [...regular, ...thins, server]) await step(peer);
            }
        };
        return {
            server,
            clients: regular,
            thinClients: thins,
            /** Advances only this peer: server ticks, client fixed-duration frames, thin drain/ack.
             * Does not advance the link clock. Client frames run preFixed, prediction and presentation. */
            step,
            /** Advances transport time, then all clients and the server, in that order. */
            advance,
            /** Tests immediately, then after each full transport round, up to the tick budget. */
            async waitFor(peer: Peer, predicate: () => boolean, ticks: number, label: string) {
                active(peer);
                count(ticks);
                for (let i = 0; ; i++) {
                    try {
                        if (predicate()) return;
                    } catch (error) {
                        throw failure(peer, `Wait predicate failed: ${label} (${String(error)})`);
                    }
                    if (i === ticks) throw failure(peer, `Wait failed: ${label}; budget=${ticks}`);
                    await advance();
                }
            },
            async dispose() {
                if (disposed) return;
                disposed = true;
                const errors: Error[] = [];
                for (const recorder of recordings) {
                    try {
                        recorder.finish();
                    } catch (error) {
                        errors.push(failure(server, `Recording write failed (${String(error)})`));
                    }
                }
                for (const peer of [...peers].reverse()) {
                    try {
                        peer.dispose();
                    } catch (error) {
                        errors.push(failure(peer, String(error)));
                    }
                }
                for (const proxy of proxies) proxy.stop();
                const port = endpoint?.port;
                try {
                    await endpoint?.stop(true);
                } catch (error) {
                    errors.push(failure(server, `Endpoint stop failed (${String(error)})`));
                }
                if (port !== undefined) {
                    const { listenerReleased } = await import("./proxy");
                    for (const [kind, listenerPort] of [
                        ["endpoint", port],
                        ...proxies.map((proxy) => ["proxy", proxy.port] as const),
                    ] as const) {
                        try {
                            if (!(await listenerReleased(listenerPort)))
                                errors.push(
                                    failure(
                                        server,
                                        `Disposal leaked loopback ${kind} 127.0.0.1:${listenerPort}`,
                                    ),
                                );
                        } catch (error) {
                            errors.push(
                                failure(
                                    server,
                                    `Endpoint release probe failed 127.0.0.1:${listenerPort} (${String(error)})`,
                                ),
                            );
                        }
                    }
                }
                for (
                    let i = 0;
                    i < 32 &&
                    (sockets.some((socket) => socket.readyState !== 3) ||
                        proxies.some((proxy) => proxy.pendingSockets));
                    i++
                )
                    await io();
                for (const peer of peers) {
                    if (!peer.world.signal.aborted || (peer.connection && !peer.connection.closed))
                        errors.push(failure(peer, "Disposal leaked peer"));
                }
                if (authority.peers.size)
                    errors.push(
                        failure(
                            server,
                            `Disposal leaked ${authority.peers.size} connection entities`,
                        ),
                    );
                if (
                    sockets.some((socket) => socket.readyState !== 3) ||
                    endpoint?.pendingWebSockets ||
                    proxies.some((proxy) => proxy.pendingSockets)
                )
                    errors.push(failure(server, "Disposal leaked loopback sockets"));
                // Force cleanup after reporting a failed ownership path, as NetCodeTestWorld does.
                endpoint?.[Symbol.dispose]();
                for (const proxy of proxies) proxy.forceClose();
                for (const app of [...apps].reverse()) app.dispose();
                for (const link of links) link.a.close();
                for (const socket of sockets) socket.close();
                if (errors.length)
                    throw new AggregateError(
                        errors,
                        errors.map((error) => error.message).join("\n"),
                    );
            },
        };
    } catch (error) {
        for (const recorder of recordings) recorder.finish();
        for (const app of [...apps].reverse()) app.dispose();
        for (const link of links) link.a.close();
        for (const socket of sockets) socket.close();
        for (const proxy of proxies) proxy.forceClose();
        await endpoint?.stop(true);
        throw error;
    }
}
