import type { ServerWebSocket } from "bun";
import { type App, createApp, Time } from "../../engine";
import { MultiplayerPlugin } from ".";
import { stampReceipt } from "./diagnostics";
import { type Game, ServerSession, server as serverRole } from "./game";
import type { Connection } from "./link";
import { type RecordingOptions, SessionRecorder } from "./recording";

type SocketData = { url: URL } | ServerWebSocketConnection | undefined;

/** A single accepted peer. close releases queued bytes as well as the socket. */
export class ServerWebSocketConnection implements Connection {
    private _messages: Uint8Array[] = [];
    private _closed = false;
    readonly socket: ServerWebSocket<SocketData>;

    constructor(socket: ServerWebSocket<SocketData>) {
        this.socket = socket;
    }

    get closed(): boolean {
        return this._closed || this.socket.readyState !== 1;
    }

    close(): void {
        if (this._closed) return;
        this._closed = true;
        this._messages = [];
        this.socket.close();
    }

    /** Bun owns queued writes; a dropped write closes the peer rather than silently losing bytes.
     * Writes to a closing or closed peer are discarded, so one peer cannot throw into a server tick;
     * `disconnect` reports its close. */
    send(bytes: Uint8Array): void {
        if (this.closed) return;
        if (this.socket.sendBinary(bytes) === 0) this.socket.close(1011, "Write failed");
    }

    receive(): Uint8Array[] {
        const messages = this._messages;
        this._messages = [];
        return messages;
    }

    /** @internal */
    push(bytes: Uint8Array): void {
        if (!this.closed) this._messages.push(stampReceipt(bytes.slice(), performance.now()));
    }
}

/** Self-hosted binary endpoint. Every upgrade gets a distinct connection; callbacks run once
 * at open and close. Authentication and routing belong in the application/reverse proxy. */
export function serveWebSocket(options: {
    hostname?: string;
    port: number;
    connect: (connection: ServerWebSocketConnection, url?: URL) => void;
    disconnect: (connection: ServerWebSocketConnection) => void;
}) {
    return Bun.serve<SocketData>({
        hostname: options.hostname,
        port: options.port,
        fetch(request, server) {
            if (server.upgrade(request, { data: { url: new URL(request.url) } })) return;
            return new Response("WebSocket upgrade required", { status: 426 });
        },
        websocket: {
            open(socket) {
                const data = socket.data;
                const connection = new ServerWebSocketConnection(socket);
                socket.data = connection;
                options.connect(connection, data && "url" in data ? data.url : undefined);
            },
            message(socket, message) {
                if (typeof message === "string") {
                    socket.close(1003, "Binary messages required");
                    return;
                }
                if (socket.data instanceof ServerWebSocketConnection) socket.data.push(message);
            },
            close(socket) {
                if (!(socket.data instanceof ServerWebSocketConnection)) return;
                socket.data.close();
                options.disconnect(socket.data);
            },
        },
    });
}

function createServerApp(game: Game, recorder?: SessionRecorder) {
    return createApp({
        defaults: false,
        plugins: [
            ...(recorder ? [recorder.plugin] : []),
            ...game.plugins,
            MultiplayerPlugin,
            serverRole(game),
        ],
    });
}

function startClock(tick: () => void) {
    let previous = performance.now();
    let accumulator = 0;
    const timer = setInterval(() => {
        const now = performance.now();
        accumulator += Math.min((now - previous) / 1000, Time.MAX_FIXED_STEPS * Time.FIXED_DT);
        previous = now;
        while (accumulator >= Time.FIXED_DT) {
            tick();
            accumulator -= Time.FIXED_DT;
        }
    }, Time.FIXED_DT * 1000);
    return () => clearInterval(timer);
}

/** Self-hosts one headless authoritative game. Owns admission, a 60 Hz clock with at most four
 * catch-up ticks per callback, dead-peer cleanup and SIGTERM shutdown. stop is idempotent and
 * releases the timer, sockets, signal listener and World. The caller chooses address and hosting. */
export async function serveGame(
    game: Game,
    options: { hostname?: string; port: number; recording?: RecordingOptions },
) {
    const recorder = options.recording && new SessionRecorder(game, "server", options.recording);
    const app = await createServerApp(game, recorder);
    recorder?.attachServer();
    const authority = app.world.resource(ServerSession).server!;
    const entities = new WeakMap<ServerWebSocketConnection, number>();
    let endpoint: ReturnType<typeof serveWebSocket>;
    try {
        endpoint = serveWebSocket({
            ...options,
            connect(connection) {
                const eid = authority.connect(connection);
                if (eid === undefined) connection.socket.close(1013, "Peer limit reached");
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
    } catch (error) {
        app.dispose();
        throw error;
    }
    let stopped = false;
    const stopClock = startClock(() => app.world.tick());
    const stop = () => {
        if (stopped) return;
        stopped = true;
        stopClock();
        process.off("SIGTERM", stop);
        try {
            recorder?.finish();
        } finally {
            try {
                endpoint.stop(true);
            } finally {
                app.dispose();
            }
        }
    };
    app.world.onDispose(stop);
    process.on("SIGTERM", stop);
    return { port: endpoint.port, world: app.world, stop };
}

interface Room {
    code: string;
    private: boolean;
    seats: Map<ServerWebSocketConnection, number | undefined>;
    app?: App;
}

/** Hosts independent authoritative Worlds of one game. On the WebSocket URL, use
 * `?intent=joinOrCreate` to fill a public room, `?intent=create&code=…` to start
 * a private room, or `?intent=join&code=…` to enter an existing room. A creator
 * supplies a random URL-safe code (for example a UUID) and shares it out of band;
 * an already active code is refused. Admission failures close the socket without
 * sending a game image. A room owns its World only while it has seats, including
 * seats reserved during startup. stop closes all rooms and is idempotent. */
export function serveRooms(game: Game, options: { hostname?: string; port: number }) {
    if (!Number.isSafeInteger(game.maxClients) || game.maxClients < 1)
        throw new Error("maxClients must be a positive integer");
    const rooms = new Map<string, Room>();
    const memberships = new WeakMap<ServerWebSocketConnection, Room>();
    let stopped = false;

    function retire(room: Room) {
        rooms.delete(room.code);
        for (const connection of room.seats.keys()) {
            memberships.delete(connection);
            connection.close();
        }
        room.seats.clear();
        room.app?.dispose();
    }

    function leave(connection: ServerWebSocketConnection) {
        const room = memberships.get(connection);
        if (!room) return;
        memberships.delete(connection);
        const eid = room.seats.get(connection);
        room.seats.delete(connection);
        const authority = room.app?.world.resource(ServerSession).server;
        try {
            if (eid !== undefined && authority?.peers.get(eid)?.connection === connection)
                authority.disconnect(eid);
        } finally {
            if (!room.seats.size) retire(room);
        }
    }

    function admit(room: Room, connection: ServerWebSocketConnection) {
        if (connection.closed) {
            leave(connection);
            return;
        }
        try {
            const eid = room.app!.world.resource(ServerSession).server!.connect(connection);
            if (eid === undefined) connection.socket.close(1013, "Room full");
            else {
                room.seats.set(connection, eid);
                return;
            }
        } catch (error) {
            console.error(error);
            connection.close();
        }
        leave(connection);
    }

    async function start(room: Room) {
        let app: App;
        try {
            app = await createServerApp(game);
        } catch (error) {
            console.error(error);
            if (rooms.get(room.code) === room) retire(room);
            return;
        }
        if (rooms.get(room.code) !== room) {
            app.dispose();
            return;
        }
        room.app = app;
        for (const connection of room.seats.keys()) admit(room, connection);
    }

    const endpoint = serveWebSocket({
        ...options,
        connect(connection, url) {
            const intent = url?.searchParams.get("intent");
            const code = url?.searchParams.get("code");
            let room: Room | undefined;
            let newCode: string | undefined;
            if (stopped) {
                connection.close();
                return;
            }
            if (intent === "joinOrCreate" && code === null) {
                for (const candidate of rooms.values()) {
                    if (!candidate.private && candidate.seats.size < game.maxClients) {
                        room = candidate;
                        break;
                    }
                }
                if (!room) {
                    do newCode = crypto.randomUUID();
                    while (rooms.has(newCode));
                }
            } else if (code && /^[\w-]{1,64}$/.test(code)) {
                if (intent === "join") room = rooms.get(code);
                if (intent === "create" && !rooms.has(code)) newCode = code;
            }
            if (newCode !== undefined) {
                room = { code: newCode, private: intent === "create", seats: new Map() };
                rooms.set(newCode, room);
            }
            if (!room) {
                connection.socket.close(1008, "Room admission refused");
                return;
            }
            if (room.seats.size >= game.maxClients) {
                connection.socket.close(1013, "Room full");
                return;
            }
            room.seats.set(connection, undefined);
            memberships.set(connection, room);
            if (newCode !== undefined) void start(room);
            else if (room.app) admit(room, connection);
        },
        disconnect: leave,
    });
    const stopClock = startClock(() => {
        for (const room of rooms.values()) room.app?.world.tick();
    });
    const stop = () => {
        if (stopped) return;
        stopped = true;
        stopClock();
        process.off("SIGTERM", stop);
        try {
            endpoint.stop(true);
        } finally {
            for (const room of rooms.values()) retire(room);
        }
    };
    process.on("SIGTERM", stop);
    return { port: endpoint.port, stop };
}
