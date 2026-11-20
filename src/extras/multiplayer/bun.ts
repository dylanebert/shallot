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

    close(code?: number, reason?: string): void {
        if (this._closed) return;
        this._closed = true;
        this._messages = [];
        this.socket.close(code, reason);
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
    fetch?: (request: Request) => Response | undefined;
}) {
    return Bun.serve<SocketData>({
        hostname: options.hostname,
        port: options.port,
        fetch(request, server) {
            if (server.upgrade(request, { data: { url: new URL(request.url) } })) return;
            return (
                options.fetch?.(request) ??
                new Response("WebSocket upgrade required", { status: 426 })
            );
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
    try {
        const host = hostRooms(game, options, app, recorder);
        app.world.onDispose(host.stop);
        return { ...host, world: app.world };
    } catch (error) {
        app.dispose();
        throw error;
    }
}

interface Room {
    code: string;
    private: boolean;
    seats: Map<ServerWebSocketConnection, number | undefined>;
    app?: App;
}

const CODE_ALPHABET = "0123456789ABCDEFGHJKMNPQRSTVWXYZ";
const CLAIM_MS = 60_000;
const MAX_UNCLAIMED_CODES = 128;

/** Hosts independent authoritative Worlds of one game. POST /rooms returns 201
 * and JSON { code } for a private room. Share the code out of band and claim it
 * within 60 seconds with `?intent=join&code=…`. A code is 10 case-insensitive
 * Crockford Base32 characters (no I, L, O or U). Codes are unique among live
 * rooms and may be randomly issued again after retirement. The host holds at
 * most 128 unclaimed codes to allow concurrent invitations without unbounded
 * idle records or timers. At capacity, POST returns 503 with Retry-After: 60
 * (seconds); claim or expiry frees a slot. Reservations do not run on the game clock.
 * A socket cannot create a private room with a client-supplied code.
 * `?intent=joinOrCreate` fills an available public room before making another.
 * Refusals send no game image: 1013 means full, 1008 means unknown/expired code
 * or invalid intent, and 1011 means server initialization failed.
 * The first seat starts the World. Pending seats count toward maxClients but
 * are not admitted before startup succeeds. Last leave or stop retires the room
 * immediately; an in-flight build is not canceled, but disposes on success
 * without admitting anyone. A rejected build cleans up its partial World and
 * retires the code, closing any seats still waiting. stop is idempotent. */
export function serveRooms(game: Game, options: { hostname?: string; port: number }) {
    return hostRooms(game, options);
}

function hostRooms(
    game: Game,
    options: { hostname?: string; port: number },
    pinnedApp?: App,
    recorder?: SessionRecorder,
) {
    if (!Number.isSafeInteger(game.maxClients) || game.maxClients < 1)
        throw new Error("maxClients must be a positive integer");
    const rooms = new Map<string, Room>();
    const active = new Set<App>();
    const unclaimed = new Map<
        string,
        { timer: ReturnType<typeof setTimeout>; expiresAt: number }
    >();
    const memberships = new WeakMap<ServerWebSocketConnection, Room>();
    const pinned: Room | undefined = pinnedApp && {
        code: "",
        private: true,
        seats: new Map(),
        app: pinnedApp,
    };
    if (pinned && pinnedApp) {
        rooms.set(pinned.code, pinned);
        active.add(pinnedApp);
    }
    let stopped = false;

    function issueCode(): string {
        let code: string;
        do {
            const bytes = crypto.getRandomValues(new Uint8Array(10));
            code = Array.from(bytes, (byte) => CODE_ALPHABET[byte & 31]).join("");
        } while (rooms.has(code) || unclaimed.has(code));
        return code;
    }

    function createRoom(code: string, isPrivate: boolean): Room {
        const room: Room = { code, private: isPrivate, seats: new Map() };
        rooms.set(code, room);
        return room;
    }

    function retire(room: Room, code?: number, reason?: string) {
        rooms.delete(room.code);
        for (const connection of room.seats.keys()) {
            memberships.delete(connection);
            connection.close(code, reason);
        }
        room.seats.clear();
        if (room.app) {
            active.delete(room.app);
            room.app.dispose();
        }
    }

    function leave(connection: ServerWebSocketConnection) {
        const room = memberships.get(connection);
        if (!room || stopped) return;
        memberships.delete(connection);
        const eid = room.seats.get(connection);
        room.seats.delete(connection);
        const authority = room.app?.world.resource(ServerSession).server;
        try {
            if (
                eid !== undefined &&
                authority?.peers.get(eid)?.connection ===
                    (recorder?.connection(connection) ?? connection)
            )
                authority?.disconnect(eid);
        } finally {
            if (room !== pinned && !room.seats.size) retire(room);
        }
    }

    function admit(room: Room, connection: ServerWebSocketConnection) {
        if (connection.closed) {
            leave(connection);
            return;
        }
        try {
            const eid = room.app!.world.resource(ServerSession).server!.connect(connection);
            if (eid === undefined) refuseFull(connection);
            else {
                room.seats.set(connection, eid);
                return;
            }
        } catch (error) {
            console.error(error);
            connection.close(1011, "Admission failed");
        }
        leave(connection);
    }

    async function start(room: Room) {
        let app: App;
        try {
            app = await createServerApp(game);
        } catch (error) {
            console.error(error);
            if (rooms.get(room.code) === room) retire(room, 1011, "Room initialization failed");
            return;
        }
        if (rooms.get(room.code) !== room) {
            app.dispose();
            return;
        }
        room.app = app;
        active.add(app);
        for (const connection of room.seats.keys()) admit(room, connection);
    }

    function refuseFull(connection: ServerWebSocketConnection) {
        connection.close(1013, pinned ? "Peer limit reached" : "Room full");
    }

    function selectRoom(url?: URL) {
        const intent = url?.searchParams.get("intent");
        const code = url?.searchParams.get("code");
        if (intent === "joinOrCreate" && code === null) {
            for (const room of rooms.values()) {
                if (!room.private && room.seats.size < game.maxClients) return room;
            }
            return createRoom(issueCode(), false);
        }
        if (intent === "join" && code) {
            const key = code.toUpperCase();
            const reservation = unclaimed.get(key);
            if (reservation !== undefined) {
                clearTimeout(reservation.timer);
                unclaimed.delete(key);
                if (performance.now() < reservation.expiresAt) return createRoom(key, true);
                return;
            }
            return rooms.get(key);
        }
    }

    const endpoint = serveWebSocket({
        ...options,
        fetch(request) {
            if (pinned || request.method !== "POST" || new URL(request.url).pathname !== "/rooms")
                return;
            if (stopped) return new Response(null, { status: 503 });
            if (unclaimed.size >= MAX_UNCLAIMED_CODES) {
                return new Response(null, {
                    status: 503,
                    headers: { "Retry-After": String(CLAIM_MS / 1000) },
                });
            }
            const code = issueCode();
            unclaimed.set(code, {
                expiresAt: performance.now() + CLAIM_MS,
                timer: setTimeout(() => unclaimed.delete(code), CLAIM_MS),
            });
            return Response.json({ code }, { status: 201 });
        },
        connect(connection, url) {
            if (stopped) {
                connection.close(1001, "Server stopping");
                return;
            }
            const room = pinned ?? selectRoom(url);
            if (!room) {
                connection.close(1008, "Unknown room or invalid intent");
                return;
            }
            if (pinned) {
                for (const peer of room.seats.keys()) if (peer.closed) leave(peer);
            }
            if (room.seats.size >= game.maxClients) {
                refuseFull(connection);
                return;
            }
            room.seats.set(connection, undefined);
            memberships.set(connection, room);
            if (room.app) admit(room, connection);
            else if (room.seats.size === 1) void start(room);
        },
        disconnect: leave,
    });
    const stopClock = startClock(() => {
        for (const app of active) app.world.tick();
    });
    const stop = () => {
        if (stopped) return;
        stopped = true;
        stopClock();
        process.off("SIGTERM", stop);
        for (const { timer } of unclaimed.values()) clearTimeout(timer);
        unclaimed.clear();
        try {
            recorder?.finish();
        } finally {
            try {
                for (const room of rooms.values()) retire(room, 1001, "Server stopping");
            } finally {
                endpoint.stop(true);
            }
        }
    };
    process.on("SIGTERM", stop);
    return { port: endpoint.port, stop };
}
