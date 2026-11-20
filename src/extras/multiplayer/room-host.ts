import { type App, createApp } from "../../engine";
import { MultiplayerPlugin } from ".";
import { transportClock } from "./diagnostics";
import { type Game, ServerSession, server } from "./game";
import type { Connection } from "./link";
import type { SessionRecorder } from "./recording";

export function createServerApp(game: Game, recorder?: SessionRecorder) {
    return createApp({
        defaults: false,
        plugins: [
            ...(recorder ? [recorder.plugin] : []),
            ...game.plugins,
            MultiplayerPlugin,
            server(game),
        ],
    });
}

interface Seat<C> {
    room: Room<C>;
    eid: number;
    id: string;
    token: string;
    transport: { stream: C } & Connection;
    connection: Connection;
    expiresAt?: number;
}

interface Room<C> {
    code: string;
    private: boolean;
    seats: Map<C, number | undefined>;
    app?: App;
    ready?: Promise<void>;
    held?: number;
    reservation?: Reservation;
}

interface Reservation {
    creator: string;
    issuedAt: number;
}

const CODE_ALPHABET = "0123456789ABCDEFGHJKMNPQRSTVWXYZ";
export const ROOM_CODE_CLAIM_MS = 60_000;
const CREATOR_HOLD_MS = 30_000;
const MAX_UNCLAIMED_CODES = 128;

export function roomHost<
    C extends Connection & {
        close(code?: number, reason?: string): void;
        [transportClock]?: () => number;
    },
>(
    game: Game,
    options: {
        reservationNow?: () => number;
        reconnectNow?: () => number;
        sendToken(connection: C, id: string, token: string): void;
        sendRoom?(connection: C, code: string): void;
    },
    pinnedApp?: App,
    recorder?: SessionRecorder,
) {
    if (!Number.isSafeInteger(game.maxClients) || game.maxClients < 1)
        throw new Error("maxClients must be a positive integer");
    const reconnectMs = game.reconnectMs ?? 0;
    if (!Number.isFinite(reconnectMs) || reconnectMs < 0)
        throw new Error("reconnectMs must be finite and nonnegative");
    const now = options.reconnectNow ?? (() => performance.now());
    const seats = new WeakMap<C, Seat<C>>();
    const resumable = new WeakSet<C>();
    const held = new Map<string, Seat<C>>();
    const count = (room: Room<C>) => room.seats.size + (room.held ?? 0);
    const rooms = new Map<string, Room<C>>();
    const active = new Set<App>();
    const unclaimed = new Map<string, Reservation>();
    const reservationNow = options.reservationNow ?? (() => performance.now());
    const memberships = new WeakMap<C, Room<C>>();
    const pinned: Room<C> | undefined = pinnedApp && {
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

    function expireCodes(now: number) {
        for (const [code, { issuedAt }] of unclaimed) {
            if (issuedAt + ROOM_CODE_CLAIM_MS > now) break;
            unclaimed.delete(code);
        }
    }

    function issueCode(): string {
        let code: string;
        do {
            const bytes = crypto.getRandomValues(new Uint8Array(10));
            code = Array.from(bytes, (byte) => CODE_ALPHABET[byte & 31]).join("");
        } while (rooms.has(code) || unclaimed.has(code));
        return code;
    }

    function createRoom(code: string, isPrivate: boolean): Room<C> {
        const room: Room<C> = { code, private: isPrivate, seats: new Map() };
        rooms.set(code, room);
        return room;
    }

    function retire(room: Room<C>, code?: number, reason?: string) {
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

    function leave(connection: C, hold = true) {
        const room = memberships.get(connection);
        if (stopped) return;
        if (!room) {
            const seat = seats.get(connection);
            if (!hold && seat?.transport.stream === connection && held.get(seat.token) === seat)
                expireSeat(seat);
            return;
        }
        memberships.delete(connection);
        const eid = room.seats.get(connection);
        room.seats.delete(connection);
        const authority = room.app?.world.resource(ServerSession).server;
        const seat = seats.get(connection);
        if (hold && seat && authority?.peers.get(seat.eid)?.connection === seat.connection) {
            authority.hold(seat.eid);
            seat.expiresAt = now() + reconnectMs;
            held.set(seat.token, seat);
            room.held = (room.held ?? 0) + 1;
            return;
        }
        try {
            if (
                eid !== undefined &&
                authority?.peers.get(eid)?.connection ===
                    (seat?.connection ?? recorder?.connection(connection) ?? connection)
            )
                authority?.disconnect(eid);
        } finally {
            if (room !== pinned && !count(room)) retire(room);
        }
    }

    function admit(room: Room<C>, connection: C) {
        if (connection.closed) {
            leave(connection);
            return;
        }
        try {
            const transport: Seat<C>["transport"] & { [transportClock]: () => number } = {
                stream: connection,
                [transportClock]: () => transport.stream[transportClock]?.() ?? performance.now(),
                get closed() {
                    return this.stream.closed;
                },
                close() {
                    this.stream.close();
                },
                send(bytes: Uint8Array) {
                    this.stream.send(bytes);
                },
                receive() {
                    return this.stream.receive();
                },
            };
            const eid = room
                .app!.world.resource(ServerSession)
                .server!.connect(reconnectMs && resumable.has(connection) ? transport : connection);
            if (eid === undefined) refuseFull(connection);
            else {
                room.seats.set(connection, eid);
                room.reservation = undefined;
                if (room.code) options.sendRoom?.(connection, room.code);
                if (reconnectMs && resumable.has(connection)) {
                    const token = crypto.randomUUID();
                    const seat = {
                        room,
                        eid,
                        id: crypto.randomUUID(),
                        token,
                        transport,
                        connection: recorder?.connection(transport) ?? transport,
                    };
                    seats.set(connection, seat);
                    sendToken(connection, seat);
                }
                return;
            }
        } catch (error) {
            console.error(error);
            connection.close(1011, "Admission failed");
        }
        leave(connection);
    }

    async function start(room: Room<C>) {
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

    function sendToken(connection: C, seat: Seat<C>) {
        options.sendToken(connection, seat.id, seat.token);
    }

    function release(seat: Seat<C>) {
        held.delete(seat.token);
        seat.room.held!--;
    }

    function expireSeat(seat: Seat<C>) {
        const authority = seat.room.app!.world.resource(ServerSession).server!;
        release(seat);
        try {
            if (authority.peers.get(seat.eid)?.connection === seat.connection)
                authority.disconnect(seat.eid);
        } finally {
            if (seat.room !== pinned && !count(seat.room)) retire(seat.room);
        }
    }

    function expireSeats() {
        const time = now();
        for (const seat of held.values()) {
            const authority = seat.room.app!.world.resource(ServerSession).server!;
            const owns = authority.peers.get(seat.eid)?.connection === seat.connection;
            if (seat.expiresAt! > time && owns) continue;
            expireSeat(seat);
        }
    }

    function refuseFull(connection: C) {
        connection.close(1013, pinned ? "Peer limit reached" : "Room full");
    }

    function selectRoom(
        connection: C,
        intent: string | null,
        code: string | null,
        creator: string | null,
    ) {
        if (intent === "joinOrCreate" && code === null) {
            for (const room of rooms.values()) {
                if (!room.private && count(room) < game.maxClients) return room;
            }
            return createRoom(issueCode(), false);
        }
        if (intent === "join" && code) {
            const time = reservationNow();
            expireCodes(time);
            const key = code.toUpperCase();
            const room = rooms.get(key);
            const reservation = room?.reservation ?? unclaimed.get(key);
            if (
                reservation &&
                time < reservation.issuedAt + CREATOR_HOLD_MS &&
                creator !== reservation.creator
            ) {
                connection.close(4001, "First seat reserved for creator");
                return null;
            }
            if (!reservation || room) return room;
            unclaimed.delete(key);
            const claimed = createRoom(key, true);
            claimed.reservation = reservation;
            return claimed;
        }
    }

    function list() {
        return [...rooms.values()]
            .filter((room) => !room.private)
            .map((room) => ({
                code: room.code,
                clients: count(room),
                maxClients: game.maxClients,
            }));
    }

    function reserve() {
        if (stopped) return;
        const time = reservationNow();
        expireCodes(time);
        if (unclaimed.size >= MAX_UNCLAIMED_CODES) return;
        const code = issueCode();
        const creator = crypto.randomUUID();
        unclaimed.set(code, { creator, issuedAt: time });
        return { code, creator };
    }

    function connect(
        connection: C,
        {
            intent = null,
            code = null,
            creator = null,
            resume: token = null,
            seat: id = null,
            reconnectable = true,
        }: {
            intent?: string | null;
            code?: string | null;
            creator?: string | null;
            resume?: string | null;
            seat?: string | null;
            reconnectable?: boolean;
        } = {},
    ) {
        if (stopped) {
            connection.close(1001, "Server stopping");
            return;
        }
        expireSeats();
        if (token !== null || id !== null) {
            const seat = token === null ? undefined : held.get(token);
            const authority = seat?.room.app?.world.resource(ServerSession).server;
            if (
                !seat ||
                seat.id !== id ||
                (intent === "join" && code?.toUpperCase() !== seat.room.code) ||
                authority?.peers.get(seat.eid)?.connection !== seat.connection
            ) {
                connection.close(1008, "Invalid or expired reconnection token");
                return;
            }
            release(seat);
            seat.transport.stream = connection;
            seat.room.seats.set(connection, seat.eid);
            memberships.set(connection, seat.room);
            seats.set(connection, seat);
            try {
                authority.resume(seat.eid);
                seat.token = crypto.randomUUID();
                sendToken(connection, seat);
            } catch (error) {
                console.error(error);
                seats.delete(connection);
                connection.close(1011, "Resumption failed");
                try {
                    authority.disconnect(seat.eid);
                } finally {
                    leave(connection);
                }
            }
            return;
        }
        const room = pinned ?? selectRoom(connection, intent, code, creator);
        if (room === null) return;
        if (!room) {
            connection.close(1008, "Unknown room or invalid intent");
            return;
        }
        if (pinned) {
            for (const peer of room.seats.keys()) if (peer.closed) leave(peer);
        }
        if (count(room) >= game.maxClients) {
            refuseFull(connection);
            return;
        }
        room.seats.set(connection, undefined);
        if (reconnectable) resumable.add(connection);
        memberships.set(connection, room);
        if (room.app) admit(room, connection);
        else room.ready ??= start(room);
        return { room, ready: room.ready };
    }

    function sweep() {
        for (const room of rooms.values())
            for (const peer of room.seats.keys()) if (peer.closed) leave(peer);
        expireSeats();
    }

    function tick(beforeTick?: (app: App) => void) {
        for (const app of active) {
            beforeTick?.(app);
            app.world.tick();
        }
    }

    function stop() {
        if (stopped) return;
        stopped = true;
        unclaimed.clear();
        held.clear();
        for (const room of rooms.values()) retire(room, 1001, "Server stopping");
    }
    return { connect, leave, sweep, tick, stop, list, reserve };
}
