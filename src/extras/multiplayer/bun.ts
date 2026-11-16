import type { ServerWebSocket } from "bun";
import type { Connection } from "./link";

/** A single accepted peer. The application owns its replication history and closes its socket. */
export class ServerWebSocketConnection implements Connection {
    private _messages: Uint8Array[] = [];
    readonly socket: ServerWebSocket<ServerWebSocketConnection | undefined>;

    constructor(socket: ServerWebSocket<ServerWebSocketConnection | undefined>) {
        this.socket = socket;
    }

    /** Bun owns queued writes; a dropped write closes the peer rather than silently losing bytes. */
    send(bytes: Uint8Array): void {
        if (this.socket.readyState !== 1) throw new Error("WebSocket is not open");
        if (this.socket.sendBinary(bytes) === 0) this.socket.close(1011, "Write failed");
    }

    receive(): Uint8Array[] {
        const messages = this._messages;
        this._messages = [];
        return messages;
    }

    /** @internal */
    push(bytes: Uint8Array): void {
        this._messages.push(bytes.slice());
    }
}

/** Self-hosted binary endpoint. Every upgrade gets a distinct connection; callbacks run once
 * at open and close. Authentication and routing belong in the application/reverse proxy. */
export function serveWebSocket(options: {
    hostname?: string;
    port: number;
    connect: (connection: ServerWebSocketConnection) => void;
    disconnect: (connection: ServerWebSocketConnection) => void;
}) {
    return Bun.serve<ServerWebSocketConnection | undefined>({
        hostname: options.hostname,
        port: options.port,
        fetch(request, server) {
            if (server.upgrade(request, { data: undefined })) return;
            return new Response("WebSocket upgrade required", { status: 426 });
        },
        websocket: {
            open(socket) {
                const connection = new ServerWebSocketConnection(socket);
                socket.data = connection;
                options.connect(connection);
            },
            message(socket, message) {
                if (typeof message === "string") {
                    socket.close(1003, "Binary messages required");
                    return;
                }
                socket.data!.push(message);
            },
            close(socket) {
                options.disconnect(socket.data!);
            },
        },
    });
}
