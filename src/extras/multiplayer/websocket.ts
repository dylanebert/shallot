import { stampReceipt } from "./diagnostics";
import type { Connection } from "./link";

/** One socket lifetime. Reconnect with a new socket and new replication/intent histories;
 * the server's first record supplies the new late-join image. No messages cross lifetimes. */
export class WebSocketConnection implements Connection {
    private _messages: Uint8Array[] = [];
    private _closed = false;
    readonly socket: WebSocket;

    constructor(socket: WebSocket, control?: (message: string) => boolean) {
        this.socket = socket;
        socket.binaryType = "arraybuffer";
        socket.addEventListener("close", () => this.close());
        socket.addEventListener("message", (event) => {
            if (this.closed) return;
            if (typeof event.data === "string" && control?.(event.data)) return;
            if (!(event.data instanceof ArrayBuffer)) {
                socket.close(1003, "Binary messages required");
                return;
            }
            this._messages.push(stampReceipt(new Uint8Array(event.data), performance.now()));
        });
    }

    get endpoint(): string {
        return this.socket.url;
    }

    get closed(): boolean {
        return this._closed || this.socket.readyState >= 2;
    }

    close(): void {
        if (this._closed) return;
        this._closed = true;
        this._messages = [];
        this.socket.close();
    }

    /** Refuses writes before open or after close; the application owns retry and lifetime. */
    send(bytes: Uint8Array): void {
        if (this.closed || this.socket.readyState !== 1) throw new Error("WebSocket is not open");
        this.socket.send(bytes.slice());
    }

    /** Drains every queued frame in arrival order, including a stall's entire backlog. */
    receive(): Uint8Array[] {
        const messages = this._messages;
        this._messages = [];
        return messages;
    }
}
