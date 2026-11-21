import { encodeText, readText, writeText } from "../../engine/ecs";
import { requireCanonical } from "./canonical";
import { MAX_MESSAGE_BYTES } from "./link";

export interface ServerMessage {
    /** Game-defined unsigned 32-bit topic. */
    topic: number;
    /** Optional UTF-8 payload, at most 1024 bytes. */
    text?: string;
}

export const ServerMessageKind = 8;
const TextCapacity = 1024;

/** @internal Little-endian u32 kind and topic, optionally followed by u16 size and UTF-8. */
export function encodeServerMessage({ topic, text }: ServerMessage): Uint8Array {
    if (!Number.isInteger(topic) || topic < 0 || topic > 0xffffffff)
        throw new Error("Server message topic must be a u32");
    const payload = text === undefined ? undefined : encodeText(text, TextCapacity);
    const bytes = new Uint8Array(8 + (payload ? 2 + payload.length : 0));
    const view = new DataView(bytes.buffer);
    view.setUint32(0, ServerMessageKind, true);
    view.setUint32(4, topic, true);
    if (payload) writeText(view, 8, payload);
    return bytes;
}

/** @internal Refuse any bytes the encoder would not produce. */
export function decodeOneOffMessage(bytes: Uint8Array, view: DataView): ServerMessage {
    if (bytes.length < 8 || bytes.length > MAX_MESSAGE_BYTES)
        throw new Error("Invalid server message size");
    const topic = view.getUint32(4, true);
    const message =
        bytes.length === 8 ? { topic } : { topic, text: readText(view, 8, TextCapacity) };
    requireCanonical(bytes, encodeServerMessage(message));
    return message;
}
