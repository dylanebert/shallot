import { existsSync } from "node:fs";
import { resolve } from "node:path";

/**
 * @param {string} messages
 * @param {string} marker
 * @param {(path: string) => string} [translatePath]
 * @returns {string | null}
 */
export function cargoRuntimeDir(messages, marker, translatePath = (path) => path) {
    let found = null;
    for (const line of messages.split(/\r?\n/)) {
        if (!line.startsWith("{")) continue;
        /** @type {Record<string, unknown>} */
        let message;
        try {
            message = JSON.parse(line);
        } catch {
            continue;
        }
        if (
            message.reason !== "build-script-executed" ||
            typeof message["package_id"] !== "string" ||
            !message["package_id"].includes("shallot-native")
        )
            continue;
        for (const pair of /** @type {[string, string][]} */ (message.env ?? [])) {
            const [key, path] = pair;
            if (key !== "SHALLOT_CEF_DIR") continue;
            const translated = translatePath(path);
            if (existsSync(resolve(translated, marker))) found = translated;
        }
    }
    return found;
}

/** @param {string | null | undefined} path @param {string} marker @returns {string} */
export function requireRuntimeDir(path, marker) {
    if (path && existsSync(resolve(path, marker))) return path;
    throw new Error(`selected native build has no CEF runtime provenance containing ${marker}`);
}
