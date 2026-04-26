import type { Plugin, ViteDevServer } from "vite";
import type { IncomingMessage, ServerResponse } from "http";
import { readFileSync, writeFileSync, existsSync, readdirSync, statSync } from "fs";
import { resolve, join, dirname, relative } from "path";

function libPath(dir: string): string {
    if (existsSync(join(dir, "src", "lib.ts"))) return join(dir, "src", "lib");
    return join(dir, "src", "lib", "index");
}

function discoverScenes(dir: string): string[] {
    const scenes: string[] = [];
    function walk(current: string) {
        for (const entry of readdirSync(current)) {
            if (entry === "node_modules" || entry === "dist") continue;
            const full = join(current, entry);
            if (statSync(full).isDirectory()) walk(full);
            else if (entry.endsWith(".scene")) scenes.push(relative(dir, full));
        }
    }
    try {
        walk(dir);
    } catch {}
    return scenes.sort();
}

function findPublicDirs(projectDir: string): string[] {
    const dirs: string[] = [];
    const own = join(projectDir, "public");
    if (existsSync(own)) dirs.push(own);
    const parent = join(dirname(projectDir), "public");
    if (existsSync(parent) && parent !== own) dirs.push(parent);
    return dirs;
}

function readBody(req: IncomingMessage): Promise<string> {
    return new Promise((resolve) => {
        let body = "";
        req.on("data", (chunk: Buffer) => {
            body += chunk.toString();
        });
        req.on("end", () => resolve(body));
    });
}

function json(res: ServerResponse, data: unknown) {
    res.setHeader("Content-Type", "application/json");
    res.end(JSON.stringify(data));
}

function error(res: ServerResponse, status: number, message: string) {
    res.statusCode = status;
    json(res, { error: message });
}

const recentSaves = new Set<string>();

function configureServer(server: ViteDevServer, projectDir?: string) {
    const publicDirs: string[] = projectDir ? findPublicDirs(resolve(projectDir)) : [];

    const logger = server.config.logger;
    const origInfo = logger.info.bind(logger);
    const origWarn = logger.warn.bind(logger);
    const origError = logger.error.bind(logger);
    logger.info = (msg, opts) => {
        origInfo(msg, opts);
        server.ws.send("shallot:log", { level: "info", message: msg });
    };
    logger.warn = (msg, opts) => {
        origWarn(msg, opts);
        server.ws.send("shallot:log", { level: "warn", message: msg });
    };
    logger.error = (msg, opts) => {
        origError(msg, opts);
        server.ws.send("shallot:log", { level: "error", message: msg });
    };

    const pending = new Map<string, (result: unknown) => void>();
    let nextId = 0;
    server.ws.on("shallot:response", (data: { id: string; result: unknown }) => {
        const resolve = pending.get(data.id);
        if (resolve) {
            pending.delete(data.id);
            resolve(data.result);
        }
    });

    function request(type: string, payload?: unknown): Promise<unknown> {
        const id = String(++nextId);
        return new Promise((resolve, reject) => {
            const timeout = setTimeout(() => {
                pending.delete(id);
                reject(new Error("No editor connected"));
            }, 5000);
            pending.set(id, (result) => {
                clearTimeout(timeout);
                resolve(result);
            });
            server.ws.send("shallot:request", { id, type, payload });
        });
    }

    server.middlewares.use((req, res, next) => {
        const url = new URL(req.url || "", "http://localhost");

        if (url.pathname === "/__api/scene" && req.method === "GET") {
            const dir = url.searchParams.get("dir");
            const path = url.searchParams.get("path");
            if (!dir || !path) {
                res.statusCode = 400;
                res.end("Missing dir or path");
                return;
            }
            const absDir = resolve(dir);
            const absPath = resolve(absDir, path);
            if (!absPath.startsWith(absDir)) {
                res.statusCode = 403;
                res.end("Path traversal");
                return;
            }
            try {
                const content = readFileSync(absPath, "utf-8");
                res.setHeader("Content-Type", "text/plain");
                res.end(content);
            } catch {
                res.statusCode = 404;
                res.end("Scene not found");
            }
            return;
        }

        if (url.pathname === "/__api/scene" && req.method === "POST") {
            const dir = url.searchParams.get("dir");
            const path = url.searchParams.get("path");
            if (!dir || !path) {
                res.statusCode = 400;
                res.end("Missing dir or path");
                return;
            }
            const absDir = resolve(dir);
            const absPath = resolve(absDir, path);
            if (!absPath.startsWith(absDir)) {
                res.statusCode = 403;
                res.end("Path traversal");
                return;
            }
            readBody(req).then((body) => {
                try {
                    const { content } = JSON.parse(body);
                    recentSaves.add(absPath);
                    setTimeout(() => recentSaves.delete(absPath), 2000);
                    writeFileSync(absPath, content, "utf-8");
                    res.setHeader("Content-Type", "application/json");
                    res.end(JSON.stringify({ ok: true }));
                } catch (e) {
                    res.statusCode = 500;
                    res.end(String(e));
                }
            });
            return;
        }

        if (url.pathname === "/__api/state" && req.method === "GET") {
            request("state")
                .then((r) => json(res, r))
                .catch((e) => error(res, 503, e.message));
            return;
        }

        if (url.pathname === "/__api/command" && req.method === "POST") {
            readBody(req)
                .then((body) => request("command", JSON.parse(body)))
                .then((r) => json(res, r))
                .catch((e) => error(res, 400, e.message));
            return;
        }

        if (url.pathname === "/__api/entities" && req.method === "GET") {
            const component = url.searchParams.get("component");
            const eid = url.searchParams.get("eid");
            request("entities", { component, eid: eid ? Number(eid) : null })
                .then((r) => json(res, r))
                .catch((e) => error(res, 503, e.message));
            return;
        }

        if (publicDirs.length > 0 && req.url && !req.url.startsWith("/__api")) {
            const pathname = new URL(req.url, "http://localhost").pathname;
            for (const dir of publicDirs) {
                const filePath = join(dir, pathname);
                if (!filePath.startsWith(dir)) continue;
                if (existsSync(filePath) && statSync(filePath).isFile()) {
                    const data = readFileSync(filePath);
                    res.end(data);
                    return;
                }
            }
        }

        next();
    });
}

const DISCOVER_PREAMBLE = `
function _isPlugin(x) {
    if (!x || typeof x !== "object" || Array.isArray(x)) return false;
    return "systems" in x || "components" in x || "initialize" in x || "warm" in x
        || "relations" in x || "dependencies" in x;
}
function _isConfig(x) {
    if (!x || typeof x !== "object" || Array.isArray(x)) return false;
    return "plugins" in x && Array.isArray(x.plugins);
}
function _discover(mod) {
    return Object.entries(mod)
        .filter(([k, v]) => k !== "default" && _isPlugin(v))
        .map(([name, plugin]) => ({ name, plugin }));
}
`;

export function projectPlugin(projectDir?: string): Plugin {
    const virtualId = "virtual:project";
    const resolvedId = "\0" + virtualId;
    let viteServer: ViteDevServer | undefined;

    return {
        name: "shallot-project",
        resolveId(id) {
            if (id === virtualId) return resolvedId;
        },
        load(id) {
            if (id !== resolvedId) return;
            if (!projectDir) {
                return `export default { custom: [], scenes: [], dir: null, config: null };`;
            }

            const absDir = resolve(projectDir);
            const lib = libPath(absDir);
            const scenes = discoverScenes(absDir);

            return [
                DISCOVER_PREAMBLE,
                `import * as _lib from "${lib}";`,
                `const custom = _discover(_lib);`,
                `const config = "config" in _lib && _isConfig(_lib.config) ? _lib.config : null;`,
                `export default { custom, scenes: ${JSON.stringify(scenes)}, dir: ${JSON.stringify(absDir)}, config };`,
            ].join("\n");
        },
        configureServer(server) {
            viteServer = server;
            configureServer(server, projectDir);
            if (projectDir) {
                const absDir = resolve(projectDir);
                server.watcher.add(absDir);
                const onSceneChange = (file: string) => {
                    if (!file.startsWith(absDir) || !file.endsWith(".scene")) return;
                    const mod = server.moduleGraph.getModuleById(resolvedId);
                    if (mod) server.moduleGraph.invalidateModule(mod);
                    if (recentSaves.has(file)) return;
                    server.ws.send({ type: "full-reload" });
                };
                server.watcher.on("change", onSceneChange);
                server.watcher.on("add", onSceneChange);
                server.watcher.on("unlink", onSceneChange);
            }
        },
        handleHotUpdate({ file }) {
            if (!projectDir || !viteServer) return;
            const absDir = resolve(projectDir);
            const isScene = file.endsWith(".scene");
            if (isScene && file.startsWith(absDir)) {
                const mod = viteServer.moduleGraph.getModuleById(resolvedId);
                if (mod) viteServer.moduleGraph.invalidateModule(mod);
                if (recentSaves.has(file)) return [];
                viteServer.ws.send({ type: "full-reload" });
                return [];
            }
            if (
                file.startsWith(absDir) &&
                relative(absDir, file).startsWith("src/") &&
                /\.[tj]sx?$/.test(file)
            ) {
                const mod = viteServer.moduleGraph.getModuleById(resolvedId);
                if (mod) viteServer.moduleGraph.invalidateModule(mod);
                viteServer.ws.send({ type: "full-reload" });
                return [];
            }
        },
    };
}
