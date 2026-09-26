import { existsSync, statSync } from "node:fs";
import { resolve, sep } from "node:path";
/** Serve a built page with the isolation headers the app's Vite entry provides. */
export function servePage(
    outDir: string,
    isolationHeaders: Record<string, string>,
): {
    server: ReturnType<typeof Bun.serve>;
    origin: string;
} {
    const root = resolve(outDir);
    const server = Bun.serve({
        port: 0,
        fetch(request) {
            const pathname = decodeURIComponent(new URL(request.url).pathname);
            const file = resolve(root, pathname === "/" ? "index.html" : `.${pathname}`);
            if (
                !file.startsWith(`${root}${sep}`) ||
                !existsSync(file) ||
                !statSync(file).isFile()
            ) {
                return new Response("not found", { status: 404 });
            }
            const body = Bun.file(file);
            return new Response(body, {
                headers: { ...isolationHeaders, "Content-Type": body.type },
            });
        },
    });
    return { server, origin: `http://localhost:${server.port}` };
}
