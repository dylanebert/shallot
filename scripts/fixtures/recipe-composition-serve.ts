import { resolve } from "node:path";
import { createServer, searchForWorkspaceRoot, type ViteDevServer } from "vite";
import { CROSS_ORIGIN_ISOLATION, projectPlugin, typegpuPlugin } from "../../src/project/vite";

const args = Bun.argv.slice(2);
const port = Number(args[args.indexOf("--port") + 1]);
const projectDir = resolve(args[args.indexOf("--project") + 1] ?? "");
const recipe = args[args.indexOf("--recipe") + 1];
if (
    !Number.isInteger(port) ||
    port <= 0 ||
    !projectDir ||
    (recipe !== "vehicle" && recipe !== "first-person")
) {
    console.error(
        "recipe composition serve refused: --port <number> --project <dir> --recipe <vehicle|first-person> are required",
    );
    process.exit(1);
}

const expectedPlugin = recipe === "vehicle" ? "Car" : "Demo";
const expectedIds =
    recipe === "vehicle"
        ? ["ground", "chassis", "front-right", "front-left", "rear-right", "rear-left"]
        : ["eye", "player", "lift"];
const roleCheck =
    recipe === "vehicle"
        ? `const role = getComponent("Vehicle");
           const roleEntities = role ? [...state.query([role])] : [];
           if (roleEntities.length !== 6 || !roleEntities.every((eid) => state.has(eid, body))) throw new Error("Vehicle role/body composition was incomplete");`
        : `const lift = getComponent("Lift");
           const liftEntities = lift ? [...state.query([lift])] : [];
           if (liftEntities.length !== 1 || !state.has(liftEntities[0], body)) throw new Error("Lift role/body composition was incomplete");`;
const html = `<!doctype html><meta charset="utf-8"><style>html,body{margin:0}canvas{display:block;width:100vw;height:100vh}</style><canvas id="canvas" width="1280" height="720"></canvas><script type="module">
import { build, Body, run } from "@dylanebert/shallot";
import { getComponent } from "@dylanebert/shallot/ecs";
import project from "virtual:project";
let verdict;
try {
    if (!project.scene || !project.plugins.some((plugin) => plugin.name === ${JSON.stringify(expectedPlugin)})) throw new Error("manifest did not select the expected scene/local plugin");
    const app = await build({ plugins: project.plugins, scene: project.scene, defaults: false });
    const state = app.state;
    const ids = ${JSON.stringify(expectedIds)};
    for (const id of ids) if ([...state.entities()].every((eid) => state.identity.id(eid) !== id)) throw new Error("selected scene missed authored id " + id);
    const body = getComponent("Body");
    if (!body) throw new Error("Physics did not register Body");
    ${roleCheck}
    app.dispose();
    const childrenBeforeRunFailure = document.body.children.length;
    let runSetupThrew = false;
    try {
        await run({
            plugins: project.plugins,
            scene: project.scene,
            defaults: false,
            ui: () => { throw new Error("intentional run UI setup failure"); },
        });
    } catch (error) {
        runSetupThrew = String(error).includes("intentional run UI setup failure");
    }
    if (!runSetupThrew) throw new Error("run() did not surface its throwing UI setup hook");
    if (document.body.children.length !== childrenBeforeRunFailure) throw new Error("run() left its overlay mounted after setup failure");
    const recovered = await build({ plugins: project.plugins, scene: project.scene, defaults: false });
    recovered.dispose();
    verdict = { ok: true, checks: [{ name: "exact manifest composition", ok: true, detail: "manifest build, throwing run setup cleanup, overlay unwind, and sequential recovery completed" }], noRender: true };
} catch (error) {
    verdict = { ok: false, checks: [{ name: "exact manifest composition", ok: false, detail: String(error) }], noRender: true };
}
window.__harness = { ready: true, noRender: true, run: async () => verdict };
</script>`;

const root = resolve(import.meta.dir, "../..");
const indexPlugin = {
    name: "recipe-composition-index",
    configureServer(server: ViteDevServer) {
        server.middlewares.use(async (request, response, next) => {
            const path = request.url?.split("?")[0];
            if (path !== "/" && path !== "/index.html") return next();
            const transformed = await server.transformIndexHtml(request.url ?? "/", html);
            response.statusCode = 200;
            response.setHeader("Content-Type", "text/html");
            response.end(transformed);
        });
    },
};
const server = await createServer({
    root: projectDir,
    configFile: false,
    plugins: [typegpuPlugin(), indexPlugin, projectPlugin(projectDir)],
    server: {
        port,
        strictPort: true,
        headers: CROSS_ORIGIN_ISOLATION,
        fs: { allow: [searchForWorkspaceRoot(projectDir), projectDir, root] },
    },
    build: { target: "esnext" },
});
await server.listen();
console.log(`recipe composition serve listening on ${port}`);
