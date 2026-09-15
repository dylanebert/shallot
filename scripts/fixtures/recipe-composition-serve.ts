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
import { build, Body, pointerLockChanged, pressKey, readBody, releaseKey, run, swap } from "@dylanebert/shallot";
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
    state.step(1 / 60);
    const controls = [...document.querySelectorAll("[data-recipe-controls]")];
    if (controls.length !== 1) throw new Error("selected recipe did not mount exactly one control card");
    const rows = [...controls[0].querySelectorAll("[data-control-row]")].map((row) => {
        const cells = [...row.children].map((cell) => cell.textContent?.trim() ?? "");
        if (cells.length !== 2) throw new Error("control row was not two-column");
        for (const cell of row.children) {
            if (getComputedStyle(cell).color !== "rgb(255, 255, 255)") throw new Error("control text was not computed white");
        }
        return cells.join(" | ");
    });
    const expectedRows = ${JSON.stringify(recipe === "vehicle" ? ["W / S | Throttle", "A / D | Steer"] : ["WASD | Move", "MOUSE | Look", "SPACE | Jump"])};
    if (JSON.stringify(rows) !== JSON.stringify(expectedRows)) throw new Error("control rows were not exact: " + JSON.stringify(rows));
    const text = getComponent("Text");
    if (text && [...state.query([text])].length !== 0) throw new Error("selected scene still contains a Text entity");
    if (project.plugins.some((plugin) => plugin.name === "Text")) throw new Error("manifest still selected TextPlugin");
    if (${recipe === "first-person"}) {
        const status = controls[0].querySelector("[data-pointer-lock-status]");
        if (!status || !status.textContent?.includes("Click the scene to enable mouse look.")) throw new Error("unlocked pointer-lock status was missing");
        if (getComputedStyle(status).color !== "rgb(255, 255, 255)") throw new Error("unlocked pointer-lock status was not computed white");
        if (controls[0].textContent?.includes("Click: look")) throw new Error("stale Click: look copy remained");
        pointerLockChanged(state, true);
        state.step(1 / 60);
        if (!status.hidden) throw new Error("pointer-lock status remained visible while locked");
        pointerLockChanged(state, false, "fixture refusal");
        state.step(1 / 60);
        if (!status.textContent?.startsWith("Mouse look unavailable.") || !status.textContent.includes("fixture refusal")) throw new Error("pointer-lock refusal status was missing");
        if (getComputedStyle(status).color !== "rgb(255, 255, 255)") throw new Error("refused pointer-lock status was not computed white");
    }
    const retainedCard = controls[0];
    const previousPlugin = project.plugins.find((plugin) => plugin.name === ${JSON.stringify(expectedPlugin)});
    if (!previousPlugin) throw new Error("selected local plugin was missing before swap");
    let liftSample;
    if (${recipe === "first-person"}) {
        const liftEntity = [...state.entities()].find((eid) => state.identity.id(eid) === "lift");
        if (liftEntity === undefined) throw new Error("selected scene missed the lift entity");
        const baseY = Body.pos.y.get(liftEntity);
        const baseX = Body.pos.x.get(liftEntity);
        const baseZ = Body.pos.z.get(liftEntity);
        liftSample = () => {
            const pose = readBody(state, liftEntity);
            if (!pose) throw new Error("lift lost its live body across plugin replacement");
            const phase = state.time.elapsed * 0.65;
            const expectedPhase = (state.time.elapsed + 1 / 60) * 0.65;
            const expectedY = baseY + 0.5 * 1.5 * (1 - Math.cos(2 * expectedPhase));
            const expectedVy = 0.65 * 1.5 * Math.sin(2 * phase);
            if (
                Math.abs(pose.pos[1] - expectedY) > 0.002 ||
                Math.abs(pose.pos[0] - baseX) > 0.002 ||
                Math.abs(pose.pos[2] - baseZ) > 0.002 ||
                Math.abs(pose.vel[0]) > 0.002 ||
                Math.abs(pose.vel[2]) > 0.002 ||
                Math.abs(pose.vel[1] - expectedVy) > 0.02
            ) throw new Error("first-person lift left its uninterrupted authored-base trajectory");
        };
        for (let i = 0; i < 4; i++) {
            state.step(1 / 60);
            liftSample();
        }
    }
    const replacementModule = await import(${JSON.stringify("/@fs" + projectDir + (recipe === "vehicle" ? "/src/car.ts?recipe-swap-evaluation=1" : "/src/demo.ts?recipe-swap-evaluation=1"))});
    const replacementPlugin = replacementModule.default;
    if (!replacementPlugin || replacementPlugin === previousPlugin) throw new Error("replacement plugin was not separately evaluated");
    const previousSystems = previousPlugin.systems ?? [];
    const replacementSystems = replacementPlugin.systems ?? [];
    if (
        previousSystems.length !== replacementSystems.length ||
        previousSystems.some((system, index) => system.update === replacementSystems[index].update)
    ) throw new Error("replacement plugin did not provide distinct system closures");
    const replacementPlugins = project.plugins.map((plugin) =>
        plugin.name === ${JSON.stringify(expectedPlugin)} ? replacementPlugin : plugin,
    );
    const heldInitializers = project.plugins.map((plugin) => [plugin, plugin.initialize]);
    for (const plugin of project.plugins) {
        if (plugin.name !== ${JSON.stringify(expectedPlugin)}) plugin.initialize = undefined;
    }
    let swapResult;
    try {
        swapResult = await swap(state, project.plugins, replacementPlugins);
    } finally {
        for (const [plugin, initialize] of heldInitializers) plugin.initialize = initialize;
    }
    if (!swapResult.ok) throw new Error("same-shape recipe plugin swap was refused: " + swapResult.reason);
    if (${recipe === "vehicle"}) {
        pressKey(state, "KeyW");
        pressKey(state, "KeyA");
        state.step(1 / 60);
        const observed = replacementModule.readVehicle(state);
        if (!observed || observed.wheels.length !== 4) throw new Error("replacement Car could not observe the preserved vehicle runtime");
        const rear = observed.wheels.filter((wheel) => wheel.role === replacementModule.VehicleRole.RearWheel);
        const front = observed.wheels.filter((wheel) => wheel.role === replacementModule.VehicleRole.FrontWheel);
        if (rear.length !== 2 || rear.some((wheel) => Math.abs(wheel.spin.target + replacementModule.VEHICLE_CONFIG.throttle) > 0.00001)) throw new Error("replacement Car did not apply both W motor targets");
        if (front.length !== 2 || front.some((wheel) => Math.abs(wheel.steering.target - replacementModule.VEHICLE_CONFIG.steeringLock) > 0.00001)) throw new Error("replacement Car did not apply both A steering targets");
        releaseKey(state, "KeyW");
        releaseKey(state, "KeyA");
        state.step(1 / 60);
        const released = replacementModule.readVehicle(state);
        if (!released || released.wheels.some((wheel) => Math.abs(wheel.spin.target) > 0.00001 || Math.abs(wheel.steering.target) > 0.00001)) throw new Error("replacement Car left W+A targets active after release");
    } else {
        if (!liftSample) throw new Error("first-person lift sample was not installed");
        for (let i = 0; i < 4; i++) {
            state.step(1 / 60);
            liftSample();
        }
        const swappedStatus = controls[0].querySelector("[data-pointer-lock-status]");
        if (!swappedStatus || getComputedStyle(swappedStatus).color !== "rgb(255, 255, 255)") throw new Error("swapped pointer-lock status was not computed white");
    }
    const postSwapCard = document.querySelector("[data-recipe-controls]");
    if (postSwapCard !== retainedCard || document.querySelectorAll("[data-recipe-controls]").length !== 1) throw new Error("plugin swap remounted or duplicated the control card");
    app.dispose();
    if (document.querySelectorAll("[data-recipe-controls]").length !== 0) throw new Error("control card was not disposed with State");
    const recipeState = Symbol.for(${JSON.stringify(recipe === "vehicle" ? "shallot.examples.drive-a-vehicle.state" : "shallot.examples.first-person.state")});
    if (state[recipeState] !== undefined) throw new Error("recipe State bag survived disposal");
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
