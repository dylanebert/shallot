import { type Plugin, runApp, ScreenSpaceAmbientOcclusionPlugin } from "@dylanebert/shallot";
import { probeTexture } from "../../src/engine/runtime";
import { ssaoState } from "../../src/standard/rendering/ssao";
import { EXAMPLE_CONTROLS, PALETTES, type PaletteName } from "./palette";
import { DIAGNOSTIC_PLUGINS, SCENES, type SceneDefinition } from "./scenes";
import { addAmbientOcclusion } from "./ssao";

const query = new URLSearchParams(location.search);
const sceneId = query.get("scene") ?? SCENES[0]!.id;
const paletteName = (query.get("palette") ?? "aperture-carbon") as PaletteName;
const selectedScene = SCENES.find((candidate) => candidate.id === sceneId);
const palette = PALETTES[paletteName];
if (!selectedScene) throw new Error(`Unknown default-look scene: ${sceneId}`);
if (!palette) throw new Error(`Unknown default-look palette: ${paletteName}`);
const scene: SceneDefinition = selectedScene;

function mountUi(container: HTMLElement): () => void {
    const panel = document.createElement("section");
    panel.dataset.defaultLookUi = "";
    panel.style.cssText =
        "position:absolute;right:24px;top:24px;width:360px;padding:16px;" +
        "display:grid;gap:8px;border:1px solid;border-radius:2px;" +
        `font:400 13px/20px ${EXAMPLE_CONTROLS.tokens.sans};`;
    panel.style.background = EXAMPLE_CONTROLS.dark.bg2;
    panel.style.borderColor = EXAMPLE_CONTROLS.dark.line;
    panel.style.color = EXAMPLE_CONTROLS.dark.ink;

    const eyebrow = document.createElement("div");
    eyebrow.textContent = `DEFAULT LOOK / ${paletteName.toUpperCase()}`;
    eyebrow.style.color = EXAMPLE_CONTROLS.dark.muted;
    eyebrow.style.font = `400 12px/20px ${EXAMPLE_CONTROLS.tokens.mono}`;
    eyebrow.style.letterSpacing = ".08em";
    const heading = document.createElement("strong");
    heading.textContent = scene.title;
    heading.style.font = `700 15px/20px ${EXAMPLE_CONTROLS.tokens.mono}`;
    const description = document.createElement("div");
    description.textContent = scene.description;
    panel.append(eyebrow, heading, description);
    container.append(panel);
    return () => panel.remove();
}

const plugins = [
    ...DIAGNOSTIC_PLUGINS,
    ...(scene.plugins ?? []),
    ScreenSpaceAmbientOcclusionPlugin,
];
const scenePlugin: Plugin = {
    name: `DefaultLook/${scene.id}`,
    dependencies: plugins,
    initialize(world) {
        scene.build(world, palette);
        addAmbientOcclusion(world, query.get("ssao") ?? "Off");
    },
};

const app = await runApp({
    plugins: [scenePlugin],
    pixelRatio: 1,
    ui: scene.ui ? mountUi : undefined,
});
export async function captureAo(): Promise<string> {
    const camera = app.world.resource(ssaoState).cameras.values().next().value;
    if (!camera) throw new Error("No SSAO output to capture");
    const probe = await probeTexture(app.world, camera.textures.at(-1)!);
    const bytes = new Uint8Array(probe.bytes);
    let binary = "";
    for (let offset = 0; offset < bytes.length; offset += 65536) {
        binary += String.fromCharCode(...bytes.subarray(offset, offset + 65536));
    }
    return btoa(binary);
}

window.addEventListener("pagehide", () => app.dispose(), { once: true });
document.body.dataset.ready = `${paletteName}/${scene.id}`;
