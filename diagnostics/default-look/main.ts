import { type Plugin, runApp } from "@dylanebert/shallot";
import { PALETTES, type PaletteName } from "./palette";
import { DIAGNOSTIC_PLUGINS, SCENES, type SceneDefinition } from "./scenes";

const query = new URLSearchParams(location.search);
const sceneId = query.get("scene") ?? SCENES[0]!.id;
const paletteName = (query.get("palette") ?? "current") as PaletteName;
const selectedScene = SCENES.find((candidate) => candidate.id === sceneId);
const palette = PALETTES[paletteName];
if (!selectedScene) throw new Error(`Unknown default-look scene: ${sceneId}`);
if (!palette) throw new Error(`Unknown default-look palette: ${paletteName}`);
const scene: SceneDefinition = selectedScene;

function mountUi(container: HTMLElement): () => void {
    const panel = document.createElement("section");
    panel.dataset.defaultLookUi = "";
    panel.style.cssText =
        "position:absolute;right:32px;top:32px;width:248px;padding:18px 20px;" +
        "display:grid;gap:10px;border:1px solid;border-radius:6px;backdrop-filter:blur(8px);" +
        "font:12px/1.45 ui-monospace,SFMono-Regular,Menlo,Consolas,monospace;";
    panel.style.background = palette.ui.panel;
    panel.style.borderColor = palette.ui.border;
    panel.style.color = palette.ui.ink;

    const eyebrow = document.createElement("div");
    eyebrow.textContent = `DEFAULT LOOK / ${paletteName.toUpperCase()}`;
    eyebrow.style.opacity = "0.65";
    const heading = document.createElement("strong");
    heading.textContent = scene.title;
    heading.style.fontSize = "15px";
    const description = document.createElement("div");
    description.textContent = scene.description;
    description.style.opacity = "0.8";
    panel.append(eyebrow, heading, description);
    container.append(panel);
    return () => panel.remove();
}

const plugins = [...DIAGNOSTIC_PLUGINS, ...(scene.plugins ?? [])];
const scenePlugin: Plugin = {
    name: `DefaultLook/${scene.id}`,
    dependencies: plugins,
    initialize(world) {
        scene.build(world, palette);
    },
};

const app = await runApp({
    plugins: [scenePlugin],
    pixelRatio: 1,
    ui: scene.ui ? mountUi : undefined,
});
window.addEventListener("pagehide", () => app.dispose(), { once: true });
document.body.dataset.ready = `${paletteName}/${scene.id}`;
