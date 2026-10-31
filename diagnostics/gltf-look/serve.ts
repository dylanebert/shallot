/*
Prepare an external glTF-Sample-Viewer checkout at viewerPin with its
assets/environments submodule initialized at environmentPin. Run npm install
and npm run build there. Fetch footprint_court.hdr through git-lfs, or:
https://media.githubusercontent.com/media/KhronosGroup/glTF-Sample-Environments/b710a9fe6bcdfb8965530ede36e8279c298261de/footprint_court.hdr

GLTF_SAMPLE_VIEWER=<checkout> bunx playwright test -c examples/gltf/capture.config.ts
GLTF_LOOK_SCENE=Box restricts iteration to one scene. A full run writes every
pair and a category-ordered index to ~/gltf-look/<eight-character commit>/.
Use a clean committed source tree. run.json records cameras, pins and adapters.

Sample Viewer v1.0.10 supplies its default Footprint Court environment and
bounds-fitted camera. Both sides load the example's synced files. Shallot keeps
its own lighting and tonemapping; the ground and truck placement are omitted.
No visual equivalence is asserted. These uncompressed scenes need neither
Draco nor KTX initialization. The viewer checkout never enters the package.
*/
import { resolve, sep } from "node:path";
import { fileURLToPath } from "node:url";

import { environmentPin, viewerPin } from "./pin";
import viewerHtml from "./viewer";
const checkout = process.env.GLTF_SAMPLE_VIEWER;
if (!checkout) throw new Error("set GLTF_SAMPLE_VIEWER to a built glTF-Sample-Viewer v1.0.10 checkout");
const revision = Bun.spawnSync(["git", "-C", checkout, "rev-parse", "HEAD"]);
if (revision.stdout.toString().trim() !== viewerPin) throw new Error(`Sample Viewer must be ${viewerPin}`);
const environment = Bun.spawnSync(["git", "-C", `${checkout}/assets/environments`, "rev-parse", "HEAD"]);
if (environment.stdout.toString().trim() !== environmentPin)
    throw new Error("initialize the pinned Sample Viewer environment submodule");
const pointer = Bun.spawnSync(["git", "-C", `${checkout}/assets/environments`, "show", "HEAD:footprint_court.hdr"]).stdout.toString();
const expected = /oid sha256:([a-f0-9]+)/.exec(pointer)?.[1];
const hdr = Bun.file(`${checkout}/assets/environments/footprint_court.hdr`);
const actual = new Bun.CryptoHasher("sha256").update(await hdr.arrayBuffer()).digest("hex");
if (!expected || actual !== expected) throw new Error("fetch the pinned footprint_court.hdr LFS object before capture");
if (!(await Bun.file(`${checkout}/dist/gltf-viewer.module.js`).exists()))
    throw new Error("build the pinned Sample Viewer with npm install && npm run build");
Bun.serve({
    hostname: "127.0.0.1",
    port: 4176,
    async fetch(request) {
        const path = decodeURIComponent(new URL(request.url).pathname);
        const root = resolve(path.startsWith("/models/") ? fileURLToPath(new URL("../../examples/gltf/public/", import.meta.url)) : checkout);
        const relative = path.startsWith("/models/") ? path.replace("/models/", "/") : path;
        const file = resolve(root, `.${relative}`);
        if (path === "/") return new Response(viewerHtml, { headers: { "content-type": "text/html" } });
        if (!file.startsWith(root + sep)) return new Response(null, { status: 403 });
        const asset = Bun.file(file);
        return (await asset.exists()) ? new Response(asset) : new Response(null, { status: 404 });
    },
});
