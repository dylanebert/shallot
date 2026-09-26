// src/project/vite.ts
import { existsSync as existsSync3, readFileSync as readFileSync3 } from "fs";
import { dirname, isAbsolute, join as join3, relative as relative2, resolve } from "path";
import typegpu from "unplugin-typegpu/vite";

// src/project/assets.ts
import { existsSync, readFileSync, statSync } from "fs";
import { join, sep } from "path";

// src/project/engine.ts
var DEFAULT_PLUGIN_NAMES = [
  "Slab",
  "Transforms",
  "Input",
  "Render",
  "Part",
  "Sear",
  "Glaze"
];
var EXTRA_PLUGIN_NAMES = [
  "Audio",
  "Cells",
  "Character",
  "Fog",
  "Gltf",
  "Lines",
  "Mirror",
  "Orbit",
  "OrbitOverlay",
  "Outline",
  "Player",
  "Profile",
  "Skin",
  "Sky",
  "Sprite",
  "Text",
  "Physics"
];
var KNOWN_ENGINE_PLUGINS = new Set([
  ...DEFAULT_PLUGIN_NAMES,
  ...EXTRA_PLUGIN_NAMES
]);

// src/project/manifest.ts
function normalize(raw) {
  let parsed;
  try {
    parsed = JSON.parse(raw ?? "{}");
  } catch {
    return {};
  }
  if (typeof parsed !== "object" || parsed === null || Array.isArray(parsed))
    return {};
  const obj = parsed;
  const manifest = {};
  if (typeof obj.$schema === "string")
    manifest.$schema = obj.$schema;
  if (typeof obj.scene === "string")
    manifest.scene = obj.scene;
  if (typeof obj.plugins === "object" && obj.plugins !== null && !Array.isArray(obj.plugins)) {
    manifest.plugins = obj.plugins;
  }
  if (typeof obj.capacity === "number")
    manifest.capacity = obj.capacity;
  if (obj.pixelRatio === "auto" || typeof obj.pixelRatio === "number" && obj.pixelRatio > 0) {
    manifest.pixelRatio = obj.pixelRatio;
  }
  if (typeof obj.identifier === "string")
    manifest.identifier = obj.identifier;
  if (Array.isArray(obj.assets) && obj.assets.every((a) => typeof a === "string")) {
    manifest.assets = obj.assets;
  }
  return manifest;
}
function localOf(value) {
  if (typeof value === "string")
    return { spec: value, enabled: true };
  if (Array.isArray(value) && typeof value[0] === "string") {
    return { spec: value[0], enabled: value[1] !== false };
  }
  return null;
}

// src/project/assets.ts
function manifestPath(dir) {
  return join(dir, "shallot.json");
}
function manifestWarnings(raw, known) {
  let parsed;
  try {
    parsed = JSON.parse(raw);
  } catch {
    return ["not valid JSON, ignored (the project runs with default plugins)"];
  }
  const plugins = parsed?.plugins;
  if (typeof plugins !== "object" || plugins === null || Array.isArray(plugins))
    return [];
  const warnings = [];
  for (const [name, value] of Object.entries(plugins)) {
    if (typeof value === "boolean" && !known.has(name)) {
      warnings.push(`"${name}" is not a known engine plugin (use a module specifier for a local plugin)`);
    }
  }
  return warnings;
}
function readManifest(absDir) {
  const path = manifestPath(absDir);
  let text;
  try {
    text = readFileSync(path, "utf-8");
  } catch {
    return {};
  }
  for (const w of manifestWarnings(text, KNOWN_ENGINE_PLUGINS))
    console.warn(`  ! ${path}: ${w}`);
  return normalize(text);
}
var MIME = {
  svg: "image/svg+xml",
  png: "image/png",
  jpg: "image/jpeg",
  jpeg: "image/jpeg",
  webp: "image/webp",
  gif: "image/gif",
  ico: "image/x-icon",
  json: "application/json",
  scene: "text/plain; charset=utf-8",
  wasm: "application/wasm",
  glb: "model/gltf-binary",
  gltf: "model/gltf+json",
  bin: "application/octet-stream",
  ktx2: "image/ktx2"
};
function contentType(path) {
  return MIME[path.slice(path.lastIndexOf(".") + 1).toLowerCase()];
}
function resolveAssetPath(dir, pathname) {
  const filePath = join(dir, pathname);
  if (filePath !== dir && !filePath.startsWith(dir + sep))
    return null;
  return existsSync(filePath) && statSync(filePath).isFile() ? filePath : null;
}

// src/project/generate.ts
var ENGINE = "@dylanebert/shallot";
function generateModuleFromPlan(project) {
  const { dir, manifest, scenes, engine, locals } = project;
  const idents = engine.map((n) => `${n}Plugin`);
  const lines = [];
  if (idents.length > 0) {
    lines.push(`import { ${idents.join(", ")} } from ${JSON.stringify(ENGINE)};`);
  }
  for (let i = 0;i < locals.length; i++) {
    lines.push(`import _l${i} from ${JSON.stringify(locals[i].path)};`);
  }
  lines.push(`const engine = [${idents.join(", ")}];`);
  lines.push(`const locals = [${locals.map((l, i) => `{ name: ${JSON.stringify(l.name)}, plugin: _l${i} }`).join(", ")}];`);
  lines.push(`for (const l of locals) if (!l.plugin || typeof l.plugin.name !== "string") throw new Error("shallot.json plugin \\"" + l.name + "\\": its module must default-export a Plugin");`);
  lines.push(`const manifest = ${JSON.stringify(manifest)};`);
  lines.push(`const scenes = ${JSON.stringify(scenes)};`);
  lines.push(`const scene = ${JSON.stringify(manifest.scene ?? null)};`);
  lines.push(`const capacity = ${JSON.stringify(manifest.capacity ?? null)};`);
  lines.push(`const pixelRatio = ${JSON.stringify(manifest.pixelRatio ?? null)};`);
  lines.push(`const dir = ${JSON.stringify(dir)};`);
  lines.push(`const project = { dir, scene, capacity, pixelRatio, scenes, manifest, locals, plugins: [...engine, ...locals.map((l) => l.plugin)] };`);
  lines.push(`export default project;`);
  return lines.join(`
`);
}

// src/project/host.ts
import { existsSync as existsSync2, readdirSync, readFileSync as readFileSync2, statSync as statSync2 } from "node:fs";
import { join as join2, relative } from "node:path";
function localPath(spec, absDir) {
  return spec.startsWith(".") ? join2(absDir, spec) : spec;
}
function plan(manifest, absDir) {
  const plugins = manifest.plugins ?? {};
  const defaults = new Set(DEFAULT_PLUGIN_NAMES);
  const engine = [];
  const locals = [];
  const disabled = [];
  for (const name of DEFAULT_PLUGIN_NAMES) {
    if (plugins[name] !== false)
      engine.push(name);
    else
      disabled.push(name);
  }
  for (const [name, value] of Object.entries(plugins)) {
    if (defaults.has(name))
      continue;
    if (value === true)
      engine.push(name);
    else if (value === false)
      disabled.push(name);
    else {
      const local = localOf(value);
      if (!local)
        continue;
      if (local.enabled)
        locals.push({ name, spec: local.spec, path: localPath(local.spec, absDir ?? "") });
      else
        disabled.push(name);
    }
  }
  return { engine, locals, disabled };
}
function discoverScenes(dir) {
  const scenes = [];
  function walk(current) {
    let entries;
    try {
      entries = readdirSync(current);
    } catch (e) {
      console.warn(`  ! scene discovery: skipping unreadable directory "${current}": ${e}`);
      return;
    }
    for (const entry of entries) {
      if (entry === "node_modules" || entry === "dist")
        continue;
      const full = join2(current, entry);
      let isDirectory;
      try {
        isDirectory = statSync2(full).isDirectory();
      } catch (e) {
        console.warn(`  ! scene discovery: skipping unreadable entry "${full}": ${e}`);
        continue;
      }
      if (isDirectory)
        walk(full);
      else if (entry.endsWith(".scene"))
        scenes.push(relative(dir, full));
    }
  }
  walk(dir);
  return scenes.sort();
}
var REAL_IO = {
  readFile(path) {
    try {
      return readFileSync2(path, "utf-8");
    } catch {
      return null;
    }
  },
  discoverScenes
};
function readProject(dir, io = REAL_IO) {
  const manifest = io === REAL_IO ? readManifest(dir) : normalize(io.readFile(manifestPath(dir)));
  const scenes = io.discoverScenes(dir);
  return { dir, manifest, scenes, ...plan(manifest, dir) };
}
function emptyPlan() {
  return { dir: null, manifest: {}, scenes: [], ...plan({}, null) };
}

// src/project/vite.ts
var CROSS_ORIGIN_ISOLATION = {
  "Cross-Origin-Opener-Policy": "same-origin",
  "Cross-Origin-Embedder-Policy": "require-corp"
};
function typegpuPlugin() {
  return typegpu();
}
function pluginPackages(projectDir) {
  if (!projectDir)
    return [];
  const path = manifestPath(resolve(projectDir));
  let raw = null;
  try {
    raw = readFileSync3(path, "utf8");
  } catch {}
  const packages = plan(normalize(raw), resolve(projectDir)).locals.map(({ spec }) => spec).filter((spec) => !spec.startsWith(".") && !isAbsolute(spec)).map((spec) => spec.split("/").slice(0, spec.startsWith("@") ? 2 : 1).join("/"));
  return [...new Set(packages)];
}
function findPublicDirs(projectDir) {
  const dirs = [];
  const own = join3(projectDir, "public");
  if (existsSync3(own))
    dirs.push(own);
  const parent = join3(dirname(projectDir), "public");
  if (existsSync3(parent) && parent !== own)
    dirs.push(parent);
  return dirs;
}
var MODEL_EXT = /\.(glb|gltf)$/i;
function assetSrc(file, publicDirs) {
  if (!MODEL_EXT.test(file))
    return null;
  for (const dir of publicDirs) {
    const rel = relative2(dir, file);
    if (rel && !rel.startsWith("..") && !isAbsolute(rel))
      return rel.replace(/\\/g, "/");
  }
  return null;
}
function signalChange(server) {
  server.ws.send({ type: "full-reload" });
}
function configureServer(server, projectDir) {
  const publicDirs = projectDir ? findPublicDirs(resolve(projectDir)) : [];
  if (publicDirs.length === 0)
    return;
  server.middlewares.use((req, res, next) => {
    if (req.url) {
      const pathname = new URL(req.url, "http://localhost").pathname;
      for (const dir of publicDirs) {
        const filePath = resolveAssetPath(dir, pathname);
        if (!filePath)
          continue;
        const data = readFileSync3(filePath);
        const mime = contentType(filePath);
        if (mime)
          res.setHeader("Content-Type", mime);
        res.setHeader("Cache-Control", "no-store");
        res.end(data);
        return;
      }
    }
    next();
  });
}
function orphanedAssets(bundle) {
  const files = Object.values(bundle);
  const text = (f) => f.type === "chunk" ? f.code : typeof f.source === "string" ? f.source : "";
  const kept = new Set(files.filter((f) => f.type === "chunk" || f.fileName.endsWith(".html")));
  for (const file of files) {
    if (file.type !== "chunk")
      continue;
    const metadata = file;
    for (const css of metadata.viteMetadata?.importedCss ?? []) {
      const imported = bundle[css];
      if (imported?.type === "asset")
        kept.add(imported);
    }
  }
  const assets = files.filter((f) => f.type === "asset" && !f.fileName.endsWith(".html"));
  let grew = true;
  while (grew) {
    grew = false;
    for (const a of assets) {
      if (kept.has(a))
        continue;
      const name = a.fileName.slice(a.fileName.lastIndexOf("/") + 1);
      if ([...kept].some((k) => text(k).includes(name))) {
        kept.add(a);
        grew = true;
      }
    }
  }
  return assets.filter((a) => !kept.has(a)).map((a) => a.fileName);
}
function classifyProjectFile(file, absDir, publicDirs) {
  if (assetSrc(file, publicDirs))
    return "asset";
  if (file.startsWith(absDir) && (file.endsWith(".scene") || file === manifestPath(absDir)))
    return "project";
  return null;
}
function projectPlugin(projectDir) {
  const virtualId = "virtual:project";
  const resolvedId = "\x00" + virtualId;
  const sharedDependencies = [
    "@dylanebert/shallot",
    "typegpu",
    ...pluginPackages(projectDir)
  ];
  let viteServer;
  let publicDirs = [];
  return {
    name: "shallot-project",
    config() {
      return {
        resolve: { dedupe: sharedDependencies },
        optimizeDeps: { exclude: sharedDependencies }
      };
    },
    async resolveId(id, importer) {
      if (id === virtualId)
        return resolvedId;
      if (importer === resolvedId && projectDir) {
        const r = await this.resolve(id, join3(resolve(projectDir), "__project__.js"), {
          skipSelf: true
        });
        if (r)
          return r;
      }
    },
    load(id) {
      if (id !== resolvedId)
        return;
      if (!projectDir)
        return generateModuleFromPlan(emptyPlan());
      return generateModuleFromPlan(readProject(resolve(projectDir)));
    },
    configureServer(server) {
      viteServer = server;
      configureServer(server, projectDir);
      if (projectDir) {
        const absDir = resolve(projectDir);
        publicDirs = findPublicDirs(absDir);
        server.watcher.add(absDir);
        for (const pub of publicDirs)
          if (!pub.startsWith(absDir))
            server.watcher.add(pub);
        const onProjectFile = (file) => {
          const kind = classifyProjectFile(file, absDir, publicDirs);
          if (kind === "asset") {
            signalChange(server);
            return;
          }
          if (kind === "project") {
            const mod = server.moduleGraph.getModuleById(resolvedId);
            if (mod)
              server.moduleGraph.invalidateModule(mod);
            signalChange(server);
          }
        };
        server.watcher.on("change", onProjectFile);
        server.watcher.on("add", onProjectFile);
        server.watcher.on("unlink", onProjectFile);
      }
    },
    handleHotUpdate({ file }) {
      if (!projectDir || !viteServer)
        return;
      const absDir = resolve(projectDir);
      if (classifyProjectFile(file, absDir, publicDirs) === "project") {
        const mod = viteServer.moduleGraph.getModuleById(resolvedId);
        if (mod)
          viteServer.moduleGraph.invalidateModule(mod);
        return [];
      }
    },
    generateBundle(_options, bundle) {
      const orphans = orphanedAssets(bundle);
      if (!orphans.length)
        return;
      let bytes = 0;
      for (const fileName of orphans) {
        const a = bundle[fileName];
        if (a?.type === "asset")
          bytes += typeof a.source === "string" ? a.source.length : a.source.byteLength;
        delete bundle[fileName];
      }
      this.info(`pruned ${orphans.length} orphaned asset(s), ${bytes / 1024 | 0}KB`);
    }
  };
}
export {
  CROSS_ORIGIN_ISOLATION,
  assetSrc,
  classifyProjectFile,
  discoverScenes,
  findPublicDirs,
  manifestPath,
  manifestWarnings,
  orphanedAssets,
  pluginPackages,
  projectPlugin,
  typegpuPlugin
};
