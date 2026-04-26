<script lang="ts">
    import {
        build, OrbitPlugin, load, diagnose, type Diagnostic, type State,
        Canvas, Camera, Orbit, OrbitSmooth, Transform, RenderTarget, Compute, SharedDevice,
        type Plugin, type Config, GizmosPlugin, Gizmos, OutlinePlugin, Outline, InputPlugin, ComputePlugin, RenderPlugin,
        setFieldValue, setString,
        parse,
        findParent, findNodeById,
    } from "@dylanebert/shallot";
    import { getComponent, getComponents, clearRegistry, inspect, snapshot, find } from "@dylanebert/shallot/ecs/core";
    import { requestGPU } from "@dylanebert/shallot/compute/core";
    import { Document, Session, ReadbackSystem, Readback, type Node, type Command } from "@dylanebert/shallot/editor";
    import { requestFrame, now } from "@dylanebert/shallot/runtime";
    import project from "virtual:project";
    import defaultScene from "./default.scene?raw";
    import MenuBar from "./lib/MenuBar.svelte";
    import Outliner from "./lib/Outliner.svelte";
    import Inspector from "./lib/Inspector.svelte";
    import { PaneGroup, Pane, PaneResizer } from "paneforge";
    import { editorLoading } from "./lib/loading";
    import { dismissOnClickOutside } from "./lib/dismiss";
    import { fetchScene, saveScene, type DiscoveredPlugin } from "./project";
    import { STANDARD_PLUGINS, SHALLOT_PLUGINS } from "./plugins";
    import StatusStrip from "./lib/StatusStrip.svelte";
    import LogDrawer from "./lib/LogDrawer.svelte";
    import { push as logPush } from "./lib/log.svelte.js";

    const _origWarn = console.warn;
    const _origError = console.error;

    console.warn = (...args: unknown[]) => {
        _origWarn(...args);
        const text = args.length === 1 ? String(args[0]) : args.map(String).join(" ");
        logPush("warning", text);
    };

    console.error = (...args: unknown[]) => {
        _origError(...args);
        const text = args.length === 1 ? String(args[0]) : args.map(String).join(" ");
        logPush("error", text);
    };

    window.addEventListener("error", (e) => {
        logPush("error", e.message || String(e.error));
    });

    window.addEventListener("unhandledrejection", (e) => {
        logPush("error", "Unhandled rejection: " + String(e.reason));
    });

    const mod = navigator.platform.startsWith("Mac") ? "\u2318" : "Ctrl+";

    const PREFS_KEY = "shallot:editor";

    interface Prefs {
        orbit?: Record<string, number>;
        grid?: boolean;
        lastScene?: Record<string, string>;
        disabledPlugins?: Record<string, string[]>;
        enabledPlugins?: Record<string, string[]>;
    }

    const prefs: Prefs = (() => {
        try { return JSON.parse(localStorage.getItem(PREFS_KEY) ?? "{}"); }
        catch { return {}; }
    })();

    function persistPrefs() {
        localStorage.setItem(PREFS_KEY, JSON.stringify(prefs));
    }

    const allKnown = [...STANDARD_PLUGINS, ...SHALLOT_PLUGINS];
    let customPlugins: DiscoveredPlugin[] = $state.raw(project.custom);
    let projectConfig: Config | null = $state.raw(project.config);
    let enabledPlugins: Set<Plugin> = $state.raw(buildEnabledSet(project.custom, null, project.config));
    let activePlugins: Plugin[] = $derived.by(() => {
        const known = [...customPlugins.map((dp) => dp.plugin), ...allKnown.map((dp) => dp.plugin), ...(projectConfig?.plugins ?? [])];
        const unique = [...new Set(known)];
        return unique.filter((p) => enabledPlugins.has(p));
    });

    let activeScenes: string[] = $state(project.scenes);
    let activeScenePath: string | null = $state(null);
    let activeProjectDir: string | null = $state(null);
    let activeSceneContent: string | null = $state(null);
    let savedBaseline: string | null = $state(null);

    const editDoc = $derived(new Document(activeSceneContent ?? defaultScene));
    let dirty = $derived.by(() => {
        void docVersion;
        if (!savedBaseline) return false;
        return editDoc.serialize() !== savedBaseline;
    });
    let playDoc: Document | null = $state.raw(null);
    const doc = $derived(playDoc ?? editDoc);

    let ecs: State | null = $state.raw(null);
    let session: Session | null = null;
    let nodeMap: Map<Node, number> = $state.raw(new Map());
    const playing = $derived(playDoc !== null);
    let version = $state(0);
    let workspace: HTMLElement;
    let viewportCanvas: HTMLElement;
    let docVersion = $state(0);
    let diagnostics: Diagnostic[] = $derived.by(() => { void docVersion; return diagnose(doc.nodes); });
    let editGrid = $state(prefs.grid ?? true);
    let playGrid = $state(false);
    const grid = $derived(playing ? playGrid : editGrid);
    let editorCamEid = -1;
    let activeCamEid = -1;
    let gizmosOpen: string | null = $state(null);
    let drawerOpen = $state(false);
    let sidebarCollapsed = $state(false);
    let sidebarPane: ReturnType<typeof Pane>;

    const orbitFields = ["yaw", "pitch", "distance", "size"];

    function saveOrbit(comp: Record<string, number[]>, eid: number) {
        const data: Record<string, number> = {};
        for (const f of orbitFields) data[f] = comp[f][eid];
        prefs.orbit = data;
        persistPrefs();
    }

    function restoreOrbit(comp: Record<string, number[]>, eid: number) {
        const data = prefs.orbit;
        if (!data) return;
        for (const f of orbitFields) if (f in data) comp[f][eid] = data[f];
    }

    function toggleGrid() {
        if (!ecs || activeCamEid < 0) return;
        if (playing) playGrid = !playGrid;
        else {
            editGrid = !editGrid;
            prefs.grid = editGrid;
            persistPrefs();
        }
        const value = (playing ? playGrid : editGrid) ? 1 : 0;
        Gizmos.grid[activeCamEid] = value;
    }

    function openGizmos(id: string) {
        gizmosOpen = gizmosOpen === id ? null : id;
    }

    $effect(() => {
        if (!gizmosOpen) return;
        return dismissOnClickOutside(() => { gizmosOpen = null; }, ".gizmos-dropdown");
    });

    $effect(() => {
        if (doc.selection.size > 0 && sidebarCollapsed && sidebarPane) {
            sidebarPane.expand();
        }
    });

    function setupViews(s: State, mode: "edit" | "play") {
        const canvasEid = s.addEntity();
        s.addComponent(canvasEid, Canvas);
        Canvas.selector[canvasEid] = "#canvas";

        if (mode === "edit") {
            editorCamEid = s.addEntity();
            s.addComponent(editorCamEid, Camera);
            s.addComponent(editorCamEid, Orbit);
            s.addComponent(editorCamEid, Transform);
            s.addComponent(editorCamEid, Gizmos);
            Gizmos.grid[editorCamEid] = editGrid ? 1 : 0;
            s.addRelation(editorCamEid, RenderTarget, canvasEid);
            activeCamEid = editorCamEid;
        } else {
            editorCamEid = -1;
            activeCamEid = -1;
        }

        return canvasEid;
    }

    const cameraFields = ["fov", "near", "far", "clearColor", "mode", "size"] as const;
    const syncExclude = new Set<object>();

    function syncCameraEffects(s: State, fromEid: number, toEid: number) {
        if (!syncExclude.size) {
            for (const c of [Camera, Transform, Orbit, Gizmos]) syncExclude.add(c);
        }
        for (const { component } of getComponents()) {
            if (syncExclude.has(component)) continue;
            const c = component as Record<string, number[]>;
            if (s.hasComponent(fromEid, component as never)) {
                if (!s.hasComponent(toEid, component as never)) s.addComponent(toEid, component as never);
                for (const field of Object.keys(c)) c[field][toEid] = c[field][fromEid];
            } else if (s.hasComponent(toEid, component as never)) {
                s.removeComponent(toEid, component as never);
            }
        }
        for (const field of cameraFields) {
            (Camera as Record<string, number[]>)[field][toEid] =
                (Camera as Record<string, number[]>)[field][fromEid];
        }
    }

    function bindGameCamera(s: State, canvasEid: number, map: Map<Node, number>, mode: "edit" | "play") {
        let sceneCam = -1;
        for (const [, eid] of map.entries()) {
            if (s.hasComponent(eid, Camera)) {
                sceneCam = eid;
                break;
            }
        }

        if (mode === "play") {
            if (sceneCam >= 0) {
                s.addRelation(sceneCam, RenderTarget, canvasEid);
                if (!s.hasComponent(sceneCam, Gizmos)) s.addComponent(sceneCam, Gizmos);
                Gizmos.grid[sceneCam] = playGrid ? 1 : 0;
                activeCamEid = sceneCam;
            }
        } else {
            if (sceneCam >= 0) {
                syncCameraEffects(s, sceneCam, editorCamEid);
            }
        }
    }

    let _deviceReady: Promise<GPUDevice> | null = null;
    function ensureDevice(): Promise<GPUDevice> {
        if (!_deviceReady) {
            _deviceReady = requestGPU().then((d) => {
                d.lost.then((info) => {
                    logPush("error", `GPU device lost: ${info.reason} — ${info.message}`);
                    _origError(`GPU device lost: ${info.reason}`, info.message);
                });
                return d;
            });
        }
        return _deviceReady;
    }

    let _prevBuild: Promise<unknown> = Promise.resolve();

    async function buildState(mode: "edit" | "play", plugins: Plugin[], nodes: Node[]): Promise<State> {
        await _prevBuild;
        const device = await ensureDevice();
        clearRegistry();
        const loading = viewportCanvas ? editorLoading(viewportCanvas) : undefined;
        const s = await build({ plugins: [InputPlugin, ComputePlugin, RenderPlugin, OrbitPlugin, GizmosPlugin, OutlinePlugin, ...plugins], loading, defaults: false, setup: (state) => state.setResource(SharedDevice, device) });
        s.scheduler.mode = mode;
        const canvasEid = setupViews(s, mode);
        nodeMap = load(nodes, s);
        for (const d of diagnose(nodes)) logPush("warning", `[shallot] ${d.message}`);
        session = new Session(s, doc, nodeMap);
        bindGameCamera(s, canvasEid, nodeMap, mode);
        s.register(ReadbackSystem);
        s.setResource(Readback, {
            session,
            onUpdate: () => {
                docVersion++;
                if (savedBaseline === null && activeSceneContent) {
                    savedBaseline = editDoc.serialize();
                }
            },
        });
        if (mode === "edit") {
            s.setResource(Outline, {
                getEntities: () => {
                    const eids: number[] = [];
                    for (const node of doc.selection) {
                        const eid = nodeMap.get(node);
                        if (eid !== undefined) eids.push(eid);
                    }
                    return eids;
                },
                color: 0xff6a00,
                thickness: Math.round(2 * devicePixelRatio),
            });
        }
        let disposed = false;
        let lastTime = now();
        function scheduleFrame(): void {
            if (disposed) return;
            requestFrame(tick);
        }
        function tick(): void {
            if (disposed) return;
            const t = now();
            const dt = (t - lastTime) / 1000;
            lastTime = t;
            s.step(dt);
            const wait = Compute.from(s)?.sync();
            if (wait) wait.then(scheduleFrame);
            else scheduleFrame();
        }
        s.onDispose(() => { disposed = true; });
        scheduleFrame();
        const compute = s.getResource(Compute);
        if (compute) {
            const { device } = compute;
            device.onuncapturederror = (event) => {
                const msg = event.error instanceof GPUValidationError ? event.error.message : String(event.error);
                logPush("error", "GPU: " + msg);
                _origError("GPU uncaptured error:", event.error);
            };
        }
        if (mode === "play" && projectConfig?.ui && viewportCanvas) {
            const overlay = document.createElement("div");
            overlay.style.cssText = "position:absolute;inset:0;pointer-events:none;z-index:2;animation:ui-overlay-in 200ms ease-out";
            viewportCanvas.appendChild(overlay);
            const cleanup = projectConfig.ui(overlay, s);
            s.onDispose(() => {
                cleanup();
                overlay.remove();
            });
        }
        return s;
    }

    function transferSelection(from: Document, to: Document) {
        if (from.selection.size === 0) return;
        function walk(a: Node[], b: Node[]) {
            for (let i = 0; i < a.length && i < b.length; i++) {
                if (from.selection.has(a[i])) to.select(b[i]);
                walk(a[i].children, b[i].children);
            }
        }
        walk(from.nodes, to.nodes);
    }

    function play() {
        playDoc = new Document(editDoc.serialize());
        transferSelection(editDoc, playDoc);
        version++;
    }

    function stop() {
        playDoc = null;
        version++;
    }

    function addField(node: Node, attrName: string) {
        if (!session) return;
        const eid = nodeMap.get(node);
        if (eid === undefined) return;
        const attr = node.attrs.find((a) => a.name === attrName);
        session.attachComponent(attrName, attr?.value ?? "", eid);
    }

    function removeField(node: Node, attrName: string) {
        if (!ecs) return;
        const eid = nodeMap.get(node);
        if (eid === undefined) return;
        const reg = getComponent(attrName);
        if (reg) ecs.removeComponent(eid, reg.component as never);
    }

    function syncField(node: Node, attrName: string, fields: Record<string, number | string>) {
        if (!ecs) return;
        const eid = nodeMap.get(node);
        if (eid === undefined) return;
        const reg = getComponent(attrName);
        if (!reg) return;
        for (const [field, value] of Object.entries(fields)) {
            if (typeof value === "number") {
                setFieldValue(reg.component, field, eid, value);
            } else if (typeof value === "string") {
                setString(reg.component, field, eid, value);
            }
        }
        if (!playing && editorCamEid >= 0 && ecs.hasComponent(eid, Camera)) {
            syncCameraEffects(ecs, eid, editorCamEid);
        }
    }

    function buildEnabledSet(custom: DiscoveredPlugin[], dir: string | null, config: Config | null): Set<Plugin> {
        const all = [...custom, ...allKnown];
        const lookup = new Map(all.map((dp) => [dp.name, dp.plugin]));

        const set = new Set<Plugin>();
        if (config) {
            for (const plugin of config.plugins) set.add(plugin);
            if (config.defaults !== false) {
                for (const dp of STANDARD_PLUGINS) {
                    if (!config.exclude?.includes(dp.plugin)) set.add(dp.plugin);
                }
            }
        } else {
            for (const dp of custom) set.add(dp.plugin);
        }

        if (dir) {
            const disabled = prefs.disabledPlugins?.[dir];
            if (disabled) {
                for (const name of disabled) {
                    const plugin = lookup.get(name);
                    if (plugin) set.delete(plugin);
                }
            }
            const enabled = prefs.enabledPlugins?.[dir];
            if (enabled) {
                for (const name of enabled) {
                    const plugin = lookup.get(name);
                    if (plugin) set.add(plugin);
                }
            }
        }

        return set;
    }

    async function switchScene(path: string) {
        if (!activeProjectDir || path === activeScenePath) return;
        try {
            savedBaseline = null;
            activeSceneContent = await fetchScene(activeProjectDir, path);
            activeScenePath = path;
            playDoc = null;
            version++;
            if (!prefs.lastScene) prefs.lastScene = {};
            prefs.lastScene[activeProjectDir] = path;
            persistPrefs();
        } catch (e) {
            console.error("Failed to load scene:", e);
        }
    }

    function togglePlugin(plugin: Plugin) {
        const next = new Set(enabledPlugins);
        if (next.has(plugin)) next.delete(plugin);
        else next.add(plugin);
        enabledPlugins = next;
        version++;

        if (activeProjectDir) {
            const baseSet = buildEnabledSet(customPlugins, null, projectConfig);
            const all = [...customPlugins, ...allKnown];
            const disabled = all.filter((dp) => baseSet.has(dp.plugin) && !next.has(dp.plugin)).map((dp) => dp.name);
            const enabled = all.filter((dp) => !baseSet.has(dp.plugin) && next.has(dp.plugin)).map((dp) => dp.name);
            if (!prefs.disabledPlugins) prefs.disabledPlugins = {};
            if (disabled.length > 0) prefs.disabledPlugins[activeProjectDir] = disabled;
            else delete prefs.disabledPlugins[activeProjectDir];
            if (!prefs.enabledPlugins) prefs.enabledPlugins = {};
            if (enabled.length > 0) prefs.enabledPlugins[activeProjectDir] = enabled;
            else delete prefs.enabledPlugins[activeProjectDir];
            persistPrefs();
        }
    }

    function syncUndoRedo(cmd: Command, isUndo: boolean): boolean {
        if (!session) return false;
        return session.syncCommand(cmd, isUndo);
    }

    function handleUndo() {
        const cmd = doc.undo();
        if (!cmd) return;
        docVersion++;
        syncUndoRedo(cmd, true);
    }

    function handleRedo() {
        const cmd = doc.redo();
        if (!cmd) return;
        docVersion++;
        syncUndoRedo(cmd, false);
    }

    function createEntity(): Node {
        const node: Node = { attrs: [], children: [] };
        doc.add(null, node);
        doc.clearSelection();
        doc.select(node);
        session?.loadNode(node, null);
        docVersion++;
        return node;
    }

    function deleteEntity(node: Node) {
        const parent = findParent(node, doc.nodes);
        if (parent === undefined) return;
        doc.deselect(node);
        doc.remove(parent, node);
        session?.unloadNode(node);
        docVersion++;
    }

    function deleteSelected() {
        if (doc.selection.size === 0) return;
        const selected = [...doc.selection];
        doc.clearSelection();
        for (const node of selected) {
            const parent = findParent(node, doc.nodes);
            if (parent === undefined) continue;
            doc.remove(parent, node);
            session?.unloadNode(node);
        }
        docVersion++;
    }

    let renameSignal = $state(0);

    async function handleSave() {
        if (!activeProjectDir || !activeScenePath) return;
        const content = editDoc.serialize();
        if (content === savedBaseline) return;
        try {
            await saveScene(activeProjectDir, activeScenePath, content);
            savedBaseline = content;
            logPush("info", "Saved " + activeScenePath);
        } catch (e) {
            const msg = e instanceof Error ? e.message : String(e);
            logPush("error", "Save failed: " + msg);
            console.error("Failed to save scene:", e);
        }
    }

    function handleKeydown(e: KeyboardEvent) {
        const tag = (e.target as HTMLElement)?.tagName;
        if (e.key === "s" && (e.ctrlKey || e.metaKey)) {
            e.preventDefault();
            handleSave();
        } else if (e.key === "z" && (e.ctrlKey || e.metaKey) && !e.shiftKey) {
            e.preventDefault();
            handleUndo();
        } else if (e.key === "y" && (e.ctrlKey || e.metaKey)) {
            e.preventDefault();
            handleRedo();
        } else if (e.key === "\\" && (e.ctrlKey || e.metaKey)) {
            e.preventDefault();
            if (sidebarCollapsed) sidebarPane?.expand();
            else sidebarPane?.collapse();
        } else if (e.key === "F2" && doc.selection.size === 1) {
            if (tag === "INPUT" || tag === "TEXTAREA") return;
            e.preventDefault();
            renameSignal++;
        } else if ((e.key === "Delete" || e.key === "Backspace") && !e.ctrlKey && !e.metaKey) {
            if (tag === "INPUT" || tag === "TEXTAREA") return;
            e.preventDefault();
            deleteSelected();
        }
    }

    async function initProject() {
        if (project.dir) {
            activeProjectDir = project.dir;
            activeScenes = project.scenes;
            projectConfig = project.config;
            customPlugins = project.custom;
            enabledPlugins = buildEnabledSet(project.custom, project.dir, project.config);
            const lastPath = prefs.lastScene?.[project.dir];
            const restorePath = lastPath && project.scenes.includes(lastPath) ? lastPath : project.scenes[0] ?? null;
            if (restorePath) {
                activeScenePath = restorePath;
                activeSceneContent = await fetchScene(project.dir, restorePath);
            }
            return;
        }
    }

    initProject();

    $effect(() => {
        const mode: "edit" | "play" = playing ? "play" : "edit";
        const plugins = activePlugins;
        void version;

        const nodes = doc.nodes;

        savedBaseline = null;
        let mounted = true;
        let currentState: State | null = null;

        const p = (async () => {
            currentState = await buildState(mode, plugins, nodes);
            if (mode === "edit") {
                restoreOrbit(Orbit as Record<string, number[]>, editorCamEid);
                if (currentState.hasComponent(editorCamEid, OrbitSmooth))
                    currentState.removeComponent(editorCamEid, OrbitSmooth);
            }
            if (!mounted) { currentState.dispose(); return; }
            ecs = currentState;
            docVersion++;
        })();
        _prevBuild = p;

        return () => {
            mounted = false;
            if (editorCamEid >= 0) saveOrbit(Orbit as Record<string, number[]>, editorCamEid);
            currentState?.dispose();
            ecs = null;
            session = null;
        };
    });

    $effect(() => {
        void docVersion;
        if (!ecs || playing || editorCamEid < 0) return;
        for (const [, eid] of nodeMap.entries()) {
            if (ecs.hasComponent(eid, Camera)) {
                syncCameraEffects(ecs, eid, editorCamEid);
                break;
            }
        }
    });

    function handleApiCommand(payload: { method: string; args?: Record<string, unknown> }) {
        const { method, args = {} } = payload;

        if (method === "undo") { handleUndo(); return { ok: true, version: doc.version }; }
        if (method === "redo") { handleRedo(); return { ok: true, version: doc.version }; }
        if (method === "clearSelection") { doc.clearSelection(); docVersion++; return { ok: true, version: doc.version }; }
        if (method === "select" || method === "deselect") {
            const nodes = ((args.ids ?? []) as string[]).map((id) => findNodeById(id, doc.nodes)).filter(Boolean) as Node[];
            if (method === "select") doc.select(...nodes); else doc.deselect(...nodes);
            docVersion++;
            return { ok: true, version: doc.version };
        }

        const node = args.id ? findNodeById(args.id as string, doc.nodes) : undefined;
        const prevLen = doc.history.undo.length;

        switch (method) {
            case "add": {
                const parsed = parse((args.xml as string) ?? "<Entity />");
                const newNode = parsed[0];
                if (!newNode) return { ok: false, error: "Invalid XML" };
                const parent = args.parent ? findNodeById(args.parent as string, doc.nodes) ?? null : null;
                doc.add(parent, newNode, args.index as number | undefined);
                break;
            }
            case "remove": {
                if (!node) return { ok: false, error: `Node not found: ${args.id}` };
                const parent = findParent(node, doc.nodes, null);
                if (parent === undefined) return { ok: false, error: "Node not in tree" };
                doc.remove(parent, node);
                break;
            }
            case "setAttr": {
                if (!node) return { ok: false, error: `Node not found: ${args.id}` };
                doc.setAttr(node, args.name as string, args.value as string);
                break;
            }
            case "addAttr": {
                if (!node) return { ok: false, error: `Node not found: ${args.id}` };
                doc.addAttr(node, args.name as string, (args.value as string) ?? "");
                break;
            }
            case "removeAttr": {
                if (!node) return { ok: false, error: `Node not found: ${args.id}` };
                doc.removeAttr(node, args.name as string);
                break;
            }
            case "setId": {
                if (!node) return { ok: false, error: `Node not found: ${args.id}` };
                doc.setId(node, args.newId as string | undefined);
                break;
            }
            case "reorder": {
                if (!node) return { ok: false, error: `Node not found: ${args.id}` };
                const parent = args.parent ? findNodeById(args.parent as string, doc.nodes) ?? null : null;
                doc.reorder(parent, node, args.to as number);
                break;
            }
            default:
                return { ok: false, error: `Unknown method: ${method}` };
        }

        if (doc.history.undo.length > prevLen) {
            const cmd = doc.history.undo[doc.history.undo.length - 1].cmd;
            syncUndoRedo(cmd, false);
        }
        docVersion++;
        return { ok: true, version: doc.version };
    }

    function handleApiEntities(payload: { component?: string; eid?: number } | null) {
        if (!ecs) return { error: "No engine running" };
        if (payload?.eid != null) return inspect(ecs, payload.eid);
        if (payload?.component) return find(ecs, payload.component);
        return snapshot(ecs);
    }

    if (import.meta.hot) {
        const strip = (s: string) => s.replace(/\x1b\[[0-9;]*m/g, "");
        import.meta.hot.on("shallot:log", ({ level, message }: { level: string; message: string }) => {
            const clean = strip(message);
            if (level === "error") { logPush("error", clean); _origError(clean); }
            else if (level === "warn") { logPush("warning", clean); _origWarn(clean); }
            else { logPush("info", clean); console.log(clean); }
        });
        import.meta.hot.on("shallot:request", ({ id, type, payload }: { id: string; type: string; payload?: unknown }) => {
            let result: unknown;
            try {
                switch (type) {
                    case "state":
                        result = {
                            scene: doc.serialize(),
                            selection: [...doc.selection].map((n) => n.id).filter(Boolean),
                            version: doc.version,
                        };
                        break;
                    case "command":
                        result = handleApiCommand(payload as { method: string; args?: Record<string, unknown> });
                        break;
                    case "entities":
                        result = handleApiEntities(payload as { component?: string; eid?: number } | null);
                        break;
                    default:
                        result = { error: `Unknown request type: ${type}` };
                }
            } catch (e: unknown) {
                result = { error: e instanceof Error ? e.message : String(e) };
            }
            import.meta.hot!.send("shallot:response", { id, result });
        });
    }
</script>

<svelte:head>
    <title>{dirty ? "* " : ""}shallot{activeProjectDir ? " | " + activeProjectDir.split("/").pop() : ""}</title>
</svelte:head>
<svelte:window onkeydown={handleKeydown} />

<div class="editor" class:playing>
    <header class="toolbar">
        <div class="toolbar-left">
            <MenuBar
                custom={customPlugins}
                standard={STANDARD_PLUGINS}
                shallot={SHALLOT_PLUGINS}
                enabled={enabledPlugins}
                ontoggle={togglePlugin}
                hasProject={customPlugins.length > 0 || !!activeProjectDir}
                scenes={activeScenes}
                activeScene={activeScenePath}
                onscene={switchScene}
            />
        </div>

        <div class="toolbar-center">
            {#if playing}
                <button class="transport-btn stop" onclick={stop} title="Stop">
                    <svg viewBox="0 0 16 16" fill="currentColor">
                        <rect x="3" y="3" width="10" height="10" rx="1" />
                    </svg>
                </button>
            {:else}
                <button class="transport-btn play" onclick={play} title="Play">
                    <svg viewBox="0 0 16 16" fill="currentColor">
                        <path d="M4.5 2.5 L13 8 L4.5 13.5 Z" />
                    </svg>
                </button>
            {/if}
        </div>

        <div class="toolbar-right">
            <button
                class="tool-btn"
                disabled={!(docVersion >= 0 && doc.history.undo.length > 0)}
                onclick={handleUndo}
                title="Undo ({mod}Z)"
            >
                <svg viewBox="0 0 16 16" fill="none" stroke="currentColor" stroke-width="1.5" stroke-linecap="round" stroke-linejoin="round">
                    <path d="M5 4L2 7l3 3" />
                    <path d="M2 7h7.5a3.5 3.5 0 1 1 0 7H8" />
                </svg>
            </button>
            <button
                class="tool-btn"
                disabled={!(docVersion >= 0 && doc.history.redo.length > 0)}
                onclick={handleRedo}
                title="Redo ({mod}Y)"
            >
                <svg viewBox="0 0 16 16" fill="none" stroke="currentColor" stroke-width="1.5" stroke-linecap="round" stroke-linejoin="round">
                    <path d="M11 4l3 3-3 3" />
                    <path d="M14 7H6.5a3.5 3.5 0 1 0 0 7H8" />
                </svg>
            </button>
        </div>
    </header>

    {#snippet gizmosMenu()}
        <div class="gizmos-dropdown">
            <button
                class="viewport-btn"
                class:active={gizmosOpen === "viewport"}
                onclick={() => openGizmos("viewport")}
                title="Gizmos"
            >
                <svg viewBox="0 0 16 16" fill="none" stroke="currentColor" stroke-width="1.3">
                    <path d="M1.5 8 C3.5 4 6 3 8 3 C10 3 12.5 4 14.5 8 C12.5 12 10 13 8 13 C6 13 3.5 12 1.5 8Z" />
                    <circle cx="8" cy="8" r="2" />
                </svg>
                <svg class="caret" viewBox="0 0 8 6" fill="currentColor">
                    <path d="M0.5 0.5 L4 5 L7.5 0.5" />
                </svg>
            </button>
            {#if gizmosOpen === "viewport"}
                <div class="dropdown-menu">
                    <button class="dropdown-item" class:active={grid} onclick={toggleGrid}>
                        <svg viewBox="0 0 16 16" fill="none" stroke="currentColor" stroke-width="1.2">
                            <line x1="1" y1="4" x2="15" y2="4" />
                            <line x1="1" y1="8" x2="15" y2="8" />
                            <line x1="1" y1="12" x2="15" y2="12" />
                            <line x1="4" y1="1" x2="4" y2="15" />
                            <line x1="8" y1="1" x2="8" y2="15" />
                            <line x1="12" y1="1" x2="12" y2="15" />
                        </svg>
                        <span>Grid</span>
                        {#if grid}
                            <svg class="check" viewBox="0 0 16 16" fill="none" stroke="currentColor" stroke-width="2">
                                <path d="M3 8.5 L6.5 12 L13 4" />
                            </svg>
                        {/if}
                    </button>
                </div>
            {/if}
        </div>
    {/snippet}

    <main class="workspace" bind:this={workspace}>
        <PaneGroup direction="horizontal">
            <Pane defaultSize={80} minSize={40}>
                <section class="viewport">
                    <div class="viewport-header">
                        <span class="viewport-label">{playing ? "Play" : "Scene"}{dirty ? " *" : ""}</span>
                        <div class="viewport-actions">
                            {@render gizmosMenu()}
                        </div>
                    </div>
                    <div class="viewport-canvas" bind:this={viewportCanvas}>
                        <canvas id="canvas"></canvas>
                    </div>
                </section>
            </Pane>
            <PaneResizer class="resizer resizer-h" />
            <Pane
                bind:this={sidebarPane}
                collapsible={true}
                collapsedSize={0}
                defaultSize={20}
                minSize={12}
                maxSize={40}
                onCollapse={() => sidebarCollapsed = true}
                onExpand={() => sidebarCollapsed = false}
            >
                <PaneGroup direction="vertical">
                    <Pane defaultSize={35} minSize={20}>
                        <aside class="sidebar">
                            <div class="sidebar-header">Outliner</div>
                            <Outliner {doc} version={docVersion} {diagnostics} {renameSignal} onselect={() => docVersion++} oncreate={createEntity} ondelete={deleteEntity} onreorder={() => docVersion++} onrename={() => docVersion++} />
                        </aside>
                    </Pane>
                    <PaneResizer class="resizer resizer-v" />
                    <Pane defaultSize={35} minSize={20}>
                        <aside class="sidebar">
                            <div class="sidebar-header">Inspector</div>
                            <Inspector {doc} version={docVersion} {diagnostics} {ecs} {nodeMap} onchange={() => docVersion++} onsync={syncField} onremove={removeField} onadd={addField} onreorder={() => docVersion++} />
                        </aside>
                    </Pane>
                </PaneGroup>
            </Pane>
        </PaneGroup>
    </main>
    {#if drawerOpen}
        <LogDrawer onclose={() => drawerOpen = false} />
    {/if}
    <StatusStrip onopen={() => drawerOpen = !drawerOpen} />
</div>

<style>
    .editor {
        --bg: #0e0d0c;
        --surface-1: rgba(255, 255, 255, 0.03);
        --surface-2: rgba(255, 255, 255, 0.07);
        --surface-3: rgba(255, 255, 255, 0.12);
        --surface-4: rgba(255, 255, 255, 0.18);
        --surface-1-solid: #161514;
        --surface-2-solid: #1f1e1d;
        --surface-3-solid: #2b2a29;
        --surface-4-solid: #363534;
        --border: rgba(255, 255, 255, 0.09);
        --text: #f0ece8;
        --text-secondary: #cdc5bc;
        --text-muted: #a09890;
        --accent: #d49560;
        --accent-hover: #e8a86b;
        --cat-spatial: #4a90e2;
        --cat-rendering: #50c878;
        --cat-lighting: #fbbf24;
        --cat-camera: #5bb8c4;
        --cat-effects: #a78bda;
        --cat-environment: #6cb4d9;
        --cat-pipeline: #e0915c;
        --cat-drawing: #d4708f;
        --cat-gameplay: #7bc876;
        --ease-out: cubic-bezier(0.34, 0, 0, 1);

        display: flex;
        flex-direction: column;
        height: 100vh;
        background: var(--bg);
        background-image: url("data:image/svg+xml,%3Csvg xmlns='http://www.w3.org/2000/svg' width='200' height='200'%3E%3Cfilter id='n'%3E%3CfeTurbulence type='fractalNoise' baseFrequency='0.8' numOctaves='4' stitchTiles='stitch'/%3E%3C/filter%3E%3Crect width='100%25' height='100%25' filter='url(%23n)' opacity='0.03'/%3E%3C/svg%3E");
        color: var(--text);
        user-select: none;
        cursor: default;
        -webkit-user-select: none;
    }

    .editor :global(input[type="text"]),
    .editor :global(input[type="number"]),
    .editor :global(input[type="search"]),
    .editor :global(input:not([type])),
    .editor :global(textarea) {
        user-select: text;
        cursor: text;
        -webkit-user-select: text;
    }

    .toolbar {
        display: grid;
        grid-template-columns: 1fr auto 1fr;
        align-items: center;
        height: 40px;
        padding: 0 12px 0 4px;
        background: var(--surface-2-solid);
        transition: background 200ms var(--ease-out);
    }

    .editor.playing .toolbar {
        background: #17140f;
    }

    .toolbar-left {
        display: flex;
        align-items: center;
        gap: 4px;
    }

    .toolbar-center {
        display: flex;
        justify-content: center;
    }

    .toolbar-right {
        display: flex;
        justify-content: flex-end;
        align-items: center;
        gap: 2px;
        padding-right: 4px;
    }

    .tool-btn {
        display: flex;
        align-items: center;
        justify-content: center;
        width: 28px;
        height: 28px;
        padding: 0;
        border: none;
        border-radius: 4px;
        background: transparent;
        color: var(--text-muted);
        cursor: pointer;
        transition: color 150ms var(--ease-out), background 150ms var(--ease-out);
    }

    .tool-btn:hover:not(:disabled) {
        color: var(--text);
        background: var(--surface-2);
    }

    .tool-btn:active:not(:disabled) {
        background: rgba(212, 149, 96, 0.08);
        transform: scale(0.95);
    }

    .tool-btn:focus-visible {
        outline: 1px solid var(--accent);
        outline-offset: -1px;
    }

    .tool-btn:disabled {
        opacity: 0.3;
        cursor: default;
    }

    .tool-btn svg {
        width: 14px;
        height: 14px;
    }

    .transport-btn {
        display: flex;
        align-items: center;
        justify-content: center;
        width: 30px;
        height: 28px;
        border: 1px solid var(--border);
        border-radius: 4px;
        cursor: pointer;
        transition: all 200ms var(--ease-out);
    }

    .transport-btn svg {
        width: 14px;
        height: 14px;
    }

    .transport-btn.play {
        background: transparent;
        color: var(--accent);
        border-color: var(--border);
    }

    .transport-btn.play:hover {
        color: var(--accent-hover);
        border-color: var(--accent);
    }

    .transport-btn:active {
        background: rgba(212, 149, 96, 0.08);
        transform: scale(0.95);
    }

    .transport-btn:focus-visible {
        outline: 1px solid var(--accent);
        outline-offset: -1px;
    }

    .transport-btn.stop {
        background: var(--accent);
        color: var(--bg);
        border-color: var(--accent);
        animation: stop-appear 200ms ease;
    }

    .transport-btn.stop:hover {
        background: var(--accent-hover);
        border-color: var(--accent-hover);
    }

    @keyframes stop-appear {
        0% { transform: scale(0.85); opacity: 0; }
        60% { transform: scale(1.03); opacity: 1; }
        100% { transform: scale(1); }
    }

    .workspace {
        flex: 1;
        min-height: 0;
    }

    .workspace :global([data-pane-group]) {
        display: flex;
        height: 100%;
    }

    .workspace :global([data-pane-group][data-direction="vertical"]) {
        flex-direction: column;
    }

    .workspace :global([data-pane]) {
        overflow: hidden;
    }

    :global(.resizer) {
        flex-shrink: 0;
        position: relative;
        background: var(--border);
        z-index: 2;
        transition: background 150ms var(--ease-out);
    }

    :global(.resizer::before) {
        content: "";
        position: absolute;
    }

    :global(.resizer:hover),
    :global(.resizer[data-resize-handle-active]) {
        background: var(--accent);
    }

    :global(.resizer-h) {
        width: 1px;
        cursor: col-resize;
    }

    :global(.resizer-h::before) {
        top: 0;
        bottom: 0;
        left: -3px;
        right: -3px;
    }

    :global(.resizer-v) {
        height: 1px;
        cursor: row-resize;
    }

    :global(.resizer-v::before) {
        left: 0;
        right: 0;
        top: -3px;
        bottom: -3px;
    }

    .sidebar {
        height: 100%;
        background: var(--surface-1);
        overflow-y: auto;
        scrollbar-gutter: stable;
    }

    .sidebar::-webkit-scrollbar {
        width: 6px;
    }

    .sidebar::-webkit-scrollbar-track {
        background: transparent;
    }

    .sidebar::-webkit-scrollbar-thumb {
        background: transparent;
        border-radius: 4px;
        transition: background 200ms var(--ease-out);
    }

    .sidebar:hover::-webkit-scrollbar-thumb {
        background: var(--border);
    }

    .sidebar-header {
        padding: 8px 12px;
        font-size: 11px;
        font-weight: 600;
        text-transform: uppercase;
        letter-spacing: 0.04em;
        color: var(--accent);
        position: sticky;
        top: 0;
        background: var(--surface-1-solid);
        z-index: 1;
    }

    .viewport {
        height: 100%;
        display: flex;
        flex-direction: column;
        min-height: 0;
        position: relative;
    }

    .viewport-header {
        display: flex;
        align-items: center;
        justify-content: space-between;
        height: 28px;
        padding: 0 8px;
        background: var(--surface-2-solid);
    }

    .viewport-label {
        font-size: 11px;
        font-weight: 500;
        color: var(--text-muted);
        letter-spacing: 0.02em;
    }

    .viewport-actions {
        display: flex;
        align-items: center;
        gap: 2px;
    }

    .gizmos-dropdown {
        position: relative;
    }

    .viewport-btn {
        display: flex;
        align-items: center;
        gap: 4px;
        height: 22px;
        padding: 0 4px;
        border: none;
        border-radius: 4px;
        background: transparent;
        color: var(--text-muted);
        cursor: pointer;
        transition: all 150ms var(--ease-out);
    }

    .viewport-btn:hover {
        background: var(--surface-2);
        color: var(--text-secondary);
    }

    .viewport-btn:active {
        background: rgba(212, 149, 96, 0.08);
        transform: scale(0.95);
    }

    .viewport-btn:focus-visible {
        outline: 1px solid var(--accent);
        outline-offset: -1px;
    }

    .viewport-btn.active {
        background: var(--surface-2);
        color: var(--text);
    }

    .viewport-btn svg {
        width: 14px;
        height: 14px;
    }

    .viewport-btn .caret {
        width: 7px;
        height: 5px;
        opacity: 0.6;
    }

    .dropdown-menu {
        position: absolute;
        top: 100%;
        right: 0;
        margin-top: 4px;
        min-width: 140px;
        padding: 4px;
        border: 1px solid var(--border);
        border-radius: 6px;
        z-index: 10;
        background: rgba(38, 37, 36, 0.85);
        backdrop-filter: blur(16px);
        -webkit-backdrop-filter: blur(16px);
        box-shadow: 0 4px 16px rgba(0, 0, 0, 0.3), 0 1px 4px rgba(0, 0, 0, 0.2);
        transform-origin: top right;
        animation: dropdown-appear 150ms var(--ease-out);
    }

    @keyframes dropdown-appear {
        from { opacity: 0; transform: scale(0.97); }
        to { opacity: 1; transform: scale(1); }
    }

    .dropdown-item {
        display: flex;
        align-items: center;
        gap: 8px;
        width: 100%;
        padding: 4px 8px;
        border: none;
        border-radius: 4px;
        background: transparent;
        color: var(--text-secondary);
        font-size: 12px;
        cursor: pointer;
        transition: background 150ms var(--ease-out);
    }

    .dropdown-item:hover {
        background: var(--surface-2);
        color: var(--text);
    }

    .dropdown-item:active {
        background: rgba(212, 149, 96, 0.08);
    }

    .dropdown-item:focus-visible {
        outline: 1px solid var(--accent);
        outline-offset: -1px;
    }

    .dropdown-item.active {
        color: var(--accent);
    }

    .dropdown-item.active:hover {
        color: var(--accent-hover);
    }

    .dropdown-item svg {
        width: 14px;
        height: 14px;
        flex-shrink: 0;
    }

    .dropdown-item span {
        flex: 1;
        text-align: left;
    }

    .dropdown-item .check {
        width: 12px;
        height: 12px;
        color: var(--accent);
    }

    .viewport-canvas {
        flex: 1;
        min-height: 0;
        position: relative;
        background: var(--surface-1);
        transition: box-shadow 200ms var(--ease-out);
    }

    .editor.playing .viewport-canvas::after {
        content: "";
        position: absolute;
        inset: 0;
        border: 2px solid rgba(212, 149, 96, 0.4);
        pointer-events: none;
        z-index: 1;
    }

    .viewport-canvas canvas {
        position: absolute;
        inset: 0;
        width: 100%;
        height: 100%;
        display: block;
    }

    @keyframes ui-overlay-in {
        from { opacity: 0; }
        to { opacity: 1; }
    }


</style>
