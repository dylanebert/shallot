<script lang="ts">
    import { tick, untrack } from "svelte";
    import {
        build, swap, load, diagnose, type App, type Diagnostic, type State, type System,
        Camera, CameraMode, Orbit, OrbitMode, Part, Transform, Compute, Sear, Tag, invert,
        loadGltf, type Plugin,
    } from "@dylanebert/shallot";
    import { findParent, formatFields, preload } from "@dylanebert/shallot/scene/core";
    import { invalidate } from "@dylanebert/shallot/gltf/core";
    import { attachCanvas, computeViewProj, Meshes } from "@dylanebert/shallot/render/core";
    import { getComponent, clear, readFields } from "@dylanebert/shallot/ecs/core";
    import { Document, Session, ReadbackSystem, Readback, type Node, type Command } from "@dylanebert/shallot/editor";
    import { requestFrame, now, requestGPU } from "@dylanebert/shallot/runtime";
    import { SaveOff } from "lucide-static";
    import { type Bundle, instantiate } from "./lib/bundles";
    import { collectFiles, groupModels, type ImportFile, mintNodes, uploadModel } from "./lib/import";
    import { Overlay, setOverlays, Outline, Handles } from "./lib/viewport";
    import { Pick, isClick, nextSelection, nodeForEid } from "./lib/pick";
    import { manipulatorFor, cursorRay, gizmoScale, localAxes, qrotvec, rayPlane, WORLD_AXES, type Pose, type Vec3 } from "./lib/gizmo";
    import { TOOLS, DEFAULT_TOOL, Tool, toolForKey } from "./lib/tool";
    import { Frame, FRAMES, DEFAULT_FRAME, FRAME_KEY, nextFrame } from "./lib/frame";
    import { enclose, frameDistance, frameSize, type Sphere } from "./lib/fit";
    import { Pivot, PIVOTS, DEFAULT_PIVOT, PIVOT_KEY, nextPivot, pivotAnchor } from "./lib/pivot";
    import { hint } from "./lib/hotkey";
    import project from "virtual:project";
    import defaultScene from "./default.scene?raw";
    import Icon from "./lib/Icon.svelte";
    import MenuBar from "./lib/MenuBar.svelte";
    import Outliner from "./lib/Outliner.svelte";
    import Inspector from "./lib/Inspector.svelte";
    import Docs from "./lib/Docs.svelte";
    import { docs } from "./lib/docs.corpus";
    import { docFor } from "./lib/docs";
    import { editorLoading } from "./lib/loading";
    import { THEMES, setTheme, packed, current } from "./lib/theme";
    import { dismissOnClickOutside } from "./lib/dismiss";
    import { matchScene, Persist, pickSaveMode, classifyExternal } from "./project";
    import { type Manifest, serialize as serializeManifest, setEnabled } from "./project/manifest";
    import { onProjectReload } from "./project/reload";
    import { entriesFor, enabledPlugins, localPlugins, compose } from "./plugins";
    import { type Prefs, PREFS_KEY, PANEL_DEFAULT, DOCS_DEFAULT, parsePrefs, serializePrefs, clampWidth, clampDocsWidth } from "./lib/prefs";
    import { applyCommand, queryEntities, type CommandPayload } from "./lib/commands";
    import { resolveShortcut, inField } from "./lib/keymap";
    import { syncCameraEffects } from "./lib/camera";
    import Toasts from "./lib/Toasts.svelte";
    import Banners from "./lib/Banners.svelte";
    import Issues from "./lib/Issues.svelte";
    import { toast, banner, clearBanner } from "./lib/notify.svelte.js";

    // the editor band carries only editor-vocabulary signals (save, build, device-lost); the runtime +
    // engine firehose (console.warn/error, GPU validation, script throws) stays in the browser console,
    // where a dev who can run a game engine already looks. An uncaught error gets one pointer toast — the
    // beginner's signpost to F12 — never the stack trace mirrored into the UI.
    let _lastCrashToast = 0;
    function crashPointer() {
        const t = Date.now();
        if (t - _lastCrashToast < 4000) return;
        _lastCrashToast = t;
        toast("error", "Unexpected error. Open the browser console (F12) for details.");
    }

    window.addEventListener("error", crashPointer);
    window.addEventListener("unhandledrejection", crashPointer);

    Persist.mode = pickSaveMode(location.search);

    const prefs: Prefs = parsePrefs(localStorage.getItem(PREFS_KEY));

    function persistPrefs() {
        localStorage.setItem(PREFS_KEY, serializePrefs(prefs));
    }

    // the `shallot.json` manifest is the single source of plugin enablement, read by both the editor and
    // dev mode; `locals` are the project's own plugin objects the generated `virtual:project` imported (the
    // editor can't import a project file itself). `entries` resolves the two into the menu's rows + the
    // build's plugin set; a toggle rewrites the manifest (and persists it), re-deriving everything.
    let manifest: Manifest = $state.raw(project.manifest);
    let locals: { name: string; plugin: Plugin }[] = $state.raw(project.locals);
    let entries = $derived(entriesFor(manifest, locals));
    let activePlugins: Plugin[] = $derived(enabledPlugins(entries));

    let activeScenes: string[] = $state(project.scenes);
    let activeScenePath: string | null = $state(null);
    let activeProjectDir: string | null = $state(null);
    let activeSceneContent: string | null = $state(null);
    let savedBaseline: string | null = $state(null);

    const editDoc = $derived(new Document(activeSceneContent ?? defaultScene));
    // one serialize per change, shared by `dirty` and autosave. keys off editDoc (the edit scene),
    // never `doc` — `doc` is playDoc in play mode. the docVersion read bridges in-place doc mutations.
    const serialized = $derived.by(() => { void docVersion; return editDoc.serialize(); });
    let dirty = $derived(!!savedBaseline && serialized !== savedBaseline);
    // unpersisted changes in a real (file) project — drives the ambient tab-title dot + canSave.
    // ephemeral never persists, so it's never "unsaved" (the no-save indicator carries that mode instead).
    const unsaved = $derived(dirty && Persist.mode === "file");
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
    // the edit viewport's enabled-overlay set (a bitmask of `Overlay` categories), persisted across State
    // rebuilds; the edit camera's `Overlays.enabled` is the runtime source of truth. Overlays are edit-only
    // editor chrome — the Scene-View grid + selection outline — so play carries none: a faithful preview
    // renders the scene as it ships (GizmosPlugin isn't even in the play build — see plugins.ts `compose`).
    let editMask = $state(prefs.overlays?.edit ?? (Overlay.Grid | Overlay.Outline));
    let editorCamEid = -1;
    let activeCamEid = -1;
    // edit-camera pose held across in-session rebuilds (plugin toggle, play/stop) so the view doesn't
    // jar — but not persisted, so a fresh load starts from the good Orbit defaults, not a stale pose
    let sessionOrbit: Record<string, number> | undefined;
    let gizmosOpen: string | null = $state(null);
    // the viewport's active transform tool (classic, one mode at a time). Session state the manipulator
    // gizmo reads in a later pass; not persisted — a fresh session starts in Move.
    let tool: Tool = $state(DEFAULT_TOOL);
    // the gizmo coordinate frame (World / Local), set from the viewport-bar dropdown. Session state, not
    // persisted. `currentFrame` is the active entry for the dropdown's indicator.
    let coordFrame: Frame = $state(DEFAULT_FRAME);
    const currentFrame = $derived(FRAMES.find((f) => f.id === coordFrame) ?? FRAMES[0]);
    // Scale is always local (a per-axis world scale of a rotated object isn't representable in TRS), so the
    // orientation dropdown shows Local + disables for Scale rather than promising a frame it can't honor.
    // contextual tool options: a control shows only for the tools whose result it changes. Frame (axis
    // orientation) applies to Move + Rotate — Scale is always local. Pivot (orbit/scale anchor) applies to
    // Rotate + Scale — Move is a translation delta the anchor doesn't change. Select shows neither.
    const showFrame = $derived(tool === Tool.Move || tool === Tool.Rotate);
    const showPivot = $derived(tool === Tool.Rotate || tool === Tool.Scale);
    // the gizmo pivot (Median / Active), set from the viewport-bar dropdown. Session state, not persisted.
    // Median = the selection centroid; Active = the active (last-picked) entity's own origin.
    let pivotMode: Pivot = $state(DEFAULT_PIVOT);
    const currentPivot = $derived(PIVOTS.find((p) => p.id === pivotMode) ?? PIVOTS[0]);

    let outlinerWidth = $state(clampWidth(prefs.outlinerWidth ?? PANEL_DEFAULT));
    let inspectorWidth = $state(clampWidth(prefs.inspectorWidth ?? PANEL_DEFAULT));
    let outlinerCollapsed = $state(prefs.outlinerCollapsed ?? false);
    let inspectorCollapsed = $state(prefs.inspectorCollapsed ?? false);
    let resizing = $state(false);

    // the docs reader is a summoned overlay (editor-ui.md: a summoned surface, not a new docked region) —
    // the menu Docs entry or the ? key opens it to the index; an inspector docs link opens it to a
    // component's reference. `docIndex` is the same generated docs/dist artifact the site renders.
    const docIndex = docs();
    let docsOpen = $state(false);
    let docsTarget: { slug: string; anchor?: string } | null = $state.raw(null);
    let docsWidth = $state(clampDocsWidth(prefs.docsWidth ?? DOCS_DEFAULT));

    function openDocs(component: string) {
        // open the component's page at the top, not deep-linked to its (bottom-of-page) reference entry
        const t = docFor(docIndex, component);
        docsTarget = t ? { slug: t.slug } : null;
        docsOpen = true;
    }

    function openDocsIndex() {
        docsTarget = null;
        docsOpen = true;
    }

    // the help shortcut toggles; the menu entry only opens (a menu pick shouldn't dismiss)
    function toggleDocs() {
        if (docsOpen) docsOpen = false;
        else openDocsIndex();
    }

    // the docs drawer is a summoned overlay: a click outside it dismisses it, like every other floating
    // editor surface. Escape closes it too (the menu entry that opens it has already closed by then).
    $effect(() => {
        if (!docsOpen) return;
        return dismissOnClickOutside(() => { docsOpen = false; }, ".docs-reader");
    });

    function startResize(side: "outliner" | "inspector" | "docs", e: PointerEvent) {
        e.preventDefault();
        resizing = true;
        const startX = e.clientX;
        const startW = side === "outliner" ? outlinerWidth : side === "docs" ? docsWidth : inspectorWidth;
        const move = (ev: PointerEvent) => {
            // outliner grows dragging right; the inspector and the docs drawer grow dragging left
            const dx = side === "outliner" ? ev.clientX - startX : startX - ev.clientX;
            if (side === "outliner") outlinerWidth = clampWidth(startW + dx);
            else if (side === "docs") docsWidth = clampDocsWidth(startW + dx);
            else inspectorWidth = clampWidth(startW + dx);
        };
        const up = () => {
            window.removeEventListener("pointermove", move);
            window.removeEventListener("pointerup", up);
            document.body.style.cursor = "";
            resizing = false;
            prefs.outlinerWidth = outlinerWidth;
            prefs.inspectorWidth = inspectorWidth;
            prefs.docsWidth = docsWidth;
            persistPrefs();
        };
        document.body.style.cursor = "col-resize";
        window.addEventListener("pointermove", move);
        window.addEventListener("pointerup", up);
    }

    function setCollapsed(outliner: boolean, inspector: boolean) {
        outlinerCollapsed = outliner;
        inspectorCollapsed = inspector;
        prefs.outlinerCollapsed = outliner;
        prefs.inspectorCollapsed = inspector;
        persistPrefs();
    }

    // console escape hatch for the persisted layout (prefs survive refresh): `editor.resetLayout()`
    // restores default panel widths, `editor.resetPrefs()` wipes all editor prefs and reloads
    (window as unknown as { editor: object }).editor = {
        prefs: () => ({ ...prefs }),
        resetLayout() {
            outlinerWidth = PANEL_DEFAULT;
            inspectorWidth = PANEL_DEFAULT;
            delete prefs.outlinerWidth;
            delete prefs.inspectorWidth;
            setCollapsed(false, false);
        },
        resetPrefs() {
            localStorage.removeItem(PREFS_KEY);
            location.reload();
        },
    };

    function saveOrbit(eid: number) {
        sessionOrbit = {
            yaw: Orbit.yaw.get(eid),
            pitch: Orbit.pitch.get(eid),
            distance: Orbit.distance.get(eid),
            size: Orbit.size.get(eid),
        };
    }

    function restoreOrbit(eid: number) {
        const data = sessionOrbit;
        if (!data) return;
        if ("yaw" in data) Orbit.yaw.set(eid, data.yaw);
        if ("pitch" in data) Orbit.pitch.set(eid, data.pitch);
        if ("distance" in data) Orbit.distance.set(eid, data.distance);
        if ("size" in data) Orbit.size.set(eid, data.size);
    }

    function toggleOverlay(category: number) {
        // edit-only: overlays are Scene-View chrome, and the gizmos control that calls this is hidden in play
        if (!ecs || editorCamEid < 0) return;
        editMask ^= category;
        setOverlays(ecs, editorCamEid, editMask);
        if (!prefs.overlays) prefs.overlays = {};
        prefs.overlays.edit = editMask;
        persistPrefs();
    }

    function openGizmos(id: string) {
        gizmosOpen = gizmosOpen === id ? null : id;
    }

    let editorEl: HTMLDivElement;
    let themeId = $state(prefs.theme ?? THEMES[0].id);
    // a theme switch retints everything live: CSS cascades from setTheme's custom properties, the grid
    // reads `current.palette` each frame, and the viewport stage + outline are pushed onto the edit camera
    // (effect below). play mode keeps the scene's own clear color — WYSIWYG.
    $effect(() => {
        setTheme(editorEl, themeId);
        prefs.theme = themeId;
        persistPrefs();
    });

    // the outline isn't a CSS surface, so push it on theme change or State rebuild. The viewport clear
    // color is not theme-driven: it mirrors the scene camera (syncCameraEffects), so edit reads WYSIWYG.
    $effect(() => {
        void themeId;
        Outline.color = packed(current.palette.outline);
    });

    function selectTheme(id: string) {
        themeId = id;
    }

    $effect(() => {
        if (!gizmosOpen) return;
        return dismissOnClickOutside(() => { gizmosOpen = null; }, ".gizmos-dropdown");
    });

    $effect(() => {
        if (doc.selection.size > 0 && inspectorCollapsed) {
            setCollapsed(outlinerCollapsed, false);
        }
    });

    function setupViews(s: State, mode: "edit" | "play"): HTMLCanvasElement | null {
        const canvasEl = document.querySelector<HTMLCanvasElement>("#canvas");

        if (mode === "edit") {
            editorCamEid = s.create();
            s.add(editorCamEid, Camera);
            // no editor-chrome clear color: the viewport mirrors the scene camera's clear color
            // (bindGameCamera → syncCameraEffects), falling back to the Camera default when the scene
            // has no camera. Edit reads WYSIWYG against play mode.
            s.add(editorCamEid, Orbit);
            // the editor can't predict scene scale, so open the orbit bounds effectively unbounded — the
            // adaptive frustum below keeps the target in view at any zoom, and frame-to-fit (F) jumps to a
            // fitting distance. A game tightens these per-camera; the editor never clamps the artist.
            Orbit.minDistance.set(editorCamEid, 1e-3);
            Orbit.maxDistance.set(editorCamEid, 1e6);
            Orbit.minSize.set(editorCamEid, 1e-3);
            Orbit.maxSize.set(editorCamEid, 1e6);
            s.add(editorCamEid, Transform);
            s.addSystem(editCameraSystem);
            // Sear renders the camera; setOverlays pulls in each enabled overlay's prepass lane
            // (Tag → selection outline, Depth → grid occlusion)
            s.add(editorCamEid, Sear);
            // the prepass id lane (view.tag) viewport picking reads — a foundational edit-mode capability,
            // added independent of the selection-outline overlay so toggling that overlay can't break pick
            s.add(editorCamEid, Tag);
            setOverlays(s, editorCamEid, editMask);
            if (canvasEl) attachCanvas(editorCamEid, canvasEl);
            activeCamEid = editorCamEid;
        } else {
            editorCamEid = -1;
            activeCamEid = -1;
        }

        return canvasEl;
    }

    function bindGameCamera(s: State, canvasEl: HTMLCanvasElement | null, map: Map<Node, number>, mode: "edit" | "play") {
        let sceneCam = -1;
        for (const [, eid] of map.entries()) {
            if (s.has(eid, Camera)) {
                sceneCam = eid;
                break;
            }
        }

        if (mode === "play") {
            if (sceneCam >= 0) {
                // no editor overlays in play — GizmosPlugin (which owns the Overlays component) isn't in the
                // play build, so the preview renders the scene exactly as it ships
                if (canvasEl) attachCanvas(sceneCam, canvasEl);
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
            _deviceReady = requestGPU().then(({ device }) => {
                device.lost.then((info) => {
                    banner("gpu-lost", "error", `GPU device lost (${info.reason}). Reload to recover.`);
                    console.error(`GPU device lost: ${info.reason}`, info.message);
                });
                return device;
            });
        }
        return _deviceReady;
    }

    let _prevBuild: Promise<unknown> = Promise.resolve();
    // the project plugins the live State was built with — the hot-reload swap's `prev`, paired by name
    // against the re-discovered set. Their system objects are the ones live in the scheduler (a swap
    // mutates them in place), so this tracks the last *build*, not the last swap; only a rebuild resets it.
    let builtProject: Plugin[] = [];
    // the build's skip set (missing-dep plugins that never initialized) — swap rejects against it
    let builtSkipped: readonly string[] = [];

    async function buildState(mode: "edit" | "play", plugins: Plugin[], nodes: Node[]): Promise<App> {
        await _prevBuild;
        const device = await ensureDevice();
        clear();
        const loading = viewportCanvas ? editorLoading(viewportCanvas) : undefined;
        // compose the build by mode: play is the app's plugins verbatim (a faithful preview), edit adds the
        // editor foundation through the engine's mode/layer axes (plugins.ts `compose`) — never a force-prepend
        // onto play, so toggling an app plugin off is honest the moment play stops running it.
        const app = await build({ plugins: compose(mode, plugins), loading, defaults: false, device, mode, capacity: project.capacity ?? undefined });
        // post-build setup (view attach, scene load, session wiring) can throw — a scene parse error in
        // `load` (a toggle leaving a scene attr for a now-unregistered component), a canvas/context failure
        // in setupViews. Guard it so the freshly-built app is torn down on failure: otherwise its State
        // leaks and its editor camera stays bound in `Views`, wedging every later rebuild with
        // "attachCanvas: eid already bound". dispose() clears Views (RenderPlugin.dispose).
        try {
            const s = app.state;
            const canvasEl = setupViews(s, mode);
            // the pre-load resolve pass (the engine's build does the same for its own scenes): a glTF
            // mesh named in the doc imports here, so `load` resolves it
            await preload(nodes, s);
            nodeMap = load(nodes, s);
            session = new Session(s, doc, nodeMap);
            bindGameCamera(s, canvasEl, nodeMap, mode);
            s.addSystem(ReadbackSystem);
            Readback.session = session;
            Readback.onUpdate = () => {
                docVersion++;
            };
            if (mode === "edit") {
                // normalize the freshly-loaded document to its canonical (serialize-equivalent) form once,
                // then pin it as the clean baseline. Captured here — after the normalizing pass, not lazily
                // in onUpdate post-mutation — so the on-disk file and the editor's "clean" state can't
                // diverge: a first edit autosaves a diff proportional to the edit, never a whole-file rewrite.
                ReadbackSystem.update!(s);
                savedBaseline = editDoc.serialize();
                Outline.getEntities = () => {
                    const eids: number[] = [];
                    for (const node of doc.selection) {
                        const eid = nodeMap.get(node);
                        if (eid !== undefined) eids.push(eid);
                    }
                    return eids;
                };
                Outline.color = packed(current.palette.outline);
                Outline.thickness = Math.round(2 * devicePixelRatio);
                // the gizmo's anchor: the selection centroid, only while a manipulator tool (Move / Rotate
                // / Scale) is active — Select shows no handles. All three read `tool` / `localFrame` live so
                // a tool switch or frame toggle retints without a rebuild.
                Handles.getOrigin = () => {
                    if (!manipulatorFor(tool)) return null;
                    // the transformable selection in order — the last is the active (last-picked) entity.
                    // pivotAnchor resolves Median (centroid) vs Active (the active entity's origin).
                    const points: Vec3[] = [];
                    for (const node of doc.selection) {
                        const eid = nodeMap.get(node);
                        if (eid === undefined || !s.has(eid, Transform)) continue;
                        points.push([
                            Transform.pos.x.get(eid),
                            Transform.pos.y.get(eid),
                            Transform.pos.z.get(eid),
                        ]);
                    }
                    return pivotAnchor(pivotMode, points, points.length - 1);
                };
                // the active tool's manipulator drives both the draw and the pick (its declared handle set)
                Handles.getManip = () => manipulatorFor(tool);
                // World / Local from the dropdown — except Scale, which is ALWAYS local: a per-axis world
                // scale of a rotated object needs shear that TRS can't represent, so three.js / Unity force
                // it (see axisScale). Local aligns to the primary (first) selection's orientation.
                Handles.getAxes = () => {
                    if (coordFrame === Frame.World && tool !== Tool.Scale) return WORLD_AXES;
                    // align to the active (last-picked) entity's frame — the same "active" the pivot reads
                    let active = -1;
                    for (const node of doc.selection) {
                        const eid = nodeMap.get(node);
                        if (eid !== undefined && s.has(eid, Transform)) active = eid;
                    }
                    if (active < 0) return WORLD_AXES;
                    return localAxes([
                        Transform.rot.x.get(active),
                        Transform.rot.y.get(active),
                        Transform.rot.z.get(active),
                        Transform.rot.w.get(active),
                    ]);
                };
                // the gizmo's world size, off the canvas CSS height — the same basis the CSS-px hit-test
                // uses, so the drawn gizmo and the grabbable region match on every DPR
                Handles.getScale = () => {
                    const o = Handles.getOrigin();
                    return o && canvasEl ? gizmoWorldScale(o, canvasEl.clientHeight) : 0;
                };
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
                const wait = Compute?.sync();
                if (wait) wait.then(scheduleFrame);
                else scheduleFrame();
            }
            // the retired lifecycle subsystem's onDispose is now a system dispose hook — the scheduler calls it
            // for every registered system on state.dispose(), stopping the frame loop and tearing down the overlay
            const lifecycle: System = {
                name: "editor-lifecycle",
                annotations: { mode: "always" },
                dispose() {
                    disposed = true;
                },
            };
            s.addSystem(lifecycle);
            scheduleFrame();
            if (Compute.device) {
                Compute.device.onuncapturederror = (event) => {
                    console.error("GPU uncaptured error:", event.error);
                };
            }
            return app;
        } catch (e) {
            app.dispose();
            throw e;
        }
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

    async function play() {
        // a rebuild re-adopts the live content as the saved baseline, so flush pending edits first or
        // they never reach disk (same guard as switchScene)
        if (dirty && !(await flushNow())) {
            toast("error", "Couldn't save — not entering play");
            return;
        }
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
        const component = getComponent(attrName);
        if (component) ecs.remove(eid, component as never);
    }

    function syncField(node: Node, attrName: string, fields: Record<string, number | string>) {
        if (!session || !ecs) return;
        const eid = nodeMap.get(node);
        if (eid === undefined) return;
        session.syncFields(attrName, fields, eid);
        if (!playing && editorCamEid >= 0 && ecs.has(eid, Camera)) {
            syncCameraEffects(ecs, eid, editorCamEid);
        }
    }

    async function switchScene(path: string) {
        if (!activeProjectDir || path === activeScenePath) return;
        // flush pending edits before leaving; a failed save must not silently discard them
        if (dirty && !(await flushNow())) {
            toast("error", "Couldn't save — staying on the current scene");
            return;
        }
        try {
            savedBaseline = null;
            activeSceneContent = await Persist.load(activeProjectDir, path);
            activeScenePath = path;
            playDoc = null;
            version++;
            if (!prefs.lastScene) prefs.lastScene = {};
            prefs.lastScene[activeProjectDir] = path;
            persistPrefs();
        } catch (e) {
            toast("error", `Couldn't load ${path}`);
            console.error("Failed to load scene:", e);
        }
    }

    async function togglePlugin(name: string) {
        // the toggle rebuilds the State; flush first so an in-flight edit isn't dropped (see play)
        if (dirty && !(await flushNow())) {
            toast("error", "Couldn't save — plugin not toggled");
            return;
        }
        const entry = entries.find((e) => e.name === name);
        if (!entry || !activeProjectDir) return;
        // the manifest is the one enablement truth — write it (the editor + dev mode both read it), and the
        // `$state.raw` swap re-derives `entries`/`activePlugins`, rebuilding the State on the next effect tick.
        manifest = setEnabled(manifest, name, !entry.enabled, entry.source, entry.spec);
        version++;
        try {
            await Persist.saveManifest(activeProjectDir, serializeManifest(manifest));
        } catch (e) {
            toast("error", "Couldn't save the plugin change");
            console.error("Failed to save manifest:", e);
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

    function createEntity(bundle: Bundle): void {
        const node = instantiate(bundle, doc);
        doc.add(null, node);
        doc.clearSelection();
        doc.select(node);
        session?.loadNode(node, null);
        docVersion++;
    }

    // ── model import (viewport drop + the Add menu's Model… picker) ──
    // upload → live loadGltf → mint document nodes (one compound undo step) → select. Feedback rides the
    // gltf-import banner (during decode; error text on failure) + a success toast. The Gltf plugin is the
    // import's substrate (preloader + route sync + surfaces), so a project without it enables it first —
    // dropping a model IS opting into glTF support.
    let importing = false;

    async function ensureGltfEnabled(): Promise<boolean> {
        const entry = entries.find((e) => e.name === "Gltf");
        if (entry?.enabled) return true;
        await togglePlugin("Gltf");
        // the toggle rebuilds through the effect — let it schedule, then wait for the build to land
        await tick();
        await _prevBuild;
        const after = entries.find((e) => e.name === "Gltf");
        if (!after?.enabled) return false;
        toast("info", "Enabled the Gltf plugin");
        return true;
    }

    async function importModels(files: ImportFile[], at: [number, number, number]): Promise<void> {
        if (playing) {
            toast("error", "Stop play mode to import models");
            return;
        }
        if (importing) return;
        const groups = groupModels(files);
        if (groups.length === 0) {
            toast("error", "No .glb or .gltf file in the drop");
            return;
        }
        importing = true;
        try {
            if (!(await ensureGltfEnabled())) throw new Error("Couldn't enable the Gltf plugin");
            const minted: Node[] = [];
            for (const group of groups) {
                banner("gltf-import", "info", `Importing ${group.root.path}…`);
                const src = await uploadModel(group);
                const state = ecs;
                if (!state || !session) throw new Error("No scene is open");
                const imp = await loadGltf(state, src);
                const nodes = mintNodes(imp, src, at, doc);
                // one compound doc entry for the whole import — a single undo removes every minted node
                doc.compound(
                    nodes.map(
                        (node, i): Command => ({
                            type: "add",
                            parent: null,
                            node,
                            index: doc.nodes.length + i,
                        }),
                    ),
                );
                for (const node of nodes) session.loadNode(node, null);
                minted.push(...nodes);
            }
            doc.clearSelection();
            for (const node of minted) doc.select(node);
            docVersion++;
            clearBanner("gltf-import");
            toast("info", `Imported ${groups.map((g) => g.root.path).join(", ")}`);
        } catch (e) {
            const msg = e instanceof Error ? e.message : String(e);
            banner("gltf-import", "error", `Import failed: ${msg}`);
            console.error("[shallot] model import failed:", e);
        } finally {
            importing = false;
        }
    }

    // the ground-plane point under the drop cursor — where a dropped model lands (the picker path uses
    // the origin). Falls back to the origin when the ray is parallel or the edit camera isn't up.
    function dropPoint(e: DragEvent): [number, number, number] {
        if (editorCamEid < 0 || !viewportCanvas) return [0, 0, 0];
        const rect = viewportCanvas.getBoundingClientRect();
        if (rect.width === 0 || rect.height === 0) return [0, 0, 0];
        computeViewProj(editorCamEid, rect.width / rect.height, _gvp);
        const ray = cursorRay(invert(_gvp, _ginv), e.clientX - rect.left, e.clientY - rect.top, rect.width, rect.height);
        const hit = rayPlane([0, 0, 0], [0, 1, 0], ray);
        return hit ? [hit[0], hit[1], hit[2]] : [0, 0, 0];
    }

    // the drop overlay shows only for a file drag (an outliner row drag also fires dragenter). The
    // depth counter absorbs the enter/leave churn of child elements.
    let dropActive = $state(false);
    let dropDepth = 0;

    function hasFiles(e: DragEvent): boolean {
        return !!e.dataTransfer && [...e.dataTransfer.types].includes("Files");
    }

    function handleDragEnter(e: DragEvent) {
        if (!hasFiles(e)) return;
        e.preventDefault();
        dropDepth++;
        dropActive = true;
    }

    function handleDragOver(e: DragEvent) {
        if (!hasFiles(e)) return;
        e.preventDefault();
    }

    function handleDragLeave(e: DragEvent) {
        if (!hasFiles(e)) return;
        if (--dropDepth <= 0) {
            dropDepth = 0;
            dropActive = false;
        }
    }

    async function handleDrop(e: DragEvent) {
        if (!hasFiles(e)) return;
        e.preventDefault();
        dropDepth = 0;
        dropActive = false;
        const at = dropPoint(e);
        const files = await collectFiles(e.dataTransfer!);
        await importModels(files, at);
    }

    // the Add menu's Model… entry: a hidden file input (drop's sibling surface — a picker import lands
    // at the origin)
    let importInput: HTMLInputElement | undefined = $state();

    function openImportPicker() {
        importInput?.click();
    }

    async function handleImportPick() {
        const list = importInput?.files;
        if (!list || list.length === 0) return;
        const files: ImportFile[] = [];
        for (const file of Array.from(list)) files.push({ path: file.name, bytes: await file.arrayBuffer() });
        if (importInput) importInput.value = "";
        await importModels(files, [0, 0, 0]);
    }

    function deleteEntity(node: Node) {
        const parent = findParent(node, doc.nodes);
        if (parent === undefined) return;
        doc.deselect(node);
        doc.remove(parent, node);
        session?.unloadNode(node);
        docVersion++;
    }

    function selectIssue(node: Node) {
        doc.clearSelection();
        doc.select(node);
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

    // viewport pick: a left tap (not an orbit drag) selects the entity under the cursor, read from the
    // editor camera's prepass id lane by the tooling PickSystem (Pick.eid). The eid → node resolution and
    // the click-vs-drag gate are pure (lib/pick); this wires them to the Document through the same
    // doc.select path the outliner uses, so selection stays one source of truth and the outline (which
    // samples the same lane) updates for free. Inert in play mode (the readback + this handler both bail).
    let pickDownX = 0;
    let pickDownY = 0;
    let picking = false;

    // scratch for the gizmo grab + hover: the editor camera's view-projection (hit-test) + its inverse (ray)
    const _gvp = new Float32Array(16);
    const _ginv = new Float32Array(16);

    // the selection's world bounding sphere (tool-independent), or null when nothing transformable is
    // selected. Each entity contributes its mesh's local sphere transformed to world — a Part with
    // registered Mesh.bounds gives a real extent; a light or empty contributes a point (radius 0). The
    // spheres enclose into one, so frame-to-fit fits the whole selection, not just its centroid.
    function selectionBounds(): Sphere | null {
        if (!ecs) return null;
        const spheres: Sphere[] = [];
        for (const node of doc.selection) {
            const eid = nodeMap.get(node);
            if (eid === undefined || !ecs.has(eid, Transform)) continue;
            const sx = Transform.scale.x.get(eid);
            const sy = Transform.scale.y.get(eid);
            const sz = Transform.scale.z.get(eid);
            let lc: Vec3 = [0, 0, 0];
            let lr = 0;
            if (ecs.has(eid, Part)) {
                const mesh = Meshes.get(Meshes.name(Part.mesh.get(eid)) ?? "");
                if (mesh?.bounds) {
                    lc = [mesh.bounds[0], mesh.bounds[1], mesh.bounds[2]];
                    lr = mesh.bounds[3];
                }
            }
            // world center = pos + rot·(localCenter ⊙ scale); world radius = localRadius · max|scale axis|
            // (matching the pack's conservative non-uniform-scale bound).
            const off = qrotvec(
                [
                    Transform.rot.x.get(eid),
                    Transform.rot.y.get(eid),
                    Transform.rot.z.get(eid),
                    Transform.rot.w.get(eid),
                ],
                [lc[0] * sx, lc[1] * sy, lc[2] * sz],
            );
            spheres.push({
                center: [
                    Transform.pos.x.get(eid) + off[0],
                    Transform.pos.y.get(eid) + off[1],
                    Transform.pos.z.get(eid) + off[2],
                ],
                radius: lr * Math.max(Math.abs(sx), Math.abs(sy), Math.abs(sz)),
            });
        }
        return enclose(spheres);
    }

    // the gizmo's world-space size for a constant on-screen extent — render + pick must compute it the same
    // way (off the editor camera's distance to `origin`), so what's drawn is exactly what's grabbable
    function gizmoWorldScale(origin: Vec3, heightPx: number): number {
        const persp = Camera.mode.get(editorCamEid) !== CameraMode.Orthographic;
        const dist = Math.hypot(
            origin[0] - Transform.pos.x.get(editorCamEid),
            origin[1] - Transform.pos.y.get(editorCamEid),
            origin[2] - Transform.pos.z.get(editorCamEid),
        );
        const fovOrSize = persp ? Camera.fov.get(editorCamEid) : Camera.size.get(editorCamEid);
        return gizmoScale(persp, fovOrSize, dist, heightPx);
    }

    // F frames the selection: recenter the orbit on the selection's bounding sphere and zoom to fit it.
    // OrbitSystem damps the camera to the new pose. A point selection (lights, empties — radius 0) recenters
    // only, keeping the current zoom, since there's no extent to fit.
    function frameSelection(): void {
        if (editorCamEid < 0) return;
        const b = selectionBounds();
        if (!b) return;
        Orbit.pan.set(editorCamEid, b.center[0], b.center[1], b.center[2], 0);
        if (b.radius < 1e-4) return;
        const canvas = document.querySelector<HTMLCanvasElement>("#canvas");
        const aspect =
            canvas && canvas.clientHeight > 0 ? canvas.clientWidth / canvas.clientHeight : 1;
        if (Camera.mode.get(editorCamEid) === CameraMode.Orthographic) {
            Orbit.size.set(editorCamEid, frameSize(b.radius, aspect));
        } else {
            Orbit.distance.set(
                editorCamEid,
                frameDistance(b.radius, Camera.fov.get(editorCamEid), aspect),
            );
        }
    }

    // scale the edit camera's frustum to the focus distance each frame so a scene at any scale stays
    // un-clipped: get within 1% of the focus (inspect a sub-unit gizmo), see 1000× past it (a terrain).
    // Reverse-Z f32 depth keeps the wide near/far ratio precise. Edit-only — the scene camera's authored
    // planes are untouched in play, so the preview ships as-is.
    const editCameraSystem: System = {
        name: "editor-camera",
        group: "simulation",
        annotations: { mode: "always" },
        update() {
            if (editorCamEid < 0) return;
            // key off the *actual* camera→focus distance (smoothed by OrbitSystem), not the target
            // Orbit.distance — so the planes track the camera through a frame-to-fit ease instead of
            // jumping ahead of it and clipping mid-transition. Works for both projections: the orbit rig
            // positions the camera `distance` from the target in ortho too (zoom changes size, not the
            // distance), so this is the depth-slab scale regardless of mode (ortho size is the view extent,
            // not the depth range — keying near/far off it would clip).
            const focus = Math.max(
                Math.hypot(
                    Transform.pos.x.get(editorCamEid) - Orbit.pan.x.get(editorCamEid),
                    Transform.pos.y.get(editorCamEid) - Orbit.pan.y.get(editorCamEid),
                    Transform.pos.z.get(editorCamEid) - Orbit.pan.z.get(editorCamEid),
                ),
                1e-3,
            );
            Camera.near.set(editorCamEid, Math.max(focus * 0.01, 1e-3));
            Camera.far.set(editorCamEid, Math.max(focus * 1000, 1000));
        },
    };

    // light the handle under the cursor (edit mode, a manipulator tool active) — the same pick the grab
    // uses, so the highlight is exactly what a press would grab.
    function viewportHover(e: PointerEvent): void {
        if (playing || editorCamEid < 0 || !ecs) return;
        // a live drag owns the gizmo: hover stays pinned to the grabbed handle (Handles.active), so passing
        // over another handle mid-drag never lights it. The drag-ownership state is the single gate.
        if (Handles.active >= 0) return;
        const manip = manipulatorFor(tool);
        const origin = manip ? Handles.getOrigin() : null;
        if (!manip || !origin) {
            Handles.hover = -1;
            return;
        }
        const canvas = e.currentTarget as HTMLCanvasElement;
        const rect = canvas.getBoundingClientRect();
        if (rect.width === 0 || rect.height === 0) return;
        computeViewProj(editorCamEid, rect.width / rect.height, _gvp);
        const eye = cursorRay(invert(_gvp, _ginv), rect.width / 2, rect.height / 2, rect.width, rect.height).dir;
        Handles.hover = manip.pick(
            { x: e.clientX - rect.left, y: e.clientY - rect.top },
            origin,
            Handles.getAxes(),
            _gvp,
            rect.width,
            rect.height,
            gizmoWorldScale(origin, rect.height),
            undefined,
            eye,
        );
    }

    // a left-press on a gizmo handle starts a constrained drag, driven by the active tool's Manipulator
    // (lib/gizmo): the entity tracks the cursor (per-axis translate / scale), writing Transform live for
    // feedback and recording the edit into a Document gesture that commits as one undoable unit on release
    // (a multi-entity drag is one undo). Orbit is locked for the drag (both are the left button); Escape
    // aborts. Returns true when it consumed the press, so the viewport pick below it doesn't also fire.
    // Hit-test in CSS px — the same length the gizmo draws (GIZMO_PX), so what's on screen is grabbable.
    function tryGizmoGrab(e: PointerEvent): boolean {
        const manip = manipulatorFor(tool);
        if (editorCamEid < 0 || !ecs || !manip) return false;
        const origin = Handles.getOrigin();
        if (!origin) return false;
        // bind the narrowed value so the drag closures see Vec3, not Vec3 | null
        const anchor = origin;
        const canvas = e.currentTarget as HTMLCanvasElement;
        const rect = canvas.getBoundingClientRect();
        if (rect.width === 0 || rect.height === 0) return false;

        computeViewProj(editorCamEid, rect.width / rect.height, _gvp);
        const inv = invert(_gvp, new Float32Array(16));
        // camera-forward (for plane cull + screen / uniform drags) and the active frame, both captured at
        // grab so the constraint stays fixed for the drag even as the entity rotates (three.js's mousedown
        // capture of space)
        const eye = cursorRay(inv, rect.width / 2, rect.height / 2, rect.width, rect.height).dir;
        const axes = Handles.getAxes();
        // the gizmo's on-screen world size, captured at grab (the camera is locked for the drag) — the
        // uniform-scale handle references it so its sensitivity is zoom-independent
        const gScale = gizmoWorldScale(anchor, rect.height);
        const downX = e.clientX - rect.left;
        const downY = e.clientY - rect.top;
        const handle = manip.pick(
            { x: downX, y: downY },
            anchor,
            axes,
            _gvp,
            rect.width,
            rect.height,
            gScale,
            undefined,
            eye,
        );
        if (handle < 0) return false;

        const start = cursorRay(inv, downX, downY, rect.width, rect.height);

        // each selected entity's pre-drag pose — the input to the Manipulator, and the Escape restore. The
        // .w lanes (unused by the manipulators) are carried so the write-back preserves them.
        const grabbed: { node: Node; eid: number; pose: Pose; posW: number; scaleW: number }[] = [];
        for (const node of doc.selection) {
            const eid = nodeMap.get(node);
            if (eid === undefined || !ecs.has(eid, Transform)) continue;
            grabbed.push({
                node,
                eid,
                pose: {
                    pos: [Transform.pos.x.get(eid), Transform.pos.y.get(eid), Transform.pos.z.get(eid)],
                    rot: [Transform.rot.x.get(eid), Transform.rot.y.get(eid), Transform.rot.z.get(eid), Transform.rot.w.get(eid)],
                    scale: [Transform.scale.x.get(eid), Transform.scale.y.get(eid), Transform.scale.z.get(eid)],
                },
                posW: Transform.pos.w.get(eid),
                scaleW: Transform.scale.w.get(eid),
            });
        }
        if (grabbed.length === 0) return false;

        function write(g: (typeof grabbed)[number], pose: Pose): void {
            Transform.pos.set(g.eid, pose.pos[0], pose.pos[1], pose.pos[2], g.posW);
            Transform.rot.set(g.eid, pose.rot[0], pose.rot[1], pose.rot[2], pose.rot[3]);
            Transform.scale.set(g.eid, pose.scale[0], pose.scale[1], pose.scale[2], g.scaleW);
        }

        Handles.active = handle;
        const prevMode = Orbit.mode.get(editorCamEid);
        Orbit.mode.set(editorCamEid, OrbitMode.Locked);
        canvas.setPointerCapture(e.pointerId);
        doc.begin();

        function teardown(): void {
            canvas.removeEventListener("pointermove", onMove);
            canvas.removeEventListener("pointerup", onUp);
            canvas.removeEventListener("pointercancel", onUp);
            window.removeEventListener("keydown", onKey);
            if (canvas.hasPointerCapture(e.pointerId)) canvas.releasePointerCapture(e.pointerId);
            Orbit.mode.set(editorCamEid, prevMode);
            Handles.active = -1;
        }
        function onMove(ev: PointerEvent) {
            const now = cursorRay(inv, ev.clientX - rect.left, ev.clientY - rect.top, rect.width, rect.height);
            const snap = ev.ctrlKey; // hold Ctrl to snap to grid / 15° / 0.1 step
            for (const g of grabbed) {
                // narrowed non-null at the guard above; svelte-check loses it across this closure
                write(g, manip!.drag(handle, anchor, axes, g.pose, start, now, eye, snap, gScale));
                // record into the gesture (first touch captures the pristine prev; coalesced on commit).
                // formatFields∘readFields matches readback's readComponent, so attr.value won't flip post-commit.
                doc.setAttr(g.node, "transform", formatFields("transform", readFields(Transform, g.eid)));
            }
            // no docVersion bump here — ReadbackSystem mirrors the live ECS to attr.value each frame and
            // bumps it, which is what re-renders the inspector during the drag
        }
        function onUp() {
            teardown();
            doc.commit();
            docVersion++;
        }
        function onKey(ev: KeyboardEvent) {
            if (ev.key !== "Escape") return;
            ev.preventDefault();
            for (const g of grabbed) write(g, g.pose);
            teardown();
            doc.cancel();
            docVersion++;
        }
        canvas.addEventListener("pointermove", onMove);
        canvas.addEventListener("pointerup", onUp);
        // a cancelled pointer (browser/OS interruption) ends the drag too — without this the listeners
        // leak and orbit stays locked
        canvas.addEventListener("pointercancel", onUp);
        window.addEventListener("keydown", onKey);
        return true;
    }

    function viewportPointerDown(e: PointerEvent) {
        if (playing || e.button !== 0) return;
        if (tryGizmoGrab(e)) return;
        picking = true;
        pickDownX = e.clientX;
        pickDownY = e.clientY;
    }

    function viewportPointerUp(e: PointerEvent) {
        // gate on the left button before touching `picking`, so a non-left release mid-press (a
        // right-click during a left-drag) doesn't cancel the pending left-click gesture
        if (e.button !== 0 || !picking) return;
        picking = false;
        if (playing || !ecs) return;
        if (!isClick(pickDownX, pickDownY, e.clientX, e.clientY)) return;
        // shift / ctrl / cmd toggles the picked entity in or out of the selection; a plain click
        // selects only it (or clears on empty space). nextSelection is the pure rule (lib/pick).
        const additive = e.shiftKey || e.ctrlKey || e.metaKey;
        const next = nextSelection([...doc.selection], nodeForEid(Pick.eid, nodeMap), additive);
        doc.clearSelection();
        doc.select(...next);
        docVersion++;
    }

    let renameSignal = $state(0);

    // autosave: flush ~600ms after the last real edit. the effect re-runs every readback frame
    // (docVersion ticks inside `serialized`), so the reschedule is gated on content differing from
    // what's already counting down — without that gate the per-frame tick resets the timer forever and
    // a save never fires. content + path are captured at schedule time and re-validated in flush.
    const SAVE_DEBOUNCE = 600;
    let saveTimer: ReturnType<typeof setTimeout> | null = null;
    let retryTimer: ReturnType<typeof setTimeout> | null = null;
    let lastScheduled: string | null = null;

    $effect(() => {
        if (Persist.mode !== "file") return;
        const content = serialized;
        // reverted to the saved state (or the null-baseline rebuild window) — cancel any pending write,
        // else a timer armed for the transient edit fires and writes content the doc no longer holds
        if (!savedBaseline || content === savedBaseline) {
            if (saveTimer) { clearTimeout(saveTimer); saveTimer = null; }
            lastScheduled = null;
            return;
        }
        if (content === lastScheduled) return;
        lastScheduled = content;
        if (saveTimer) clearTimeout(saveTimer);
        const dir = activeProjectDir;
        const path = activeScenePath;
        saveTimer = setTimeout(() => { saveTimer = null; flush(dir, path, content); }, SAVE_DEBOUNCE);
    });

    async function flush(dir: string | null, path: string | null, content: string): Promise<boolean> {
        if (Persist.mode !== "file" || !dir || !path) return true;
        if (dir !== activeProjectDir || path !== activeScenePath) return true;
        if (content === savedBaseline) return true;
        try {
            await Persist.save(dir, path, content);
            // the doc may have moved on during the await — only claim saved if it still matches
            if (serialized === content) savedBaseline = content;
            clearBanner("save");
            // a successful write is local-wins: it resolves any open external-change conflict, whether the
            // author clicked Keep mine or just kept editing (the next autosave overwrites the disk version)
            clearBanner("conflict");
            return true;
        } catch (e) {
            console.error("Autosave failed:", e);
            banner("save", "error", "Couldn't save — retrying…");
            scheduleRetry(dir, path);
            return false;
        }
    }

    // backoff governs *when*; the flush guard governs *what*. retry the current content, not the
    // stale payload that failed.
    function scheduleRetry(dir: string, path: string) {
        if (retryTimer) clearTimeout(retryTimer);
        retryTimer = setTimeout(() => {
            retryTimer = null;
            if (Persist.mode !== "file" || dir !== activeProjectDir || path !== activeScenePath) return;
            flush(dir, path, serialized);
        }, 2000);
    }

    async function flushNow(): Promise<boolean> {
        if (saveTimer) { clearTimeout(saveTimer); saveTimer = null; }
        return flush(activeProjectDir, activeScenePath, serialized);
    }

    async function handleSave() {
        await flushNow();
    }

    // the beforeunload guard catches *accidental* navigation away from unsaved work. An intentional
    // discard-reload (the conflict banner's Reload) sets this first, so it isn't double-confirmed by the
    // browser's generic "leave site?" on top of the choice the author already made.
    let discarding = false;
    window.addEventListener("beforeunload", (e) => {
        if (unsaved && !discarding) {
            e.preventDefault();
            e.returnValue = "";
        }
    });

    function handleKeydown(e: KeyboardEvent) {
        if (e.key === "Escape" && docsOpen) {
            docsOpen = false;
            e.preventDefault();
            return;
        }
        const shortcut = resolveShortcut(e);
        if (shortcut === "help") {
            e.preventDefault();
            toggleDocs();
            return;
        }
        if (shortcut) {
            if (shortcut === "rename" && doc.selection.size !== 1) return;
            e.preventDefault();
            switch (shortcut) {
                case "save": handleSave(); break;
                case "undo": handleUndo(); break;
                case "redo": handleRedo(); break;
                case "toggle-sidebar": {
                    const next = !(outlinerCollapsed && inspectorCollapsed);
                    setCollapsed(next, next);
                    break;
                }
                case "rename": renameSignal++; break;
                case "delete": deleteSelected(); break;
            }
            return;
        }
        // edit-mode viewport hotkeys: never while typing or under a modifier. The fly camera owns WASD/QE,
        // tools sit on the number row (tool.ts keys); F frames the selection; FRAME_KEY (X) cycles the
        // handle orientation. Each key is declared with its control, the one source of truth for the tooltip.
        if (!playing && !inField(e) && !(e.ctrlKey || e.metaKey || e.altKey)) {
            const t = toolForKey(e.key);
            if (t !== null) {
                tool = t;
                e.preventDefault();
                return;
            }
            const k = e.key.toLowerCase();
            if (k === "f") {
                frameSelection();
                e.preventDefault();
            } else if (k === FRAME_KEY && tool !== Tool.Scale) {
                coordFrame = nextFrame(coordFrame); // Scale is local-only, so X is inert there
                e.preventDefault();
            } else if (k === PIVOT_KEY) {
                pivotMode = nextPivot(pivotMode);
                e.preventDefault();
            }
        }
    }

    async function initProject() {
        if (project.dir) {
            activeProjectDir = project.dir;
            activeScenes = project.scenes;
            const lastPath = prefs.lastScene?.[project.dir];
            const restorePath =
                (lastPath && project.scenes.includes(lastPath) ? lastPath : null) ??
                matchScene(project.scene ?? undefined, project.scenes) ??
                project.scenes[0] ??
                null;
            if (restorePath) {
                activeScenePath = restorePath;
                activeSceneContent = await Persist.load(project.dir, restorePath);
            }
            return;
        }
    }

    initProject();

    $effect(() => {
        const mode: "edit" | "play" = playing ? "play" : "edit";
        // track the rebuild triggers (the manifest enablement, explicit version, scene); read the plugin
        // objects untracked so a hot swap (which reassigns `locals` with fresh objects) doesn't retrigger this
        // effect and tear down the live State it just patched. A toggle rebuilds via `manifest` + `version`.
        void manifest;
        void version;
        const nodes = doc.nodes;
        const plugins = untrack(() => activePlugins);
        // the project (local) subset of what's built — the swap's `prev`, snapshotted with `plugins`
        const projectSubset = untrack(() => localPlugins(entries));

        savedBaseline = null;
        let mounted = true;
        let currentApp: App | null = null;

        const p = (async () => {
            currentApp = await buildState(mode, plugins, nodes);
            const state = currentApp.state;
            if (mode === "edit") {
                // restore the saved orbit pose; OrbitSystem snaps OrbitSmooth to it on the first frame
                // (its not(OrbitSmooth) add path), so no remove is needed — the fresh build State has no
                // OrbitSmooth membership yet
                restoreOrbit(editorCamEid);
            }
            if (!mounted) { currentApp.dispose(); return; }
            ecs = state;
            builtProject = projectSubset;
            builtSkipped = currentApp.skipped;
            clearBanner("build");
            docVersion++;
        })().catch((e: unknown) => {
            // a failed build reports and leaves the editor responsive; a rejected _prevBuild would
            // otherwise rethrow into every later build and wedge rebuilds permanently. The scene won't
            // run, so it's a blocking banner, not a transient toast.
            banner("build", "error", `Build failed: ${e instanceof Error ? e.message : String(e)}`);
            console.error("[shallot] build failed:", e);
        });
        _prevBuild = p;

        return () => {
            mounted = false;
            if (editorCamEid >= 0) saveOrbit(editorCamEid);
            currentApp?.dispose();
            ecs = null;
            session = null;
        };
    });

    $effect(() => {
        void docVersion;
        if (!ecs || playing || editorCamEid < 0) return;
        for (const [, eid] of nodeMap.entries()) {
            if (ecs.has(eid, Camera)) {
                syncCameraEffects(ecs, eid, editorCamEid);
                break;
            }
        }
    });

    if (import.meta.hot) {
        const strip = (s: string) => s.replace(/\x1b\[[0-9;]*m/g, "");
        import.meta.hot.on("shallot:log", ({ level, message }: { level: string; message: string }) => {
            const clean = strip(message);
            if (level === "error") console.error(clean);
            else if (level === "warn") console.warn(clean);
            else console.log(clean);
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
                        result = applyCommand(doc, payload as CommandPayload, {
                            undo: handleUndo,
                            redo: handleRedo,
                            sync: syncUndoRedo,
                            bump: () => { docVersion++; },
                        });
                        break;
                    case "entities":
                        result = queryEntities(ecs, payload as { component?: string; eid?: number } | null);
                        break;
                    default:
                        result = { error: `Unknown request type: ${type}` };
                }
            } catch (e: unknown) {
                result = { error: e instanceof Error ? e.message : String(e) };
            }
            import.meta.hot!.send("shallot:response", { id, result });
        });

        // hot reload: re-imported local plugins from the `virtual:project` accept. Swap them onto the live
        // State in place; a shape change the swap can't carry forces a rebuild via `version`. `prev` is
        // `builtProject` (the scheduler's live system objects), never the last-swapped set. The manifest is
        // unchanged on a code edit (only the plugin code moved), so enablement holds.
        onProjectReload(async ({ locals: nextLocals, manifest: nextManifest }) => {
            const state = ecs;
            if (!state) return;
            locals = nextLocals;
            const next = localPlugins(entriesFor(nextManifest, nextLocals));
            const result = await swap(state, builtProject, next, builtSkipped);
            if (result.ok) {
                docVersion++;
            } else {
                console.log(`[shallot] reload needs a rebuild: ${result.reason}`);
                version++;
            }
        });

        // a model asset (.glb/.gltf) changed on disk (the dev server's public-dir watcher). The asset is
        // not the editor's document — the nodes referencing it by name re-spawn unchanged — so re-decoding
        // it can never clobber unsaved edits, and we drop it from the glTF cache + rebuild unconditionally
        // (no conflict weighing, no page reload). The freed GPU resources release behind the old State's
        // submit fence; the `version` rebuild's preload re-decodes off-thread and re-registers before the
        // next frame (invalidate's contract).
        import.meta.hot.on("shallot:asset", ({ src }: { src: string }) => {
            invalidate(src);
            version++;
        });

        // a scene/manifest write the editor didn't make (IDE, dev mode, git). The dev server already
        // filtered out our own writes; here we weigh it against unsaved work. Clean → reload to pick it up.
        // Mid-edit → cancel the pending autosave so it can't clobber the external version, and surface the
        // choice (Reload takes disk, Keep mine writes local over it) — never the silent reload that drops
        // the edits + session state. `unsaved` is false in ephemeral mode, so an ephemeral session just reloads.
        import.meta.hot.on("shallot:external", ({ path, manifest: isManifest }: { path: string; manifest: boolean }) => {
            const change = classifyExternal({ path, manifest: isManifest, scene: activeScenePath, unsaved });
            if (change === "ignore") return;
            if (change === "reload") {
                location.reload();
                return;
            }
            if (saveTimer) { clearTimeout(saveTimer); saveTimer = null; }
            lastScheduled = null;
            banner(
                "conflict",
                "warning",
                `${isManifest ? "Project" : "Scene"} changed on disk while you have unsaved edits.`,
                [
                    { label: "Reload", fn: () => { discarding = true; location.reload(); } },
                    { label: "Keep mine", fn: () => { void flushNow(); } },
                ],
            );
        });
    }
</script>

<svelte:head>
    <title>{unsaved ? "• " : ""}{activeProjectDir ? activeProjectDir.split("/").pop() + " | " : ""}shallot</title>
</svelte:head>
<svelte:window onkeydown={handleKeydown} />

<div class="editor" class:playing bind:this={editorEl}>
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
                    <button class="dropdown-item" class:active={(editMask & Overlay.Grid) !== 0} onclick={() => toggleOverlay(Overlay.Grid)}>
                        <svg viewBox="0 0 16 16" fill="none" stroke="currentColor" stroke-width="1.2">
                            <line x1="1" y1="4" x2="15" y2="4" />
                            <line x1="1" y1="8" x2="15" y2="8" />
                            <line x1="1" y1="12" x2="15" y2="12" />
                            <line x1="4" y1="1" x2="4" y2="15" />
                            <line x1="8" y1="1" x2="8" y2="15" />
                            <line x1="12" y1="1" x2="12" y2="15" />
                        </svg>
                        <span>Grid</span>
                        {#if (editMask & Overlay.Grid) !== 0}
                            <svg class="check" viewBox="0 0 16 16" fill="none" stroke="currentColor" stroke-width="2">
                                <path d="M3 8.5 L6.5 12 L13 4" />
                            </svg>
                        {/if}
                    </button>
                </div>
            {/if}
        </div>
    {/snippet}

    {#snippet toolGroup()}
        <div class="tool-group" role="toolbar" aria-label="Transform tool">
            {#each TOOLS as t (t.id)}
                <button
                    class="viewport-btn tool-btn"
                    class:active={tool === t.id}
                    aria-pressed={tool === t.id}
                    title="{t.label}{hint(t.key)}"
                    onclick={() => { tool = t.id; }}
                >
                    <Icon icon={t.icon} size={15} strokeWidth={1.6} />
                </button>
            {/each}
        </div>
    {/snippet}

    <!-- handle orientation (World / Local) — a contextual option of the active tool (Move / Rotate); Scale
         is local-only so it never shows this. -->
    {#snippet frameMenu()}
        <div class="gizmos-dropdown">
            <button
                class="viewport-btn option-btn"
                class:active={gizmosOpen === "frame"}
                onclick={() => openGizmos("frame")}
                title={`Orientation: ${currentFrame.label}${hint(FRAME_KEY)}`}
            >
                <Icon icon={currentFrame.icon} size={14} strokeWidth={1.6} />
                <svg class="caret" viewBox="0 0 8 6" fill="currentColor">
                    <path d="M0.5 0.5 L4 5 L7.5 0.5" />
                </svg>
            </button>
            {#if gizmosOpen === "frame"}
                <div class="dropdown-menu start">
                    {#each FRAMES as f (f.id)}
                        <button class="dropdown-item" class:active={coordFrame === f.id} onclick={() => { coordFrame = f.id; gizmosOpen = null; }}>
                            <Icon icon={f.icon} size={14} strokeWidth={1.4} />
                            <span>{f.label}</span>
                            {#if coordFrame === f.id}
                                <svg class="check" viewBox="0 0 16 16" fill="none" stroke="currentColor" stroke-width="2">
                                    <path d="M3 8.5 L6.5 12 L13 4" />
                                </svg>
                            {/if}
                        </button>
                    {/each}
                </div>
            {/if}
        </div>
    {/snippet}

    <!-- pivot point (Median / Active) — a contextual option of the active tool (Rotate / Scale): the anchor
         the gizmo orbits / scales about for a multi-entity selection. -->
    {#snippet pivotMenu()}
        <div class="gizmos-dropdown">
            <button
                class="viewport-btn option-btn"
                class:active={gizmosOpen === "pivot"}
                onclick={() => openGizmos("pivot")}
                title={`Pivot: ${currentPivot.label}${hint(PIVOT_KEY)}`}
            >
                <Icon icon={currentPivot.icon} size={14} strokeWidth={1.6} />
                <svg class="caret" viewBox="0 0 8 6" fill="currentColor">
                    <path d="M0.5 0.5 L4 5 L7.5 0.5" />
                </svg>
            </button>
            {#if gizmosOpen === "pivot"}
                <div class="dropdown-menu start">
                    {#each PIVOTS as p (p.id)}
                        <button class="dropdown-item" class:active={pivotMode === p.id} onclick={() => { pivotMode = p.id; gizmosOpen = null; }}>
                            <Icon icon={p.icon} size={14} strokeWidth={1.4} />
                            <span>{p.label}</span>
                            {#if pivotMode === p.id}
                                <svg class="check" viewBox="0 0 16 16" fill="none" stroke="currentColor" stroke-width="2">
                                    <path d="M3 8.5 L6.5 12 L13 4" />
                                </svg>
                            {/if}
                        </button>
                    {/each}
                </div>
            {/if}
        </div>
    {/snippet}

    <main class="workspace" class:resizing bind:this={workspace}>
        <aside class="sidebar panel" class:collapsed={outlinerCollapsed} style="width: {outlinerCollapsed ? 0 : outlinerWidth}px">
            <div class="sidebar-header">
                <MenuBar
                    {entries}
                    ontoggle={togglePlugin}
                    hasProject={!!activeProjectDir}
                    scenes={activeScenes}
                    activeScene={activeScenePath}
                    onscene={switchScene}
                    onundo={handleUndo}
                    onredo={handleRedo}
                    onsave={handleSave}
                    ondocs={openDocsIndex}
                    canUndo={docVersion >= 0 && doc.history.undo.length > 0}
                    canRedo={docVersion >= 0 && doc.history.redo.length > 0}
                    canSave={unsaved}
                    themes={THEMES}
                    activeTheme={themeId}
                    ontheme={selectTheme}
                />
            </div>
            <Outliner {doc} version={docVersion} {diagnostics} {renameSignal} onselect={() => docVersion++} oncreate={createEntity} onimport={openImportPicker} ondelete={deleteEntity} onreorder={() => docVersion++} onrename={() => docVersion++} />
            <input class="import-input" type="file" multiple accept=".glb,.gltf,.bin,.png,.jpg,.jpeg,.webp,.ktx2" bind:this={importInput} onchange={handleImportPick} />
        </aside>
        {#if !outlinerCollapsed}
            <div class="resizer resizer-h" role="separator" aria-orientation="vertical" onpointerdown={(e) => startResize("outliner", e)}></div>
        {/if}

        <section class="viewport">
            <div class="viewport-bar">
                <div class="bar-left">
                    {#if !playing}
                        {@render toolGroup()}
                        <!-- contextual options for the active tool — subordinate to the tool group (a
                             divider + lighter chips), shown only where the option changes the result -->
                        {#if showFrame || showPivot}
                            <div class="bar-divider" role="separator" aria-orientation="vertical"></div>
                            <div class="tool-options">
                                {#if showFrame}{@render frameMenu()}{/if}
                                {#if showPivot}{@render pivotMenu()}{/if}
                            </div>
                        {/if}
                    {/if}
                </div>
                <div class="bar-center">
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
                <div class="bar-right">
                    <Issues {diagnostics} onselect={selectIssue} />
                    {#if Persist.mode === "ephemeral"}
                        <span class="no-save" title="Changes won't be saved"><Icon icon={SaveOff} size={14} strokeWidth={1.6} /></span>
                    {/if}
                    <!-- overlays are edit-only Scene-View chrome; play is a faithful preview (no gizmos) -->
                    {#if !playing}
                        {@render gizmosMenu()}
                    {/if}
                </div>
            </div>
            <div class="viewport-canvas" role="region" aria-label="Scene viewport" bind:this={viewportCanvas} ondragenter={handleDragEnter} ondragover={handleDragOver} ondragleave={handleDragLeave} ondrop={handleDrop}>
                <canvas id="canvas" onpointerdown={viewportPointerDown} onpointerup={viewportPointerUp} onpointermove={viewportHover} onpointerleave={() => (Handles.hover = -1)}></canvas>
                {#if dropActive}
                    <div class="drop-overlay">
                        <span class="drop-overlay-label">Drop model to import</span>
                    </div>
                {/if}
                <Banners />
                <Toasts />
            </div>
        </section>

        {#if !inspectorCollapsed}
            <div class="resizer resizer-h" role="separator" aria-orientation="vertical" onpointerdown={(e) => startResize("inspector", e)}></div>
        {/if}
        <aside class="sidebar panel" class:collapsed={inspectorCollapsed} style="width: {inspectorCollapsed ? 0 : inspectorWidth}px">
            <Inspector {doc} version={docVersion} {diagnostics} {ecs} {nodeMap} docsFor={(c) => docFor(docIndex, c)} ondocs={openDocs} onchange={() => docVersion++} onsync={syncField} onremove={removeField} onadd={addField} onreorder={() => docVersion++} />
        </aside>

        <!-- the docs reader is a summoned overlay drawer pinned to the editor's right edge: it floats over
             the canvas + inspector (no layout reflow on open), resizes from its own left handle, and
             dismisses back to the untouched inspector -->
        {#if docsOpen}
            <div class="docs-overlay" style="width: {docsWidth}px">
                <div class="resizer resizer-h docs-resizer" role="separator" aria-orientation="vertical" onpointerdown={(e) => startResize("docs", e)}></div>
                <Docs index={docIndex} target={docsTarget} onclose={() => (docsOpen = false)} />
            </div>
        {/if}
    </main>
</div>

<style>
    .editor {
        /* color tokens are applied at runtime from lib/theme.ts (the single palette source); only
           non-color design tokens live here */
        --ease-out: cubic-bezier(0.34, 0, 0, 1);
        --header-h: 40px;

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


    /* flush top bar: the transport + gizmos sit in chrome the game view renders below, never over it.
       Three clusters on a 1fr-auto-1fr grid so the transport stays optically centered as the side
       clusters change width. */
    .viewport-bar {
        flex: none;
        display: grid;
        grid-template-columns: 1fr auto 1fr;
        align-items: center;
        height: var(--header-h);
        padding: 0 10px;
        background: var(--surface-1-solid);
        border-bottom: 1px solid var(--border);
        z-index: 1;
    }

    .bar-left {
        display: flex;
        align-items: center;
        gap: 6px;
        justify-self: start;
    }

    /* separates the persistent tool group from the active tool's contextual options */
    .bar-divider {
        width: 1px;
        height: 16px;
        background: var(--border);
    }

    /* the active tool's options — a subordinate cluster, read as settings of the tool, not peer tools */
    .tool-options {
        display: flex;
        align-items: center;
        gap: 2px;
    }

    /* option chips sit quieter than the filled tool group — they qualify the active tool, not switch it */
    .option-btn {
        opacity: 0.8;
    }

    .option-btn:hover,
    .option-btn.active {
        opacity: 1;
    }

    .bar-center {
        justify-self: center;
    }

    .bar-right {
        display: flex;
        align-items: center;
        gap: 6px;
        justify-self: end;
    }

    /* the ephemeral-only (?save=off) no-save indicator: deliberately the muted glyph alone, no label. */
    .no-save {
        display: inline-flex;
        align-items: center;
        color: var(--text-muted);
    }

    .transport-btn {
        display: flex;
        align-items: center;
        justify-content: center;
        width: 30px;
        height: 28px;
        border: 1px solid transparent;
        border-radius: 5px;
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
    }

    .transport-btn.play:hover {
        color: var(--accent-hover);
        border-color: var(--accent);
    }

    .transport-btn:active {
        background: color-mix(in srgb, var(--accent) 8%, transparent);
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
        animation: stop-appear 200ms var(--ease-out);
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
        display: flex;
        position: relative;
    }

    /* the summoned docs drawer: floats over the right of the canvas + inspector, so opening it reflows
       nothing behind it. Opaque (it sits over the live viewport), elevated by a left border + shadow. */
    .docs-overlay {
        position: absolute;
        top: 0;
        right: 0;
        bottom: 0;
        display: flex;
        background: var(--surface-1-solid);
        border-left: 1px solid var(--border);
        box-shadow: -8px 0 24px rgba(0, 0, 0, 0.28);
        z-index: 40;
        overflow: hidden;
        animation: docs-slide-in 150ms var(--ease-out);
    }

    .docs-overlay > :global(.docs-reader) {
        flex: 1;
        min-width: 0;
    }

    @keyframes docs-slide-in {
        from { opacity: 0; transform: translateX(8px); }
        to { opacity: 1; transform: translateX(0); }
    }

    .panel {
        flex: none;
        transition: width 180ms var(--ease-out);
    }

    .panel.collapsed {
        overflow: hidden;
    }

    .workspace.resizing .panel {
        transition: none;
    }

    .resizer {
        flex: none;
        position: relative;
        width: 1px;
        background: var(--border);
        cursor: col-resize;
        touch-action: none;
        z-index: 2;
        transition: background 150ms var(--ease-out);
    }

    .resizer::before {
        content: "";
        position: absolute;
        top: 0;
        bottom: 0;
        left: -3px;
        right: -3px;
    }

    .resizer:hover,
    .workspace.resizing .resizer {
        background: var(--accent);
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
        display: flex;
        align-items: center;
        gap: 6px;
        height: var(--header-h);
        padding: 0 10px;
        color: var(--accent);
        position: sticky;
        top: 0;
        background: var(--surface-1-solid);
        z-index: 1;
    }

    .viewport {
        flex: 1;
        min-width: 0;
        height: 100%;
        display: flex;
        flex-direction: column;
        min-height: 0;
        position: relative;
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
        background: color-mix(in srgb, var(--accent) 8%, transparent);
        transform: scale(0.95);
    }

    .viewport-btn:focus-visible {
        outline: 1px solid var(--accent);
        outline-offset: -1px;
    }

    .viewport-btn:disabled {
        opacity: 0.4;
        cursor: default;
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

    /* transform-tool segmented group: buttons share the .viewport-btn substrate (size, press, focus),
       lifted one surface level inside the group fill so hover/active read against it */
    .tool-group {
        display: inline-flex;
        align-items: center;
        gap: 1px;
        padding: 2px;
        border-radius: 6px;
        background: var(--surface-2);
    }

    .tool-btn {
        width: 26px;
        padding: 0;
        justify-content: center;
    }

    .tool-btn:hover {
        background: var(--surface-3);
    }

    .tool-btn.active {
        background: var(--surface-4);
        color: var(--accent);
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
        background: var(--surface-3-solid);
        box-shadow: 0 4px 16px rgba(0, 0, 0, 0.3), 0 1px 4px rgba(0, 0, 0, 0.2);
        transform-origin: top right;
        animation: dropdown-appear 150ms var(--ease-out);
    }

    /* a left-cluster dropdown (the frame menu) opens from the button's left edge, not the right */
    .dropdown-menu.start {
        right: auto;
        left: 0;
        transform-origin: top left;
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
        background: color-mix(in srgb, var(--accent) 8%, transparent);
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
        border: 2px solid color-mix(in srgb, var(--accent) 40%, transparent);
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

    /* the drop target: an accent-tinted wash + dashed inset border over the viewport while a file drag
       hovers. pointer-events: none — the drop lands on the container, the overlay only signals */
    .drop-overlay {
        position: absolute;
        inset: 8px;
        display: flex;
        align-items: center;
        justify-content: center;
        border: 2px dashed color-mix(in srgb, var(--accent) 60%, transparent);
        border-radius: 8px;
        background: color-mix(in srgb, var(--accent) 8%, transparent);
        pointer-events: none;
        z-index: 2;
        animation: ui-overlay-in 150ms var(--ease-out);
    }

    .drop-overlay-label {
        padding: 6px 14px;
        border-radius: 6px;
        background: var(--surface-2-solid);
        color: var(--text);
        font-size: 13px;
    }

    .import-input {
        display: none;
    }

    @keyframes ui-overlay-in {
        from { opacity: 0; }
        to { opacity: 1; }
    }


</style>
