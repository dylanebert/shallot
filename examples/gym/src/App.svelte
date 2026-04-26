<script lang="ts">
    import { PaneGroup, Pane, PaneResizer } from "paneforge";
    import type { GymInit } from "./bridge";
    import * as bridge from "./bridge";
    import {
        activeScenarioName,
        capabilities,
        urlPipeline,
        urlEffects,
        urlLighting,
        urlShape,
        urlVariant,
        urlLayout,
        urlCamera,
        urlPhysicsTest,
        urlShapes,
        PIPELINES,
        CAMERA_EFFECTS,
        ENV_EFFECTS,
        AUDIO_ROOMS,
        RENDER_TEST_SHAPES,
        RENDER_TEST_VARIANTS,
        RENDER_TEST_LIGHTING,
        PHYSICS_TEST_VARIANTS,
        PILE_SHAPES,
        urlAudioRoom,
        urlSources,
        urlRamp,
        urlText,
        urlArrow,
    } from "./lib";
    import type { Pipeline, CameraModeName, LayoutName, RenderTestShape, RenderTestVariant, RenderTestLighting, PhysicsTestVariant, PileShape, AudioRoom } from "./lib";
    import { gymState, setInitCallback } from "./state.svelte";
    import { Benchmark } from "./lib";
    import { SCENARIOS } from "./capabilities";

    let maxCapacity = $state(16_777_216);

    let currentPipeline = $state<Pipeline>(urlPipeline ?? "raster");
    let currentCamera = $state<CameraModeName>(urlCamera);
    let currentLayout = $state<LayoutName>(urlLayout);
    let currentRenderShape = $state<RenderTestShape>(urlShape ?? "box");
    let currentRenderVariant = $state<RenderTestVariant>(urlVariant ?? "default");
    let objectCount = $state(0);

    let effectsOpen = $state(true);
    let activeEffects = $state<Set<string>>(new Set(urlEffects));

    let currentLightingMode = $state<RenderTestLighting>(urlLighting ?? "directional");

    let textEnabled = $state(urlText);
    let arrowEnabled = $state(urlArrow);

    let currentPhysicsTest = $state<PhysicsTestVariant>(urlPhysicsTest ?? "box");
    let activePileShapes = $state<Set<PileShape>>(new Set(urlShapes ?? [0]));

    let currentAudioRoom = $state<AudioRoom>(urlAudioRoom ?? "living");
    let audioSourceCount = $state(urlSources);

    function ecs() {
        return gymState.ecs;
    }

    function reloadWithParam(key: string, value: string | null) {
        const p = new URLSearchParams(location.search);
        if (value === null) {
            p.delete(key);
        } else {
            p.set(key, value);
        }
        location.search = p.toString();
    }

    function selectScenario(id: string | null) {
        const current = activeScenarioName;
        if (id === current) return;
        const p = new URLSearchParams(location.search);
        if (id === null) {
            p.delete("scenario");
        } else {
            p.set("scenario", id);
        }
        for (const key of [
            "test", "shape", "variant", "shapes", "count", "height",
            "effects", "lighting", "room", "sources", "text", "arrow",
            "camera", "layout",
        ]) {
            p.delete(key);
        }
        location.search = p.toString();
    }

    function toggleRamp() {
        const p = new URLSearchParams(location.search);
        if (p.has("ramp")) {
            p.delete("ramp");
        } else {
            p.set("ramp", "");
        }
        location.search = p.toString();
    }

    function togglePileShape(id: PileShape) {
        if (activePileShapes.has(id)) {
            if (activePileShapes.size <= 1) return;
            activePileShapes.delete(id);
        } else {
            activePileShapes.add(id);
        }
        activePileShapes = new Set(activePileShapes);
        const sorted = [...activePileShapes].sort();
        if (!urlRamp) {
            reloadWithParam("shapes", sorted.join(","));
            return;
        }
        bridge.setBridgePileShapes(sorted);
    }

    function setPipeline(p: Pipeline) {
        if (!urlRamp) {
            reloadWithParam("pipeline", p);
            return;
        }
        const s = ecs();
        if (!s) return;
        currentPipeline = p;
        bridge.setPipeline(s, p);
    }

    function setCameraMode(mode: CameraModeName) {
        if (!urlRamp) {
            reloadWithParam("camera", mode);
            return;
        }
        const s = ecs();
        if (!s) return;
        currentCamera = mode;
        bridge.setCameraMode(s, mode);
    }

    function setLayout(mode: LayoutName) {
        if (!urlRamp) {
            reloadWithParam("layout", mode);
            return;
        }
        const s = ecs();
        if (!s) return;
        currentLayout = mode;
        bridge.setLayout(s, mode);
    }

    function onCountChange(e: Event) {
        const input = e.target as HTMLInputElement;
        const n = Math.max(1, Math.min(maxCapacity, parseInt(input.value) || 1));
        if (!urlRamp) {
            reloadWithParam("count", String(n));
            return;
        }
        const s = ecs();
        if (!s) return;
        objectCount = n;
        input.value = String(objectCount);
        bridge.setCount(s, objectCount);
    }

    function toggleEffect(name: string) {
        if (activeEffects.has(name)) {
            activeEffects.delete(name);
        } else {
            activeEffects.add(name);
        }
        activeEffects = new Set(activeEffects);
        if (!urlRamp) {
            const joined = [...activeEffects].join(",");
            reloadWithParam("effects", joined || null);
            return;
        }
        applyEffects();
    }

    function applyEffects() {
        const s = ecs();
        if (!s) return;
        bridge.setEffects(s, [...activeEffects]);
    }

    function selectRenderShape(shape: RenderTestShape) {
        if (!urlRamp) {
            reloadWithParam("shape", shape);
            return;
        }
        currentRenderShape = shape;
        bridge.setRenderShape(shape);
    }

    function selectRenderVariant(variant: RenderTestVariant) {
        if (!urlRamp) {
            reloadWithParam("variant", variant);
            return;
        }
        currentRenderVariant = variant;
        bridge.setRenderVariant(variant);
    }

    function selectPhysicsTest(name: PhysicsTestVariant) {
        if (!urlRamp) {
            reloadWithParam("test", name);
            return;
        }
        currentPhysicsTest = name;
        bridge.setPhysicsTest(name);
    }

    function selectAudioRoom(room: AudioRoom) {
        if (!urlRamp) {
            reloadWithParam("room", room);
            return;
        }
        currentAudioRoom = room;
        bridge.setRoom(room);
    }

    function onSourceCountChange(e: Event) {
        const input = e.target as HTMLInputElement;
        const n = Math.max(1, Math.min(64, parseInt(input.value) || 1));
        audioSourceCount = n;
        input.value = String(n);
        reloadWithParam("sources", String(n));
    }

    function selectLightingMode(mode: RenderTestLighting) {
        if (!urlRamp) {
            reloadWithParam("lighting", mode);
            return;
        }
        currentLightingMode = mode;
        if (capabilities.renderTestShapes) {
            bridge.setRenderLighting(mode);
            return;
        }
        for (const name of ["nosun", "pl1", "pl2", "pl3", "pl4"]) {
            activeEffects.delete(name);
        }
        if (mode === "point" || mode === "multipoint") {
            activeEffects.add("nosun");
        }
        if (mode === "point" || mode === "dir+pt") {
            activeEffects.add("pl1");
        }
        if (mode === "multipoint") {
            activeEffects.add("pl1");
            activeEffects.add("pl2");
            activeEffects.add("pl3");
            activeEffects.add("pl4");
        }
        activeEffects = new Set(activeEffects);
        applyEffects();
        bridge.syncParam("lighting", mode);
    }

    function toggleText() {
        textEnabled = !textEnabled;
        if (!urlRamp) {
            reloadWithParam("text", textEnabled ? "" : null);
            return;
        }
        bridge.setRenderText(textEnabled);
    }

    function toggleArrow() {
        arrowEnabled = !arrowEnabled;
        if (!urlRamp) {
            reloadWithParam("arrow", arrowEnabled ? "" : null);
            return;
        }
        bridge.setRenderArrow(arrowEnabled);
    }

    setInitCallback((data: GymInit) => {
        if (!urlPipeline) currentPipeline = data.pipeline;
        objectCount = data.objectCount;
        maxCapacity = data.maxCapacity;
    });

    $effect(() => {
        if (!urlRamp) return;
        const id = setInterval(() => {
            const s = gymState.ecs;
            if (!s) return;
            const eid = s.only([Benchmark]);
            if (eid >= 0) objectCount = Benchmark.count[eid];
        }, 250);
        return () => clearInterval(id);
    });

    const scenarioLabel = SCENARIOS.find(s => s.id === activeScenarioName)?.label ?? activeScenarioName ?? "benchmark";
</script>

<div class="gym-root bench">
    <PaneGroup direction="horizontal">
        <Pane defaultSize={80} minSize={40}>
            <div class="canvas-pane">
                <canvas id="canvas"></canvas>
            </div>
        </Pane>
        <PaneResizer class="resizer resizer-h" />
        <Pane
            defaultSize={20}
            minSize={12}
            maxSize={35}
            collapsible={true}
            collapsedSize={0}
        >
            <div class="sidebar">
                <div class="sidebar-header">
                    <span class="scenario-label">{scenarioLabel}</span>
                    <span class="pipeline-badge">{currentPipeline}</span>
                </div>

                <div class="sidebar-body">
                    <div class="section">
                        <div class="section-label">scenario</div>
                        <select
                            class="scenario-select"
                            value={activeScenarioName ?? ""}
                            onchange={(e) => selectScenario((e.target as HTMLSelectElement).value || null)}
                        >
                            {#each SCENARIOS as s}
                                <option value={s.id ?? ""}>{s.label}</option>
                            {/each}
                        </select>
                    </div>

                    <div class="section">
                        <div class="section-label">mode</div>
                        <div class="segmented-row">
                            <button
                                class="seg"
                                class:active={!urlRamp}
                                onclick={() => { if (urlRamp) toggleRamp(); }}
                            >
                                reload
                            </button>
                            <button
                                class="seg"
                                class:active={urlRamp}
                                onclick={() => { if (!urlRamp) toggleRamp(); }}
                            >
                                ramp
                            </button>
                        </div>
                    </div>

                    <div class="section">
                        <div class="section-label">pipeline</div>
                        <div class="segmented-row">
                            {#each PIPELINES as p}
                                <button
                                    class="seg"
                                    class:active={currentPipeline === p}
                                    onclick={() => setPipeline(p)}
                                >
                                    {p}
                                </button>
                            {/each}
                        </div>
                    </div>

                    {#if capabilities.pileShapes}
                        <div class="section">
                            <div class="section-label">shapes</div>
                            <div class="segmented-row">
                                {#each PILE_SHAPES as shape}
                                    <button
                                        class="seg"
                                        class:active={activePileShapes.has(shape.id)}
                                        onclick={() => togglePileShape(shape.id)}
                                    >
                                        {shape.label}
                                    </button>
                                {/each}
                            </div>
                        </div>
                    {/if}

                    {#if capabilities.audioRoom}
                        <div class="section">
                            <div class="section-label">room</div>
                            <div class="segmented">
                                {#each AUDIO_ROOMS as room}
                                    <button
                                        class="seg"
                                        class:active={currentAudioRoom === room}
                                        onclick={() => selectAudioRoom(room)}
                                    >
                                        {room}
                                    </button>
                                {/each}
                            </div>
                        </div>
                    {/if}

                    {#if capabilities.audioSources}
                        <div class="section count-input">
                            <div class="section-label">sources</div>
                            <input
                                type="number"
                                min="1"
                                max="64"
                                step="1"
                                value={audioSourceCount}
                                onchange={onSourceCountChange}
                            />
                        </div>
                    {/if}

                    {#if capabilities.physicsTestVariants}
                        <div class="section">
                            <div class="section-label">variant</div>
                            <div class="segmented">
                                {#each PHYSICS_TEST_VARIANTS as name}
                                    <button
                                        class="seg"
                                        class:active={currentPhysicsTest === name}
                                        onclick={() => selectPhysicsTest(name)}
                                    >
                                        {name}
                                    </button>
                                {/each}
                            </div>
                        </div>
                    {/if}

                    {#if capabilities.cameraMode}
                        <div class="section">
                            <div class="section-label">camera</div>
                            <div class="segmented-row">
                                <button
                                    class="seg"
                                    class:active={currentCamera === "static"}
                                    onclick={() => setCameraMode("static")}
                                >
                                    static
                                </button>
                                <button
                                    class="seg"
                                    class:active={currentCamera === "pan"}
                                    onclick={() => setCameraMode("pan")}
                                >
                                    pan
                                </button>
                            </div>
                        </div>
                    {/if}

                    {#if capabilities.layout}
                        <div class="section">
                            <div class="section-label">layout</div>
                            <div class="segmented-row">
                                <button
                                    class="seg"
                                    class:active={currentLayout === "lorenz"}
                                    onclick={() => setLayout("lorenz")}
                                >
                                    lorenz
                                </button>
                                <button
                                    class="seg"
                                    class:active={currentLayout === "grid"}
                                    onclick={() => setLayout("grid")}
                                >
                                    grid
                                </button>
                            </div>
                        </div>
                    {/if}

                    {#if capabilities.dynamicCount}
                        <div class="section count-input">
                            <div class="section-label">objects</div>
                            <input
                                type="number"
                                min="1"
                                max={maxCapacity}
                                step="100"
                                value={objectCount}
                                onchange={onCountChange}
                            />
                        </div>
                    {/if}

                    {#if capabilities.renderTestShapes}
                        <div class="section">
                            <div class="section-label">shape</div>
                            <div class="segmented-row">
                                {#each RENDER_TEST_SHAPES as shape}
                                    <button
                                        class="seg"
                                        class:active={currentRenderShape === shape}
                                        onclick={() => selectRenderShape(shape)}
                                    >
                                        {shape}
                                    </button>
                                {/each}
                            </div>
                        </div>
                    {/if}

                    {#if capabilities.renderTestVariants}
                        <div class="section">
                            <div class="section-label">variant</div>
                            <div class="segmented">
                                {#each RENDER_TEST_VARIANTS as variant}
                                    <button
                                        class="seg"
                                        class:active={currentRenderVariant === variant}
                                        onclick={() => selectRenderVariant(variant)}
                                    >
                                        {variant}
                                    </button>
                                {/each}
                            </div>
                        </div>
                    {/if}

                    {#if capabilities.pointLights}
                        <div class="section">
                            <div class="section-label">lighting</div>
                            <div class="segmented-row">
                                {#each RENDER_TEST_LIGHTING as mode}
                                    <button
                                        class="seg"
                                        class:active={currentLightingMode === mode.value}
                                        onclick={() => selectLightingMode(mode.value)}
                                    >
                                        {mode.label}
                                    </button>
                                {/each}
                            </div>
                        </div>
                    {/if}

                    {#if capabilities.renderTestText || capabilities.renderTestArrow}
                        <div class="section">
                            <div class="section-label">overlays</div>
                            <div class="segmented-row">
                                {#if capabilities.renderTestText}
                                    <button
                                        class="seg"
                                        class:active={textEnabled}
                                        onclick={() => toggleText()}
                                    >
                                        text
                                    </button>
                                {/if}
                                {#if capabilities.renderTestArrow}
                                    <button
                                        class="seg"
                                        class:active={arrowEnabled}
                                        onclick={() => toggleArrow()}
                                    >
                                        arrow
                                    </button>
                                {/if}
                            </div>
                        </div>
                    {/if}

                    <div class="section">
                        <button class="section-label collapse-btn" onclick={() => (effectsOpen = !effectsOpen)}>
                            effects {effectsOpen ? "▾" : "▸"}
                        </button>
                        {#if effectsOpen}
                            <div class="chips">
                                <div class="chip-group-label">camera</div>
                                <div class="chip-row">
                                    {#each Object.keys(CAMERA_EFFECTS) as name}
                                        <button
                                            class="chip"
                                            class:active={activeEffects.has(name)}
                                            onclick={() => toggleEffect(name)}
                                        >
                                            {name}
                                        </button>
                                    {/each}
                                </div>
                                <div class="chip-group-label">environment</div>
                                <div class="chip-row">
                                    {#each Object.keys(ENV_EFFECTS) as name}
                                        <button
                                            class="chip"
                                            class:active={activeEffects.has(name)}
                                            onclick={() => toggleEffect(name)}
                                        >
                                            {name}
                                        </button>
                                    {/each}
                                </div>
                            </div>
                        {/if}
                    </div>

                </div>
            </div>
        </Pane>
    </PaneGroup>
</div>

<style>
    .gym-root {
        width: 100%;
        height: 100%;
    }

    .canvas-pane {
        width: 100%;
        height: 100%;
        position: relative;
        background: #0c0a09;
    }
    .canvas-pane canvas {
        display: block;
        width: 100%;
        height: 100%;
    }

    .sidebar {
        height: 100%;
        display: flex;
        flex-direction: column;
        background: rgba(12, 10, 9, 0.97);
        color: #f0ece8;
        font-family: "JetBrains Mono", ui-monospace, monospace;
        font-size: 11px;
        border-left: 1px solid rgba(255, 255, 255, 0.06);
        overflow: hidden;
    }

    .sidebar-header {
        display: flex;
        align-items: center;
        gap: 8px;
        padding: 10px 14px 8px;
        border-bottom: 1px solid rgba(255, 255, 255, 0.04);
        flex-shrink: 0;
    }
    .scenario-label {
        font-family: "Outfit", system-ui, sans-serif;
        font-weight: 600;
        font-size: 10px;
        letter-spacing: 0.08em;
        text-transform: uppercase;
        color: #d49560;
    }
    .pipeline-badge {
        margin-left: auto;
        font-size: 9px;
        color: #706860;
        background: rgba(255, 255, 255, 0.04);
        padding: 1px 5px;
        border-radius: 3px;
        letter-spacing: 0.04em;
    }

    .sidebar-body {
        flex: 1;
        overflow-y: auto;
        scrollbar-gutter: stable;
    }
    .sidebar-body::-webkit-scrollbar {
        width: 4px;
    }
    .sidebar-body::-webkit-scrollbar-track {
        background: transparent;
    }
    .sidebar-body::-webkit-scrollbar-thumb {
        background: rgba(255, 255, 255, 0.06);
        border-radius: 2px;
    }

    .section {
        padding: 6px 14px;
        border-bottom: 1px solid rgba(255, 255, 255, 0.04);
    }

    .section-label {
        font-family: "Outfit", system-ui, sans-serif;
        font-weight: 600;
        font-size: 9px;
        letter-spacing: 0.1em;
        text-transform: uppercase;
        color: #504840;
        margin-bottom: 2px;
    }

    .scenario-select {
        width: 100%;
        background: rgba(255, 255, 255, 0.04);
        color: #cdc5bc;
        border: 1px solid rgba(255, 255, 255, 0.09);
        border-radius: 3px;
        padding: 5px 6px;
        font-size: 11px;
        font-family: "JetBrains Mono", ui-monospace, monospace;
        cursor: pointer;
        transition: border-color 150ms cubic-bezier(0.34, 0, 0, 1);
        appearance: none;
        margin-top: 4px;
    }
    .scenario-select:hover {
        border-color: rgba(255, 255, 255, 0.12);
    }
    .scenario-select:focus {
        outline: none;
        border-color: #d49560;
        box-shadow: 0 0 0 1px rgba(212, 149, 96, 0.15);
    }
    .scenario-select option {
        background: #161412;
        color: #cdc5bc;
    }

    .segmented {
        display: flex;
        flex-direction: column;
        gap: 1px;
        background: rgba(255, 255, 255, 0.03);
        border-radius: 4px;
        overflow: hidden;
        margin: 4px 0 2px;
    }
    .seg {
        padding: 5px 0;
        background: rgba(255, 255, 255, 0.04);
        color: #706860;
        font-family: "JetBrains Mono", ui-monospace, monospace;
        font-size: 10px;
        text-align: center;
        border: none;
        cursor: pointer;
        transition:
            background 100ms cubic-bezier(0.34, 0, 0, 1),
            color 100ms cubic-bezier(0.34, 0, 0, 1);
        letter-spacing: 0.02em;
    }
    .seg:hover {
        background: rgba(255, 255, 255, 0.08);
        color: #a09888;
    }
    .seg:active {
        background: rgba(212, 149, 96, 0.08);
        transform: scale(0.95);
    }
    .seg.active {
        background: #d49560;
        color: #0c0a09;
        font-weight: 600;
    }

    .segmented-row {
        display: flex;
        gap: 1px;
        background: rgba(255, 255, 255, 0.03);
        border-radius: 4px;
        overflow: hidden;
        margin: 4px 0 2px;
    }
    .segmented-row .seg {
        flex: 1;
    }

    .count-input input {
        background: rgba(255, 255, 255, 0.04);
        color: #cdc5bc;
        border: 1px solid rgba(255, 255, 255, 0.09);
        border-radius: 3px;
        padding: 4px 6px;
        font-size: 11px;
        font-family: "JetBrains Mono", ui-monospace, monospace;
        transition: border-color 120ms cubic-bezier(0.34, 0, 0, 1);
        width: 100%;
        text-align: center;
    }
    .count-input input:hover {
        border-color: rgba(255, 255, 255, 0.12);
    }
    .count-input input:focus {
        outline: none;
        border-color: #d49560;
        box-shadow: 0 0 0 1px rgba(212, 149, 96, 0.15);
    }
    .count-input input::-webkit-inner-spin-button,
    .count-input input::-webkit-outer-spin-button {
        -webkit-appearance: none;
        appearance: none;
        margin: 0;
    }
    .count-input input[type="number"] {
        -moz-appearance: textfield;
        appearance: textfield;
    }

    .collapse-btn {
        background: none;
        border: none;
        cursor: pointer;
        padding: 0;
        font-family: "Outfit", system-ui, sans-serif;
        font-weight: 600;
        font-size: 9px;
        letter-spacing: 0.1em;
        text-transform: uppercase;
        color: #504840;
        margin-bottom: 2px;
        width: 100%;
        text-align: left;
    }
    .collapse-btn:hover {
        color: #706860;
    }

    .chips {
        margin-top: 4px;
    }
    .chip-group-label {
        font-size: 8px;
        color: #504840;
        text-transform: uppercase;
        letter-spacing: 0.08em;
        margin: 6px 0 2px;
    }
    .chip-group-label:first-child {
        margin-top: 0;
    }
    .chip-row {
        display: flex;
        flex-wrap: wrap;
        gap: 3px;
    }
    .chip {
        padding: 2px 6px;
        background: rgba(255, 255, 255, 0.04);
        color: #706860;
        border: 1px solid rgba(255, 255, 255, 0.06);
        border-radius: 3px;
        font-family: "JetBrains Mono", ui-monospace, monospace;
        font-size: 9px;
        cursor: pointer;
        transition:
            background 100ms cubic-bezier(0.34, 0, 0, 1),
            color 100ms cubic-bezier(0.34, 0, 0, 1),
            border-color 100ms cubic-bezier(0.34, 0, 0, 1);
    }
    .chip:hover {
        background: rgba(255, 255, 255, 0.08);
        color: #a09888;
    }
    .chip:active {
        background: rgba(212, 149, 96, 0.08);
        transform: scale(0.95);
    }
    .chip.active {
        background: rgba(212, 149, 96, 0.2);
        color: #d49560;
        border-color: rgba(212, 149, 96, 0.3);
    }

    .gym-root :global([data-pane-group]) {
        display: flex;
        height: 100%;
    }
    .gym-root :global([data-pane]) {
        overflow: hidden;
    }
    :global(.resizer) {
        flex-shrink: 0;
        position: relative;
        background: rgba(255, 255, 255, 0.09);
        z-index: 2;
        transition: background 150ms cubic-bezier(0.34, 0, 0, 1);
    }
    :global(.resizer::before) {
        content: "";
        position: absolute;
    }
    :global(.resizer:hover),
    :global(.resizer[data-resize-handle-active]) {
        background: #d49560;
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
</style>
