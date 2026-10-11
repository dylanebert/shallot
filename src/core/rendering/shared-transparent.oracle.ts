import { setupGlobals } from "@dylanebert/shallot/webgpu";
import { createApp } from "../../engine";
import {
    AlphaMode,
    engineLayout,
    Materials,
    MeshMaterial,
    MeshRenderPlugin,
    StandardMaterial,
    StandardRenderer,
    StandardRenderingPlugin,
} from "../../standard/rendering";
import { cube, Meshes, MeshInstance, registerMesh } from "../mesh";
import { Transform } from "../transform";
import { attachTexture, Camera, TransparentPhases } from ".";

const MAX_ITEMS = 8192;
const FRAME_BUDGET_MS = 16.7;
const ORDERING_BUDGET_FRACTION = 0.1;
const WARM_FRAMES = 2;
const MEASURED_FRAMES = 7;

function median(values: number[]): number {
    const sorted = [...values].sort((a, b) => a - b);
    return sorted[Math.floor(sorted.length / 2)];
}

function storageCount(layout: unknown, stage: "vertex" | "fragment"): number {
    const entries = (
        layout as {
            entries: Record<string, { storage?: unknown; visibility?: readonly string[] }>;
        }
    ).entries;
    return Object.values(entries).filter(
        (entry) => entry.storage && (entry.visibility ?? ["vertex", "fragment"]).includes(stage),
    ).length;
}

type ViewMetrics = {
    items: number;
    draws: number;
    pipelineChanges: number;
    bindGroupChanges: number;
    uploadBytes: number;
    sortComparisons: number;
    keyMs: number;
    sortMs: number;
    prepareMs: number;
    drawMs: number;
    uploadMs: number;
};

function metrics(world: Awaited<ReturnType<typeof createApp>>["world"], cameras: number[]) {
    const perView: ViewMetrics[] = [];
    let keyMs = 0;
    let sortMs = 0;
    let prepareMs = 0;
    let drawMs = 0;
    let uploadMs = 0;
    let uploadBytes = 0;
    let items = 0;
    let sortComparisons = 0;
    let draws = 0;
    let pipelineChanges = 0;
    let bindGroupChanges = 0;
    const phases = world.resource(TransparentPhases);
    for (const eid of cameras) {
        const phase = phases.get(eid)!;
        const values: ViewMetrics = {
            items: phase.metrics.items,
            draws: phase.metrics.draws,
            pipelineChanges: phase.metrics.pipelineChanges,
            bindGroupChanges: phase.metrics.bindGroupChanges,
            uploadBytes: phase.metrics.uploadBytes,
            sortComparisons: phase.metrics.sortComparisons,
            keyMs: phase.metrics.keyMs,
            sortMs: phase.metrics.sortMs,
            prepareMs: phase.metrics.prepareMs,
            drawMs: phase.metrics.drawMs,
            uploadMs: phase.metrics.uploadMs,
        };
        perView.push(values);
        items += values.items;
        draws += values.draws;
        pipelineChanges += values.pipelineChanges;
        bindGroupChanges += values.bindGroupChanges;
        uploadBytes += values.uploadBytes;
        sortComparisons += values.sortComparisons;
        keyMs += values.keyMs;
        sortMs += values.sortMs;
        prepareMs += values.prepareMs;
        drawMs += values.drawMs;
        uploadMs += values.uploadMs;
    }
    return {
        perView,
        totals: {
            items,
            draws,
            pipelineChanges,
            bindGroupChanges,
            uploadBytes,
            sortComparisons,
            keyMs,
            sortMs,
            prepareMs,
            drawMs,
            uploadMs,
            orderingMs: keyMs + sortMs + prepareMs + drawMs,
        },
    };
}

type Workload = "singleMaterial" | "alternatingMeshes" | "alternatingMaterials";

async function measure(device: GPUDevice, viewCount: number, workload: Workload) {
    const app = await createApp({
        defaults: false,
        device,
        plugins: [StandardRenderingPlugin, MeshRenderPlugin],
    });
    const world = app.world;
    const cameras: number[] = [];
    for (let i = 0; i < viewCount; i++) {
        const camera = world.create();
        world.add(camera, Transform, { translation: [0, 0, 5, 0] });
        world.add(camera, Camera);
        world.storage(Camera).antialias.set(camera, 0);
        world.add(camera, StandardRenderer);
        attachTexture(world, camera, { width: 1, height: 1 });
        cameras.push(camera);
    }

    const materials = world.resource(Materials);
    const first = materials.add(StandardMaterial({ baseColor: [1, 0, 0, 0.25], unlit: true }), {
        alphaMode: AlphaMode.Blend,
    });
    const second = materials.add(StandardMaterial({ baseColor: [0, 0, 1, 0.25], unlit: true }), {
        alphaMode: AlphaMode.Multiply,
    });
    const meshes = world.resource(Meshes);
    const defaultMesh = [...meshes.entries()].find(([, mesh]) => mesh.name === "cube")![0];
    const alternateMesh =
        workload === "alternatingMeshes"
            ? registerMesh(world, { name: "transparent-oracle-alternate-cube", ...cube() })
            : defaultMesh;
    const entities = new Uint32Array(MAX_ITEMS);
    for (let i = 0; i < MAX_ITEMS; i++) {
        const eid = world.create();
        entities[i] = eid;
        world.add(eid, Transform, {
            translation: [0, 0, -1 - ((i * 8191) % MAX_ITEMS) / MAX_ITEMS, 0],
            scale: [0.001, 0.001, 0.001, 0],
        });
        world.add(eid, MeshInstance);
    }

    world.step(0);
    world.step(0);
    const scenarios: Record<string, unknown>[] = [];
    for (const itemCount of [256, 1024, MAX_ITEMS]) {
        for (let i = 0; i < MAX_ITEMS; i++) {
            const eid = entities[i]!;
            const active = i < itemCount;
            if (active) {
                if (!world.has(eid, MeshInstance)) world.add(eid, MeshInstance);
                if (!world.has(eid, MeshMaterial))
                    world.add(
                        eid,
                        MeshMaterial,
                        workload === "alternatingMaterials" && i & 1 ? second : first,
                    );
                if (workload === "alternatingMeshes")
                    world.storage(MeshInstance).mesh.set(eid, i & 1 ? alternateMesh : defaultMesh);
            } else if (world.has(eid, MeshInstance)) {
                world.remove(eid, MeshInstance);
            }
        }
        for (let frame = 0; frame < WARM_FRAMES; frame++) world.step(0);

        let standardMeshInstanceRows = 0;
        for (const _eid of world.query([MeshInstance])) standardMeshInstanceRows++;
        if (standardMeshInstanceRows !== itemCount)
            throw new Error(
                `standard MeshInstance storage has ${standardMeshInstanceRows} rows for ${itemCount} active items`,
            );

        const samples: ViewMetrics[][] = [];
        const wholeFrameMs: number[] = [];
        for (let frame = 0; frame < MEASURED_FRAMES; frame++) {
            const started = performance.now();
            world.step(0);
            wholeFrameMs.push(performance.now() - started);
            samples.push(metrics(world, cameras).perView);
        }
        await device.queue.onSubmittedWorkDone();

        const perView: Record<string, unknown>[] = [];
        const visibleStandardItemsByView: number[] = [];
        for (let slot = 0; slot < viewCount; slot++) {
            const viewSamples = samples.map((sample) => sample[slot]!);
            const visibleItems = median(viewSamples.map((sample) => sample.items));
            if (visibleItems !== standardMeshInstanceRows)
                throw new Error(
                    `view ${slot} sees ${visibleItems} of ${standardMeshInstanceRows} standard MeshInstance rows`,
                );
            visibleStandardItemsByView.push(visibleItems);
            perView.push({
                slot,
                standardMeshInstanceRows,
                visibleItems,
                draws: median(viewSamples.map((sample) => sample.draws)),
                pipelineChanges: median(viewSamples.map((sample) => sample.pipelineChanges)),
                bindGroupChanges: median(viewSamples.map((sample) => sample.bindGroupChanges)),
                uploadBytes: median(viewSamples.map((sample) => sample.uploadBytes)),
                sortComparisons: median(viewSamples.map((sample) => sample.sortComparisons)),
                keyCollectionMs: median(viewSamples.map((sample) => sample.keyMs)),
                sortMs: median(viewSamples.map((sample) => sample.sortMs)),
                prepareMs: median(viewSamples.map((sample) => sample.prepareMs)),
                drawMs: median(viewSamples.map((sample) => sample.drawMs)),
                orderedUploadMs: median(viewSamples.map((sample) => sample.uploadMs)),
            });
        }
        const totalSamples = samples.map((sample) => {
            let items = 0;
            let draws = 0;
            let pipelineChanges = 0;
            let bindGroupChanges = 0;
            let uploadBytes = 0;
            let sortComparisons = 0;
            let keyMs = 0;
            let sortMs = 0;
            let prepareMs = 0;
            let drawMs = 0;
            let uploadMs = 0;
            for (const view of sample) {
                items += view.items;
                draws += view.draws;
                pipelineChanges += view.pipelineChanges;
                bindGroupChanges += view.bindGroupChanges;
                uploadBytes += view.uploadBytes;
                sortComparisons += view.sortComparisons;
                keyMs += view.keyMs;
                sortMs += view.sortMs;
                prepareMs += view.prepareMs;
                drawMs += view.drawMs;
                uploadMs += view.uploadMs;
            }
            return {
                items,
                draws,
                pipelineChanges,
                bindGroupChanges,
                uploadBytes,
                sortComparisons,
                keyMs,
                sortMs,
                prepareMs,
                drawMs,
                uploadMs,
                orderingMs: keyMs + sortMs + prepareMs + drawMs,
            };
        });
        const orderingMs = median(totalSamples.map((sample) => sample.orderingMs));
        scenarios.push({
            workload,
            itemsPerView: itemCount,
            views: viewCount,
            storageByVisibility: {
                worldEntities: MAX_ITEMS,
                standardMeshInstanceRows,
                entitiesOutsideStandardMeshInstanceLayout: MAX_ITEMS - standardMeshInstanceRows,
                visibleStandardItemsByView,
            },
            perFrame: {
                standardMeshInstanceRows,
                visibleItems: median(totalSamples.map((sample) => sample.items)),
                draws: median(totalSamples.map((sample) => sample.draws)),
                pipelineChanges: median(totalSamples.map((sample) => sample.pipelineChanges)),
                bindGroupChanges: median(totalSamples.map((sample) => sample.bindGroupChanges)),
                uploadBytes: median(totalSamples.map((sample) => sample.uploadBytes)),
                sortComparisons: median(totalSamples.map((sample) => sample.sortComparisons)),
                keyCollectionMs: median(totalSamples.map((sample) => sample.keyMs)),
                sortMs: median(totalSamples.map((sample) => sample.sortMs)),
                prepareMs: median(totalSamples.map((sample) => sample.prepareMs)),
                drawMs: median(totalSamples.map((sample) => sample.drawMs)),
                orderedUploadMs: median(totalSamples.map((sample) => sample.uploadMs)),
                orderingMs,
                reservedBudgetPercent:
                    (orderingMs / (FRAME_BUDGET_MS * ORDERING_BUDGET_FRACTION)) * 100,
                wholeFrameCpuMs: median(wholeFrameMs),
            },
            perView,
        });
    }
    app.dispose();
    return scenarios;
}

async function main(): Promise<void> {
    await setupGlobals();
    const adapter = await navigator.gpu.requestAdapter();
    if (!adapter) throw new Error("Transparent rendering oracle requires a WebGPU adapter");
    const adapterInfo = adapter.info;
    const device = await adapter.requestDevice({
        requiredFeatures: ["indirect-first-instance", "rg11b10ufloat-renderable"],
        requiredLimits: { maxStorageBuffersPerShaderStage: 10 },
    });
    const workloads: Record<string, unknown>[] = [];
    for (const workload of ["singleMaterial", "alternatingMeshes", "alternatingMaterials"] as const)
        for (const views of [1, 2, 8]) workloads.push(...(await measure(device, views, workload)));
    const perWorkload = (workload: Workload, itemsPerView: number, views: number) =>
        workloads.find(
            (scene) =>
                scene.workload === workload &&
                scene.itemsPerView === itemsPerView &&
                scene.views === views,
        );
    const batchingReferences = {
        singleMaterialOneMesh: {
            items: MAX_ITEMS,
            draws: 1,
            pipelineChanges: 1,
            bindGroupChanges: 3,
            note: "All items share one material row, alpha mode and mesh; one consecutive run per view.",
        },
        alternatingMeshes: {
            items: MAX_ITEMS,
            draws: 2,
            pipelineChanges: 1,
            bindGroupChanges: 5,
            note: "Hypothetical grouping by mesh; not correctness-equivalent for overlapping items. The sorted alternating workload itself cannot merge non-consecutive mesh runs.",
        },
        alternatingMaterials: {
            items: MAX_ITEMS,
            draws: 2,
            pipelineChanges: 2,
            bindGroupChanges: 5,
            note: "Hypothetical grouping by material; not correctness-equivalent for overlapping items.",
        },
    };
    const largest = perWorkload("alternatingMaterials", MAX_ITEMS, 8)!;
    const largestMs = (largest.perFrame as { orderingMs: number }).orderingMs;
    console.log(
        JSON.stringify(
            {
                adapter: {
                    vendor: adapterInfo.vendor,
                    device: adapterInfo.device,
                    description: adapterInfo.description,
                    architecture: adapterInfo.architecture,
                },
                runtime: { bun: Bun.version, node: process.versions.node },
                methodology: {
                    scene: "StandardRenderingPlugin plus MeshRenderPlugin; one default cube per visible item, centered at distinct depths, with independent 1x1 camera targets. singleMaterial uses one StandardMaterial row in Blend mode on one mesh; alternatingMeshes alternates two mesh handles with that same material; alternatingMaterials alternates Blend and Multiply rows on one mesh.",
                    warmFrames: WARM_FRAMES,
                    measuredFrames: MEASURED_FRAMES,
                    statistic:
                        "Median of per-frame samples; phase timings come from the production transparent callbacks. orderedUploadMs is included in prepareMs, not added twice. wholeFrameCpuMs times World.step and excludes asynchronous GPU execution.",
                },
                frameBudgetMs: FRAME_BUDGET_MS,
                reservedForTransparentOrderingFraction: ORDERING_BUDGET_FRACTION,
                reservedForTransparentOrderingMs: FRAME_BUDGET_MS * ORDERING_BUDGET_FRACTION,
                standardLayoutStorageCountsByVisibility: {
                    vertex:
                        storageCount(engineLayout, "vertex") +
                        storageCount(Materials.layout, "vertex"),
                    fragment:
                        storageCount(engineLayout, "fragment") +
                        storageCount(Materials.layout, "fragment"),
                },
                budgetRationale:
                    "Reserve only 10% of a 60 Hz frame for shared transparent collection, sorting, staging and pass recording, leaving 90% for simulation, other rendering, game work and host overhead.",
                singleSubmission: true,
                workloads,
                batchingReferences,
                largestBudgetOverrunMs: Math.max(
                    0,
                    largestMs - FRAME_BUDGET_MS * ORDERING_BUDGET_FRACTION,
                ),
            },
            null,
            2,
        ),
    );
    device.destroy();
}

await main();
