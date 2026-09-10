// The registry every stage-4 sample twin plugs into: one entry per ported
// upstream sample — its committed gold, its build()/update() through the escape hatch, and 2-3 non-default
// knob points for the boundedness probe (the gold trajectory exists at defaults only). `physics-golds.test.ts`
// iterates this list; a new sample twin adds ONE entry here — see that file's header for the full recipe.

import goldJsonBodyType from "../../../src/standard/physics/samples/bodies-body-type.json";
import goldJsonMotionLocks from "../../../src/standard/physics/samples/bodies-motion-locks.json";
import goldJsonSpinningBook from "../../../src/standard/physics/samples/bodies-spinning-book.json";
import goldJsonCharacterMover from "../../../src/standard/physics/samples/character-mover.json";
import goldJsonOverlapBox from "../../../src/standard/physics/samples/collision-overlap-box.json";
import goldJsonRayCurtain from "../../../src/standard/physics/samples/collision-ray-curtain.json";
import goldJsonShapeCast from "../../../src/standard/physics/samples/collision-shape-cast.json";
import goldJsonCompoundSimple from "../../../src/standard/physics/samples/compound-simple.json";
import goldJsonCompoundSpheres from "../../../src/standard/physics/samples/compound-spheres.json";
import goldJsonTileFloor from "../../../src/standard/physics/samples/compound-tile-floor.json";
import goldJsonBulletVsStack from "../../../src/standard/physics/samples/continuous-bullet-vs-stack.json";
import goldJsonThinWall from "../../../src/standard/physics/samples/continuous-thin-wall.json";
import goldJsonFallingRagdolls from "../../../src/standard/physics/samples/determinism-falling-ragdolls.json";
import goldJsonHitEvents from "../../../src/standard/physics/samples/events-hit.json";
import goldJsonJointBreak from "../../../src/standard/physics/samples/events-joint-break.json";
import goldJsonSensorSweep from "../../../src/standard/physics/samples/events-sensor-sweep.json";
import goldJsonConvexHull from "../../../src/standard/physics/samples/geometry-convex-hull.json";
import goldJsonConvexPrimitives from "../../../src/standard/physics/samples/geometry-convex-primitives.json";
import goldJsonHullReduction from "../../../src/standard/physics/samples/geometry-hull-reduction.json";
import goldJsonBridge from "../../../src/standard/physics/samples/joints-bridge.json";
import goldJsonCantilever from "../../../src/standard/physics/samples/joints-cantilever.json";
import goldJsonDriving from "../../../src/standard/physics/samples/joints-driving.json";
import goldJsonElevator from "../../../src/standard/physics/samples/joints-elevator.json";
import goldJsonFilter from "../../../src/standard/physics/samples/joints-filter.json";
import goldJsonPaddle from "../../../src/standard/physics/samples/joints-paddle.json";
import goldJsonParallel from "../../../src/standard/physics/samples/joints-parallel.json";
import goldJsonPendulum from "../../../src/standard/physics/samples/joints-pendulum.json";
import goldJsonRope from "../../../src/standard/physics/samples/joints-rope.json";
import goldJsonSuspension from "../../../src/standard/physics/samples/joints-suspension.json";
import goldJsonTerrain from "../../../src/standard/physics/samples/mesh-terrain.json";
import goldJsonTorus from "../../../src/standard/physics/samples/mesh-torus.json";
import goldJsonRagdoll from "../../../src/standard/physics/samples/ragdoll-ragdoll.json";
import goldJsonInclinedPlane from "../../../src/standard/physics/samples/shapes-inclined-plane.json";
import goldJsonRestitution from "../../../src/standard/physics/samples/shapes-restitution.json";
import goldJsonShapeSoup from "../../../src/standard/physics/samples/shapes-shape-soup.json";
import goldJsonArch from "../../../src/standard/physics/samples/stacking-arch.json";
import goldJsonBoxPyramid from "../../../src/standard/physics/samples/stacking-box-pyramid.json";
import goldJsonDominoes from "../../../src/standard/physics/samples/stacking-dominoes.json";
import { buildArch } from "./physics-arch";
import { buildBodyType, updateBodyType } from "./physics-body-type";
import { buildBoxPyramid } from "./physics-box-pyramid";
import { buildBridge } from "./physics-bridge";
import { buildBulletVsStack } from "./physics-bullet-vs-stack";
import { buildCantilever } from "./physics-cantilever";
import { buildCharacterMover, updateCharacterMover } from "./physics-character-mover";
import { buildCompoundSimple } from "./physics-compound-simple";
import { buildCompoundSpheres } from "./physics-compound-spheres";
import { buildTileFloor } from "./physics-compound-tile-floor";
import { buildConvexHull } from "./physics-convex-hull";
import { buildConvexPrimitives } from "./physics-convex-primitives";
import { buildDominoes } from "./physics-dominoes";
import { buildDriving } from "./physics-driving";
import { buildElevator, updateElevator } from "./physics-elevator";
import { buildFallingRagdolls } from "./physics-falling-ragdolls";
import { buildFilter } from "./physics-filter";
import { buildHitEvents } from "./physics-hit";
import { buildHullReduction } from "./physics-hull-reduction";
import { buildInclinedPlane } from "./physics-inclined-plane";
import { buildJointBreak, updateJointBreak } from "./physics-joint-break";
import { buildMotionLocks, updateMotionLocks } from "./physics-motion-locks";
import type { SampleBuild, SampleGold, SampleParams, SampleUpdate } from "./physics-oracle";
import { buildOverlapBox } from "./physics-overlap-box";
import { buildPaddle } from "./physics-paddle";
import { buildParallel, updateParallel } from "./physics-parallel";
import { buildPendulum } from "./physics-pendulum";
import { buildRagdoll } from "./physics-ragdoll";
import { buildRayCurtain } from "./physics-ray-curtain";
import { buildRestitution } from "./physics-restitution";
import { buildRope } from "./physics-rope";
import { buildSensorSweep, updateSensorSweep } from "./physics-sensor-sweep";
import { buildShapeCast } from "./physics-shape-cast";
import { buildShapeSoup } from "./physics-shape-soup";
import { buildSpinningBook } from "./physics-spinning-book";
import { buildSuspension } from "./physics-suspension";
import { buildTerrain } from "./physics-terrain";
import { buildThinWall } from "./physics-thin-wall";
import { buildTorus } from "./physics-torus";

/** one registry entry: a ported sample's committed gold + its build/update, plus 2-3 non-default knob
 *  points the boundedness probe steps through (never gold-checked — only asserted finite). Omit
 *  `knobPoints` for a sample with no knobs. */
export interface GoldEntry {
    slug: string;
    gold: SampleGold;
    build: SampleBuild;
    update?: SampleUpdate;
    knobPoints?: SampleParams[];
}

export const goldRegistry: GoldEntry[] = [
    {
        slug: "joints-paddle",
        gold: goldJsonPaddle as unknown as SampleGold,
        build: buildPaddle,
        knobPoints: [{ speed: -8 }, { speed: 0 }, { speed: 8 }],
    },
    {
        slug: "bodies-body-type",
        gold: goldJsonBodyType as unknown as SampleGold,
        build: buildBodyType,
        update: updateBodyType,
        knobPoints: [{ type: "dynamic" }, { type: "static" }],
    },
    {
        slug: "stacking-arch",
        gold: goldJsonArch as unknown as SampleGold,
        build: buildArch,
    },
    {
        slug: "stacking-box-pyramid",
        gold: goldJsonBoxPyramid as unknown as SampleGold,
        build: buildBoxPyramid,
        knobPoints: [{ rows: 3 }, { rows: 6 }, { rows: 14 }],
    },
    {
        slug: "stacking-dominoes",
        gold: goldJsonDominoes as unknown as SampleGold,
        build: buildDominoes,
        knobPoints: [{ rings: 1 }, { rings: 2 }, { rings: 8 }],
    },
    {
        slug: "shapes-inclined-plane",
        gold: goldJsonInclinedPlane as unknown as SampleGold,
        build: buildInclinedPlane,
    },
    {
        slug: "shapes-restitution",
        gold: goldJsonRestitution as unknown as SampleGold,
        build: buildRestitution,
        knobPoints: [{ shape: "box" }, { count: 4 }, { count: 40 }],
    },
    {
        slug: "shapes-shape-soup",
        gold: goldJsonShapeSoup as unknown as SampleGold,
        build: buildShapeSoup,
        knobPoints: [{ rows: 2 }, { rows: 5 }, { rows: 8 }],
    },
    {
        slug: "bodies-motion-locks",
        gold: goldJsonMotionLocks as unknown as SampleGold,
        build: buildMotionLocks,
        update: updateMotionLocks,
    },
    {
        slug: "bodies-spinning-book",
        gold: goldJsonSpinningBook as unknown as SampleGold,
        build: buildSpinningBook,
    },
    {
        slug: "character-mover",
        gold: goldJsonCharacterMover as unknown as SampleGold,
        build: buildCharacterMover,
        update: updateCharacterMover,
    },
    {
        slug: "continuous-bullet-vs-stack",
        gold: goldJsonBulletVsStack as unknown as SampleGold,
        build: buildBulletVsStack,
        knobPoints: [{ speed: 40 }, { speed: 80 }, { speed: 200 }],
    },
    {
        slug: "continuous-thin-wall",
        gold: goldJsonThinWall as unknown as SampleGold,
        build: buildThinWall,
    },
    {
        slug: "collision-overlap-box",
        gold: goldJsonOverlapBox as unknown as SampleGold,
        build: buildOverlapBox,
        knobPoints: [{ size: 1 }, { size: 2.5 }, { size: 4 }],
    },
    {
        slug: "collision-ray-curtain",
        gold: goldJsonRayCurtain as unknown as SampleGold,
        build: buildRayCurtain,
    },
    {
        slug: "collision-shape-cast",
        gold: goldJsonShapeCast as unknown as SampleGold,
        build: buildShapeCast,
    },
    {
        slug: "compound-simple",
        gold: goldJsonCompoundSimple as unknown as SampleGold,
        build: buildCompoundSimple,
    },
    {
        slug: "compound-spheres",
        gold: goldJsonCompoundSpheres as unknown as SampleGold,
        build: buildCompoundSpheres,
    },
    {
        slug: "compound-tile-floor",
        gold: goldJsonTileFloor as unknown as SampleGold,
        build: buildTileFloor,
    },
    {
        slug: "events-hit",
        gold: goldJsonHitEvents as unknown as SampleGold,
        build: buildHitEvents,
    },
    {
        slug: "events-joint-break",
        gold: goldJsonJointBreak as unknown as SampleGold,
        build: buildJointBreak,
        update: updateJointBreak,
    },
    {
        slug: "events-sensor-sweep",
        gold: goldJsonSensorSweep as unknown as SampleGold,
        build: buildSensorSweep,
        update: updateSensorSweep,
    },
    {
        slug: "geometry-convex-hull",
        gold: goldJsonConvexHull as unknown as SampleGold,
        build: buildConvexHull,
        knobPoints: [{ count: 4 }, { count: 12 }, { count: 24 }],
    },
    {
        slug: "geometry-convex-primitives",
        gold: goldJsonConvexPrimitives as unknown as SampleGold,
        build: buildConvexPrimitives,
    },
    {
        slug: "geometry-hull-reduction",
        gold: goldJsonHullReduction as unknown as SampleGold,
        build: buildHullReduction,
        knobPoints: [
            { vertices: 4, count: 1 },
            { vertices: 8, count: 6 },
            { vertices: 20, count: 12 },
        ],
    },
    {
        slug: "joints-bridge",
        gold: goldJsonBridge as unknown as SampleGold,
        build: buildBridge,
        knobPoints: [{ planks: 10 }, { planks: 18 }, { planks: 40 }],
    },
    {
        slug: "joints-cantilever",
        gold: goldJsonCantilever as unknown as SampleGold,
        build: buildCantilever,
        knobPoints: [{ stiffness: 2 }, { stiffness: 30 }, { stiffness: 60 }],
    },
    {
        slug: "joints-driving",
        gold: goldJsonDriving as unknown as SampleGold,
        build: buildDriving,
        knobPoints: [{ throttle: -10 }, { throttle: 0 }, { throttle: 20 }],
    },
    {
        slug: "joints-elevator",
        gold: goldJsonElevator as unknown as SampleGold,
        build: buildElevator,
        update: updateElevator,
    },
    {
        slug: "joints-filter",
        gold: goldJsonFilter as unknown as SampleGold,
        build: buildFilter,
    },
    {
        slug: "joints-parallel",
        gold: goldJsonParallel as unknown as SampleGold,
        build: buildParallel,
        update: updateParallel,
    },
    {
        slug: "joints-pendulum",
        gold: goldJsonPendulum as unknown as SampleGold,
        build: buildPendulum,
        knobPoints: [{ links: 2 }, { links: 6 }, { links: 14 }],
    },
    {
        slug: "joints-rope",
        gold: goldJsonRope as unknown as SampleGold,
        build: buildRope,
        knobPoints: [{ links: 4 }, { links: 12 }, { links: 20 }],
    },
    {
        slug: "joints-suspension",
        gold: goldJsonSuspension as unknown as SampleGold,
        build: buildSuspension,
        knobPoints: [{ stiffness: 1 }, { stiffness: 6 }, { stiffness: 12 }],
    },
    {
        slug: "mesh-terrain",
        gold: goldJsonTerrain as unknown as SampleGold,
        build: buildTerrain,
        knobPoints: [{ shape: "sphere" }, { shape: "box" }],
    },
    {
        slug: "mesh-torus",
        gold: goldJsonTorus as unknown as SampleGold,
        build: buildTorus,
    },
    {
        slug: "ragdoll-ragdoll",
        gold: goldJsonRagdoll as unknown as SampleGold,
        build: buildRagdoll,
    },
    {
        slug: "determinism-falling-ragdolls",
        gold: goldJsonFallingRagdolls as unknown as SampleGold,
        build: buildFallingRagdolls,
        knobPoints: [{ grid: 1 }, { grid: 2 }, { grid: 4 }],
    },
];
