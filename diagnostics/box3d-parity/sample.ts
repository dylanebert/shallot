import { readFileSync, writeFileSync } from "node:fs";
import { tmpdir } from "node:os";
import { join } from "node:path";
import { nativeBinary } from "./native";

/** Compile the selected upstream classes verbatim, without the samples app's graphics dependencies. */
export function sampleBinary(samples = join(process.env.BOX3D ?? "", "samples")): string {
    const shapes = readFileSync(join(samples, "sample_shapes.cpp"), "utf8");
    const bodies = readFileSync(join(samples, "sample_bodies.cpp"), "utf8");
    const stacking = readFileSync(join(samples, "sample_stacking.cpp"), "utf8");
    const joints = readFileSync(join(samples, "sample_joint.cpp"), "utf8");
    const sample = readFileSync(join(samples, "sample.cpp"), "utf8");
    const events = readFileSync(join(samples, "sample_events.cpp"), "utf8");
    const benchmark = readFileSync(join(samples, "sample_benchmark.cpp"), "utf8");
    function section(source: string, start: string, end: string): string {
        const first = source.indexOf(start);
        const last = source.indexOf(end, first);
        if (first < 0 || last < 0) throw new Error(`missing sample source boundary: ${start} / ${end}`);
        return source.slice(first, last);
    }
    const sources = [
        section(sample, "b3BodyId Sample::AddGroundBox(", "struct RowDef"),
        ...["InclinedPlane", "RollingResistance", "HighResistance", "IsotropicFriction", "SlideTwist", "Restitution", "StaticInvoke", "Wind", "ConveyorBelt", "ConveyorMesh", "WindDrop", "WindFlap"].map(name => section(shapes, `class ${name} :`, `static int sample${name}`)),
        ...[["GyroscopicPrecession", "sampleGyroscopicPrecession"], ["BodyType", "sampleBodyType"], ["Weeble", "sampleWeeble"], ["DisableBody", "sampleDisable"], ["SpinningBooks", "sampleBook"], ["GyroscopicTorque", "sampleGyroscopicTorque"], ["Kinematic", "sampleKinematic"], ["LockMixing", "sampleLockMixing"], ["FixedRotation", "sampleFixedRotation"], ["OffsetKinematic", "sampleOffsetKinematic"]].map(([name, end]) => section(bodies, `class ${name} :`, `static int ${end}`)),
        section(benchmark.slice(benchmark.lastIndexOf("class BenchmarkLargeWorld :")), "class BenchmarkLargeWorld :", "static int sampleLargeWorld"),
        ...["BenchmarkLargePyramid", "BenchmarkWidePyramid", "BenchmarkManyPyramids", "BenchmarkJointGrid", "FallingBoxes", "BenchmarkRain", "BenchmarkJunkyard", "BenchmarkWasher", "BenchmarkConvexPile", "BenchmarkFallingTrees", "BenchmarkExplosion", "BenchmarkDestruction", "BenchmarkChains", "BenchmarkHeightField", "BenchmarkHull"].map(name =>
            section(benchmark, `class ${name} :`, name === "BenchmarkHull" ? "static int sampleBenchmarkHull" : name === "BenchmarkHeightField" ? "static int sampleHeightFieldBenchmark" : name === "BenchmarkChains" ? "static int sampleBenchmarkChains" : name === "BenchmarkWasher" ? "static int benchmarkWasher" : `static int sample${name.replace("Benchmark", "")}`)),
        section(benchmark, "inline bool operator<", "class BenchmarkLargePyramid"),
        section(benchmark, "struct ShapeUserData", "static int benchmarkSensor"),
        section(benchmark, "class CandyCups :", "static int sampleSmallConvexes"),
        ...["BoxStack", "CardHouse", "SphereStack", "CapsuleStack", "JengaStack", "SingleBox", "Cylinder", "CylinderStack", "Dominoes", "Wedge", "Arch", "DoubleDomino", "Pyramid2D", "EdgeCrossing"].map(name => section(stacking, `class ${name} :`, `static int sample${name}`)),
        ...["DistanceJoint", "SphericalJoint", "MotionLocks", "BallAndChain", "FoldOut", "TopDownFriction", "WeldJoint", "GearLift", "Driving"].map(name => section(joints, `class ${name} :`, `static int sample${name}`)),
        section(joints, "class Door :", "static int sampleDoor").replace(/\n\tvoid MouseDown\([\s\S]*?(?=\n\tbool DrawControls)/, ""),
        section(joints, "class Bridge :", "static int sampleBridgeIndex"),
        ...["Filter", "Prismatic", "Parallel", "Revolute", "Wheel", "Motor"].map((name) =>
            section(joints, `class ${name}Joint :`, `static int sample${name}Joint`)),
        section(events, "class SensorVisit :", "static int sampleSensorVisit"),
        section(events, "class HitEvent :", "static int sampleHitEvent"),
        section(events, "class MoveEvent :", "static int sampleMoveEvent"),
        section(events, "class SensorHits :", "static int sampleSensorHits"),
        section(events, "class PersistentContact :", "static int samplePersistentContact"),
        section(events, "class JointEvent :", "static int sampleJointEvent"),
        section(events, "struct BodyUserData", "static int sampleContactEvent"),
    ].join("\n");
    const grab = section(sample, "\t\t// Ctrl click grabs", "\n\telse if ( modifiers & MOD_SHIFT )")
        .replace(/\t\tPickRay pickRay = m_camera->BuildPickRay\( p.x, p.y \);/, "")
        .replace(/\n\t}\s*$/, "");
    const up = section(sample, "void Sample::MouseUp(", "void Sample::MouseMove(");
    const meshLoader = [
        readFileSync(join(process.env.BOX3D ?? "", "samples/mesh_loader.h"), "utf8"),
        "#define TINYOBJLOADER_IMPLEMENTATION\n#define TINYOBJLOADER_USE_MAPBOX_EARCUT",
        readFileSync(join(process.env.BOX3D ?? "", "samples/tiny_obj_loader.h"), "utf8").replace('#include "earcut.h"', ''),
        readFileSync(join(process.env.BOX3D ?? "", "samples/mesh_loader.cpp"), "utf8")
            .replace('#include "mesh_loader.h"', '')
            .replace('#include "tiny_obj_loader.h"', '')
            .replace('reader.ParseFromFile( path, readerConfig )', 'reader.ParseFromFile( (std::string(getenv("BOX3D")) + "/" + path), readerConfig )'),
    ].join("\n");
    const source = readFileSync(join(import.meta.dir, "sample.cpp"), "utf8")
        .replace("// SAMPLE_EARCUT", readFileSync(join(process.env.BOX3D ?? "", "samples/earcut.h"), "utf8"))
        .replace("// SAMPLE_MESH_LOADER", meshLoader)
        .replace("// SAMPLE_GRAB", grab)
        .replace("// SAMPLE_UP", up.slice(up.indexOf("{")))
        .replace("// SAMPLE_EVENTS", readFileSync(join(import.meta.dir, "sample-events.h"), "utf8"))
        .replace("// SAMPLE_SOURCES", sources);
    const path = join(tmpdir(), `box3d-sample-${Bun.hash(source).toString(16)}.cpp`);
    writeFileSync(path, source);
    return nativeBinary(path, undefined, join(samples, "../shared/benchmarks.c"));
}
