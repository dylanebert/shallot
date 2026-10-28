import { readFileSync, writeFileSync } from "node:fs";
import { tmpdir } from "node:os";
import { join } from "node:path";
import { nativeBinary } from "./native";

export function levelMeshBinary(): string {
    const samples = join(process.env.BOX3D ?? "", "samples");
    const loader = [
        readFileSync(join(samples, "earcut.h"), "utf8"),
        readFileSync(join(samples, "mesh_loader.h"), "utf8"),
        "#define TINYOBJLOADER_IMPLEMENTATION\n#define TINYOBJLOADER_USE_MAPBOX_EARCUT",
        readFileSync(join(samples, "tiny_obj_loader.h"), "utf8").replace('#include "earcut.h"', ""),
        readFileSync(join(samples, "mesh_loader.cpp"), "utf8")
            .replace('#include "mesh_loader.h"', "")
            .replace('#include "tiny_obj_loader.h"', ""),
    ].join("\n");
    const source = readFileSync(join(import.meta.dir, "level-mesh.cpp"), "utf8").replace("// MESH_LOADER", loader);
    const path = join(tmpdir(), `box3d-level-mesh-${Bun.hash(source).toString(16)}.cpp`);
    writeFileSync(path, source);
    return nativeBinary(path);
}
