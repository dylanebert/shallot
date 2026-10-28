import tgpu from "typegpu";
import * as d from "typegpu/data";
import { pointCastersSchema, SunShadow, tileRectsSchema } from "./shade";
import { MAX_POINT_CASTERS } from "./shadows";

// Each schema factory call creates a distinct struct object. Reuse these instances within a resolve;
// resolving a second same-named PointCasters/TileRects instance would suffix its WGSL name and break
// free-name references. Fog resolves separately; consumers combined in one module must share instances.
const _shadowCasters = pointCastersSchema();
const _shadowRects = tileRectsSchema(MAX_POINT_CASTERS * 6);

/** Group-1 shadow resources. Color shaders touch these bindings because free WGSL shadow
 * references are invisible to TypeGPU's call-graph walk. */
export const shadowLayout = tgpu
    .bindGroupLayout({
        shadowMap: { texture: d.textureDepth2d(), visibility: ["fragment"] },
        shadowSamp: { sampler: "comparison", visibility: ["vertex", "fragment"] },
        sunShadow: { uniform: SunShadow, visibility: ["fragment"] },
        pointAtlas: { texture: d.textureDepth2d(), visibility: ["vertex", "fragment"] },
        pointShadows: { uniform: _shadowCasters, visibility: ["vertex", "fragment"] },
        tileRects: { uniform: _shadowRects, visibility: ["vertex", "fragment"] },
    })
    .$idx(1);
