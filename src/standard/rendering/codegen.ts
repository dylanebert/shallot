// StandardRenderer's relocatable clustered-light WGSL.

import { chunk, octEncodeWgsl, spliceNs } from "../../engine/utils";
import { clusterCell } from "./cluster";
import { distanceAttenuation, pointLightsWgsl, spotFactor } from "./lighting";

/** relocatable clustered-light WGSL (`distanceAttenuation` / `spotFactor` / `clusterCell`) so a screen-space consumer evaluates the same froxel lights sear's color FS does */
export function lightEvalWgsl(): string {
    // force the base chunks first: `PointLightGpu` belongs to the light-list chunk and `octDecodeNormal`
    // to the oct chunk, and every consumer splices both ahead of this one
    pointLightsWgsl();
    octEncodeWgsl();
    return lightEvalChunk();
}

const lightEvalChunk = chunk(
    "lightEvalWgsl",
    [distanceAttenuation, spotFactor, clusterCell],
    spliceNs,
);
