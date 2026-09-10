// Unpublished TGSL seam for the real-GPU gym differential. These functions are implementation details:
// engine code imports their sibling directly, while the gym reaches tests/ through its one sanctioned
// distribution-boundary escape.

export {
    collideBoxBox,
    collideHull,
    collideRounded,
    collideRoundedPolytope,
    polyMake,
    SatResult,
} from "../../src/standard/avbd/collide";
