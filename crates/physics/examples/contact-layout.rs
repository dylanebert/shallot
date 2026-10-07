use shallot_physics::manifold_abi::*;
fn main() {
    println!("// Generated from Rust resident record offsets by build-kernel.ts.");
    for (name, value) in [
        ("DIR_STRIDE", DIR_STRIDE),
        ("DIR_COUNT", DIR_MANIFOLD_COUNT),
        ("DIR_BLOCK", DIR_MANIFOLD_BASE),
        ("DIR_FLAGS", DIR_FLAGS),
        ("DIR_FRICTION", DIR_FRICTION),
        ("DIR_RESTITUTION", DIR_RESTITUTION),
        ("DIR_ROLLING_RESISTANCE", DIR_ROLLING_RESISTANCE),
        ("DIR_TANGENT_VELOCITY", DIR_TANGENT_VELOCITY),
        ("MANIFOLD_STRIDE", MANIFOLD_STRIDE),
        ("M_NORMAL", M_NORMAL),
        ("M_TWIST", M_TWIST),
        ("M_FRICTION", M_FRICTION),
        ("M_ROLLING", M_ROLLING),
        ("M_POINT_COUNT", M_POINT_COUNT),
        ("M_POINTS", M_POINTS),
        ("POINT_STRIDE", POOL_POINT_STRIDE),
        ("P_ANCHOR_A", P_ANCHOR_A),
        ("P_ANCHOR_B", P_ANCHOR_B),
        ("P_SEPARATION", P_SEPARATION),
        ("P_BASE_SEPARATION", P_BASE_SEPARATION),
        ("P_NORMAL_IMPULSE", P_NORMAL_IMPULSE),
        ("P_TOTAL_NORMAL_IMPULSE", P_TOTAL_NORMAL_IMPULSE),
        ("P_NORMAL_VELOCITY", P_NORMAL_VELOCITY),
        ("P_FEATURE_ID", P_FEATURE_ID),
        ("P_TRIANGLE_INDEX", P_TRIANGLE_INDEX),
        ("P_PERSISTED", P_PERSISTED),
    ] {
        println!("export const {name} = {value};");
    }
    println!("export const ContactField = {{");
    for (name, value) in [
        ("flags", DIR_FLAGS),
        ("manifoldCount", DIR_MANIFOLD_COUNT),
        ("bodySimIndexA", DIR_INDEX_A),
        ("bodySimIndexB", DIR_INDEX_B),
        ("setIndex", DIR_SET_INDEX),
        ("colorIndex", DIR_COLOR_INDEX),
        ("localIndex", DIR_LOCAL_INDEX),
        ("bodyIdA", DIR_EDGE_A),
        ("prevKeyA", DIR_EDGE_A + 1),
        ("nextKeyA", DIR_EDGE_A + 2),
        ("bodyIdB", DIR_EDGE_B),
        ("prevKeyB", DIR_EDGE_B + 1),
        ("nextKeyB", DIR_EDGE_B + 2),
        ("shapeIdA", DIR_SHAPE_A),
        ("shapeIdB", DIR_SHAPE_B),
        ("childIndex", DIR_CHILD_INDEX),
        ("islandId", DIR_ISLAND_ID),
        ("islandIndex", DIR_ISLAND_INDEX),
        ("contactId", DIR_CONTACT_ID),
        ("generation", DIR_GENERATION),
    ] {
        println!("    {name}: {value},");
    }
    println!("}} as const;");
}
