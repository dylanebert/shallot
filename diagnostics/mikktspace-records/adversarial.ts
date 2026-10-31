function mesh(positions: number[][], uv: number[][], indices: number[], normal = [0, 0, 1]) {
    return {
        mesh: 0, primitive: 0,
        vertices: Float32Array.from(positions.flatMap((p, i) => [...p, uv[i][0], ...normal, uv[i][1]])),
        indices: Uint32Array.from(indices),
    };
}

export function adversarial(name: string) {
    switch (name) {
        case "degenerate": return mesh([[0,0,0],[1,0,0],[0,1,0],[0,0,0],[2,2,0]], [[0,0],[1,0],[0,1],[0,0],[0,0]], [0,0,1,0,1,2,3,4,4,4,4,4]);
        case "zero-area": return mesh([[0,0,0],[1,0,0],[2,0,0],[0,1,0]], [[0,0],[1,0],[2,0],[0,0]], [0,1,2,0,2,3,0,3,1]);
        case "mirrored-seam": return mesh([[0,0,0],[1,0,0],[0,1,0],[-1,0,0],[0,-1,0]], [[0,0],[1,0],[0,1],[1,0],[0,-1]], [0,1,2,0,3,4]);
        case "seam": return mesh([[0,0,0],[1,0,0],[0,1,0],[0,0,0],[0,1,0],[-1,0,0]], [[0,0],[1,0],[0,1],[1,1],[0,1],[1,0]], [0,1,2,3,4,5]);
        case "shared-vertex": {
            const count = 256;
            const positions = [[0,0,0], ...Array.from({length:count}, (_,i) => [Math.cos(i*Math.PI*2/count),Math.sin(i*Math.PI*2/count),0])];
            return mesh(positions, positions.map(p=>[p[0],p[1]]), Array.from({length:count},(_,i)=>[0,i+1,(i+1)%count+1]).flat());
        }
        case "non-manifold": return mesh([[0,0,0],[1,0,0],[0,1,0],[0,-1,0],[0,2,0],[0,-2,0]], [[0,0],[1,0],[0,1],[0,-1],[0,2],[0,-2]], [0,1,2,1,0,3,0,1,4,1,0,5]);
        case "opposite": return mesh([[0,0,0],[1,0,0],[0,1,0],[-1,0,0]], [[0,0],[1,0],[0,1],[1,0]], [0,1,2,0,2,3]);
        case "all-degenerate": return mesh([[0,0,0],[0,0,0],[0,0,0]], [[0,0],[1,0],[0,1]], [0,1,2]);
        // The welded bounding box chooses Z rather than X or Y.
        case "z-channel": return mesh([[0,0,0],[1,0,0],[0,0,20]], [[0,0],[1,0],[0,1]], [0,1,2], [0,1,0]);
        // Finite positions overflow the welding separator's min+max sum.
        case "nonfinite-sep": return mesh([[2e38,2e38,2e38],[2e38,2e38,2e38],[2e38,2e38,2e38]], [[0,0],[1,0],[0,1]], [0,1,2]);
        // The UV-degenerate second face adopts its first healthy group's orientation.
        case "connected-any": return mesh([[0,0,0],[1,0,0],[0,1,0],[0,-1,0]], [[0,0],[1,0],[0,1],[2,0]], [0,1,2,1,0,3]);
        // A same-direction candidate precedes the opposite-direction neighbor.
        case "nonmanifold-skip": return mesh([[0,0,0],[1,0,0],[0,1,0],[0,2,0],[0,-1,0]], [[0,0],[1,1],[0,1],[0,1],[0,-2]], [0,1,2,0,1,3,1,0,4]);
        // Finding the first good triangle skips two consecutive degenerate faces.
        case "skip-degen": return mesh([[0,0,0],[1,0,0],[0,1,0]], [[0,0],[1,0],[0,1]], [0,0,0,1,1,1,0,1,2]);
        default: throw new Error(`Unknown adversarial mesh ${name}`);
    }
}
export const adversarialNames = ["degenerate", "zero-area", "mirrored-seam", "seam", "shared-vertex", "non-manifold", "opposite", "all-degenerate", "z-channel", "nonfinite-sep", "connected-any", "nonmanifold-skip", "skip-degen"];
