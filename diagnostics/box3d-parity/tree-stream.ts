// The operation inputs survive a re-pin; native outputs replace only checkpoints and observations.
export type TreeStream = { proxyCapacity: number; ops: Record<string, any>[] };
export function treeStreamInput(gold: TreeStream): string {
    const vec = (v: string[]) => v.join(" ");
    const box = (b: {lo:string[];hi:string[]}) => `${vec(b.lo)} ${vec(b.hi)}`;
    return [gold.proxyCapacity,...gold.ops.map(op => {
        switch(op.op) {
            case "create": return `0 ${box(op.aabb)} ${op.category} ${op.userData}`;
            case "move": return `1 ${op.handle} ${box(op.aabb)}`;
            case "enlarge": return `2 ${op.handle} ${box(op.aabb)}`;
            case "destroy": return `3 ${op.handle}`;
            case "rebuild": return `4 ${Number(op.full)}`;
            case "checkpoint": return "5";
            case "query": return `6 ${box(op.aabb)} ${op.mask} ${Number(op.requireAll)}`;
            case "raycast": return `7 ${vec(op.origin)} ${vec(op.translation)} ${op.maxFraction} ${op.mask} ${Number(op.requireAll)} ${op.shrink}`;
            case "boxcast": return `8 ${box(op.box)} ${vec(op.translation)} ${op.maxFraction} ${op.mask} ${Number(op.requireAll)}`;
            case "closest": return `9 ${vec(op.point)} ${op.mask} ${Number(op.requireAll)} ${op.shrink}`;
            default: throw new Error(`unknown tree operation ${op.op}`);
        }
    })].join("\n")+"\n";
}
export function treeStreamObservations(gold: TreeStream): unknown[] {
    return gold.ops.flatMap(op => {
        if(op.op==="checkpoint")return [op.tree];
        if(["query","raycast","boxcast","closest"].includes(op.op))return [{hits:op.hits,nodeVisits:op.nodeVisits,leafVisits:op.leafVisits,...(op.op==="closest"?{minDistanceSqr:op.minDistanceSqr}:{})}];
        return [];
    });
}
