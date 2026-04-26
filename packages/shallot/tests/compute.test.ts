import { test, expect, describe, beforeEach } from "bun:test";
import { CycleError } from "../src/engine/ecs/scheduler";
import { ComputeGraph, type ComputeNode, type ExecutionContext } from "../src/standard/compute";

describe("ComputeGraph", () => {
    let graph: ComputeGraph;

    beforeEach(() => {
        graph = new ComputeGraph();
    });

    describe("Node Management", () => {
        test("should add a node", () => {
            const node: ComputeNode = {
                name: "test-node",
                inputs: [],
                outputs: [],
                execute: () => {},
            };

            graph.add(node);

            expect(graph.nodes.has("test-node")).toBe(true);
        });

        test("should reject duplicate node names", () => {
            const node: ComputeNode = {
                name: "dup",
                inputs: [],
                outputs: [],
                execute: () => {},
            };

            graph.add(node);
            expect(() => graph.add(node)).toThrow("Node 'dup' already exists");
        });

        test("should remove a node", () => {
            const node: ComputeNode = {
                name: "removable",
                inputs: [],
                outputs: [],
                execute: () => {},
            };

            graph.add(node);
            expect(graph.nodes.has("removable")).toBe(true);

            const removed = graph.remove("removable");
            expect(removed).toBe(true);
            expect(graph.nodes.has("removable")).toBe(false);
        });

        test("should return false when removing non-existent node", () => {
            const removed = graph.remove("nonexistent");
            expect(removed).toBe(false);
        });
    });

    describe("Compilation", () => {
        test("should compile empty graph", () => {
            const plan = graph.compile();
            expect(plan.frame).toEqual([]);
            expect(plan.view).toEqual([]);
        });

        test("should compile single node to view by default", () => {
            const node: ComputeNode = {
                name: "single",
                inputs: [],
                outputs: [],
                execute: () => {},
            };

            graph.add(node);
            const plan = graph.compile();

            expect(plan.view).toHaveLength(1);
            expect(plan.view[0].name).toBe("single");
            expect(plan.frame).toHaveLength(0);
        });

        test("should order nodes by resource dependencies", () => {
            const producer: ComputeNode = {
                name: "producer",
                inputs: [],
                outputs: ["color"],
                execute: () => {},
            };

            const consumer: ComputeNode = {
                name: "consumer",
                inputs: ["color"],
                outputs: [],
                execute: () => {},
            };

            graph.add(consumer);
            graph.add(producer);

            const plan = graph.compile();

            expect(plan.view[0].name).toBe("producer");
            expect(plan.view[1].name).toBe("consumer");
        });

        test("should detect circular dependencies", () => {
            const nodeA: ComputeNode = {
                name: "A",
                inputs: ["from-B"],
                outputs: ["from-A"],
                execute: () => {},
            };

            const nodeB: ComputeNode = {
                name: "B",
                inputs: ["from-A"],
                outputs: ["from-B"],
                execute: () => {},
            };

            graph.add(nodeA);
            graph.add(nodeB);

            expect(() => graph.compile()).toThrow(CycleError);
        });

        test("should handle diamond dependency pattern", () => {
            const nodeA: ComputeNode = {
                name: "A",
                inputs: [],
                outputs: ["a-out"],
                execute: () => {},
            };

            const nodeB: ComputeNode = {
                name: "B",
                inputs: ["a-out"],
                outputs: ["b-out"],
                execute: () => {},
            };

            const nodeC: ComputeNode = {
                name: "C",
                inputs: ["a-out"],
                outputs: ["c-out"],
                execute: () => {},
            };

            const nodeD: ComputeNode = {
                name: "D",
                inputs: ["b-out", "c-out"],
                outputs: [],
                execute: () => {},
            };

            graph.add(nodeD);
            graph.add(nodeC);
            graph.add(nodeB);
            graph.add(nodeA);

            const plan = graph.compile();
            const order = plan.view.map((n) => n.name);

            expect(order.indexOf("A")).toBeLessThan(order.indexOf("B"));
            expect(order.indexOf("A")).toBeLessThan(order.indexOf("C"));
            expect(order.indexOf("B")).toBeLessThan(order.indexOf("D"));
            expect(order.indexOf("C")).toBeLessThan(order.indexOf("D"));
        });

        test("should cache compiled plan", () => {
            const node: ComputeNode = {
                name: "cached",
                inputs: [],
                outputs: [],
                execute: () => {},
            };

            graph.add(node);

            const plan1 = graph.compile();
            const plan2 = graph.compile();

            expect(plan1).toBe(plan2);
        });

        test("should invalidate cache when node added", () => {
            const node1: ComputeNode = {
                name: "first",
                inputs: [],
                outputs: [],
                execute: () => {},
            };

            graph.add(node1);
            const plan1 = graph.compile();

            const node2: ComputeNode = {
                name: "second",
                inputs: [],
                outputs: [],
                execute: () => {},
            };

            graph.add(node2);
            const plan2 = graph.compile();

            expect(plan1).not.toBe(plan2);
            expect(plan2.view).toHaveLength(2);
        });

        test("should invalidate cache when node removed", () => {
            const node: ComputeNode = {
                name: "removable",
                inputs: [],
                outputs: [],
                execute: () => {},
            };

            graph.add(node);
            const plan1 = graph.compile();

            graph.remove("removable");
            const plan2 = graph.compile();

            expect(plan1).not.toBe(plan2);
            expect(plan2.view).toHaveLength(0);
        });

        test("should handle nodes with same resource as input and output", () => {
            const writer: ComputeNode = {
                name: "writer",
                inputs: [],
                outputs: ["resource"],
                execute: () => {},
            };

            const updater: ComputeNode = {
                name: "updater",
                inputs: ["resource"],
                outputs: ["resource"],
                execute: () => {},
            };

            graph.add(updater);
            graph.add(writer);

            const plan = graph.compile();
            expect(plan.view.map((n) => n.name)).toEqual(["writer", "updater"]);
        });

        test("should route consumers through transformer chain", () => {
            const producer: ComputeNode = {
                name: "producer",
                inputs: [],
                outputs: ["color"],
                execute: () => {},
            };

            const transformer: ComputeNode = {
                name: "transformer",
                inputs: ["color"],
                outputs: ["color"],
                execute: () => {},
            };

            const consumer: ComputeNode = {
                name: "consumer",
                inputs: ["color"],
                outputs: [],
                execute: () => {},
            };

            graph.add(consumer);
            graph.add(transformer);
            graph.add(producer);

            const order = graph.compile().view.map((n) => n.name);

            expect(order.indexOf("producer")).toBeLessThan(order.indexOf("transformer"));
            expect(order.indexOf("transformer")).toBeLessThan(order.indexOf("consumer"));
        });

        test("should not cycle when fresh producers cross-depend before a transformer", () => {
            const forward: ComputeNode = {
                name: "forward",
                inputs: [],
                outputs: ["color", "eid"],
                execute: () => {},
            };

            const overlay: ComputeNode = {
                name: "overlay",
                inputs: ["eid"],
                outputs: ["color"],
                execute: () => {},
            };

            const outline: ComputeNode = {
                name: "outline",
                inputs: ["color", "eid"],
                outputs: ["color"],
                execute: () => {},
            };

            const postprocess: ComputeNode = {
                name: "postprocess",
                inputs: ["color"],
                outputs: [],
                execute: () => {},
            };

            graph.add(overlay);
            graph.add(postprocess);
            graph.add(outline);
            graph.add(forward);

            const order = graph.compile().view.map((n) => n.name);

            expect(order.indexOf("forward")).toBeLessThan(order.indexOf("overlay"));
            expect(order.indexOf("forward")).toBeLessThan(order.indexOf("outline"));
            expect(order.indexOf("overlay")).toBeLessThan(order.indexOf("outline"));
            expect(order.indexOf("outline")).toBeLessThan(order.indexOf("postprocess"));
        });

        test("should chain multiple transformers in registration order", () => {
            const producer: ComputeNode = {
                name: "producer",
                inputs: [],
                outputs: ["color"],
                execute: () => {},
            };

            const t1: ComputeNode = {
                name: "t1",
                inputs: ["color"],
                outputs: ["color"],
                execute: () => {},
            };

            const t2: ComputeNode = {
                name: "t2",
                inputs: ["color"],
                outputs: ["color"],
                execute: () => {},
            };

            const consumer: ComputeNode = {
                name: "consumer",
                inputs: ["color"],
                outputs: [],
                execute: () => {},
            };

            graph.add(producer);
            graph.add(t1);
            graph.add(t2);
            graph.add(consumer);

            const order = graph.compile().view.map((n) => n.name);

            expect(order.indexOf("producer")).toBeLessThan(order.indexOf("t1"));
            expect(order.indexOf("t1")).toBeLessThan(order.indexOf("t2"));
            expect(order.indexOf("t2")).toBeLessThan(order.indexOf("consumer"));
        });

        test("should handle nodes with multiple outputs", () => {
            const producer: ComputeNode = {
                name: "producer",
                inputs: [],
                outputs: ["color", "depth"],
                execute: () => {},
            };

            const colorConsumer: ComputeNode = {
                name: "color-consumer",
                inputs: ["color"],
                outputs: [],
                execute: () => {},
            };

            const depthConsumer: ComputeNode = {
                name: "depth-consumer",
                inputs: ["depth"],
                outputs: [],
                execute: () => {},
            };

            graph.add(depthConsumer);
            graph.add(colorConsumer);
            graph.add(producer);

            const plan = graph.compile();
            const order = plan.view.map((n) => n.name);

            expect(order.indexOf("producer")).toBeLessThan(order.indexOf("color-consumer"));
            expect(order.indexOf("producer")).toBeLessThan(order.indexOf("depth-consumer"));
        });
    });

    describe("Subgraph Compilation", () => {
        test("should partition frame and view nodes", () => {
            const frameNode: ComputeNode = {
                name: "data",
                scope: "frame",
                inputs: [],
                outputs: ["data"],
                execute: () => {},
            };

            const viewNode: ComputeNode = {
                name: "forward",
                inputs: ["data"],
                outputs: [],
                execute: () => {},
            };

            graph.add(frameNode);
            graph.add(viewNode);

            const plan = graph.compile();

            expect(plan.frame).toHaveLength(1);
            expect(plan.frame[0].name).toBe("data");
            expect(plan.view).toHaveLength(1);
            expect(plan.view[0].name).toBe("forward");
        });

        test("should sort frame nodes independently", () => {
            const a: ComputeNode = {
                name: "a",
                scope: "frame",
                inputs: [],
                outputs: ["a-out"],
                execute: () => {},
            };

            const b: ComputeNode = {
                name: "b",
                scope: "frame",
                inputs: ["a-out"],
                outputs: ["b-out"],
                execute: () => {},
            };

            graph.add(b);
            graph.add(a);

            const plan = graph.compile();

            expect(plan.frame.map((n) => n.name)).toEqual(["a", "b"]);
        });

        test("should throw if frame node depends on view-scope resource", () => {
            const viewNode: ComputeNode = {
                name: "view-producer",
                inputs: [],
                outputs: ["view-out"],
                execute: () => {},
            };

            const frameNode: ComputeNode = {
                name: "frame-consumer",
                scope: "frame",
                inputs: ["view-out"],
                outputs: [],
                execute: () => {},
            };

            graph.add(viewNode);
            graph.add(frameNode);

            expect(() => graph.compile()).toThrow(
                "Frame-scope node 'frame-consumer' depends on view-scope resource 'view-out'",
            );
        });

        test("should allow view node to depend on frame-scope resource", () => {
            const frameNode: ComputeNode = {
                name: "upload",
                scope: "frame",
                inputs: [],
                outputs: ["batched"],
                execute: () => {},
            };

            const viewNode: ComputeNode = {
                name: "render",
                inputs: ["batched"],
                outputs: [],
                execute: () => {},
            };

            graph.add(frameNode);
            graph.add(viewNode);

            const plan = graph.compile();
            expect(plan.frame).toHaveLength(1);
            expect(plan.view).toHaveLength(1);
        });

        test("should handle mixed graph with multiple frame and view nodes", () => {
            const data: ComputeNode = {
                name: "data",
                scope: "frame",
                inputs: [],
                outputs: ["data-out"],
                execute: () => {},
            };

            const batch: ComputeNode = {
                name: "batch",
                scope: "frame",
                inputs: ["data-out"],
                outputs: ["batched"],
                execute: () => {},
            };

            const forward: ComputeNode = {
                name: "forward",
                inputs: [],
                outputs: ["color"],
                execute: () => {},
            };

            const postprocess: ComputeNode = {
                name: "postprocess",
                inputs: ["color"],
                outputs: [],
                execute: () => {},
            };

            graph.add(postprocess);
            graph.add(forward);
            graph.add(batch);
            graph.add(data);

            const plan = graph.compile();

            expect(plan.frame.map((n) => n.name)).toEqual(["data", "batch"]);
            expect(plan.view.map((n) => n.name)).toEqual(["forward", "postprocess"]);
        });
    });

    describe("Execution", () => {
        test("should execute nodes in order", () => {
            const executionOrder: string[] = [];

            const nodeA: ComputeNode = {
                name: "A",
                inputs: [],
                outputs: ["res"],
                execute: () => executionOrder.push("A"),
            };

            const nodeB: ComputeNode = {
                name: "B",
                inputs: ["res"],
                outputs: [],
                execute: () => executionOrder.push("B"),
            };

            graph.add(nodeB);
            graph.add(nodeA);

            const plan = graph.compile();
            const mockCtx = { device: {}, context: {}, format: "bgra8unorm" } as ExecutionContext;
            for (const node of plan.view) {
                node.execute(mockCtx);
            }

            expect(executionOrder).toEqual(["A", "B"]);
        });

        test("should pass context to execute", () => {
            let receivedContext: unknown = null;

            const node: ComputeNode = {
                name: "ctx-test",
                inputs: [],
                outputs: [],
                execute: (ctx) => {
                    receivedContext = ctx;
                },
            };

            graph.add(node);
            const plan = graph.compile();

            const mockCtx = { device: {}, context: {}, format: "bgra8unorm" } as ExecutionContext;
            for (const n of plan.view) {
                n.execute(mockCtx);
            }

            expect(receivedContext).toEqual(mockCtx);
        });
    });

    describe("set", () => {
        test("should add node if not exists", () => {
            const node: ComputeNode = {
                name: "new-node",
                inputs: [],
                outputs: [],
                execute: () => {},
            };

            graph.set("new-node", node);
            expect(graph.nodes.has("new-node")).toBe(true);
        });

        test("should replace node if exists", () => {
            const original: ComputeNode = {
                name: "slot",
                inputs: [],
                outputs: ["out"],
                execute: () => {},
            };

            const replacement: ComputeNode = {
                name: "slot",
                inputs: [],
                outputs: ["different"],
                execute: () => {},
            };

            graph.add(original);
            graph.set("slot", replacement);

            expect(graph.nodes.get("slot")).toBe(replacement);
            expect(graph.nodes.size).toBe(1);
        });

        test("should throw if node name does not match slot name", () => {
            const node: ComputeNode = {
                name: "wrong-name",
                inputs: [],
                outputs: [],
                execute: () => {},
            };

            expect(() => graph.set("expected-name", node)).toThrow();
        });

        test("should invalidate plan on set", () => {
            const node: ComputeNode = {
                name: "test",
                inputs: [],
                outputs: [],
                execute: () => {},
            };

            graph.add(node);
            graph.compile();

            const replacement: ComputeNode = {
                name: "test",
                inputs: [],
                outputs: [],
                execute: () => {},
            };

            graph.set("test", replacement);
            expect(graph.planCached).toBe(false);
        });
    });

    describe("Sub-graphs", () => {
        test("sub-graph nodes only included when that sub-graph is compiled", () => {
            const shared: ComputeNode = {
                name: "overlay",
                inputs: [],
                outputs: [],
                execute: () => {},
            };
            graph.add(shared);

            const rasterNode: ComputeNode = {
                name: "forward",
                inputs: [],
                outputs: [],
                execute: () => {},
            };
            graph.subGraph("raster").add(rasterNode);

            const rtNode: ComputeNode = {
                name: "rt-render",
                inputs: [],
                outputs: [],
                execute: () => {},
            };
            graph.subGraph("raytracing").add(rtNode);

            const rasterPlan = graph.compile("raster");
            expect(rasterPlan.view.map((n) => n.name)).toEqual(["overlay", "forward"]);

            const rtPlan = graph.compile("raytracing");
            expect(rtPlan.view.map((n) => n.name)).toEqual(["overlay", "rt-render"]);
        });

        test("shared view nodes always included regardless of sub-graph", () => {
            const shared: ComputeNode = {
                name: "overlay",
                inputs: [],
                outputs: [],
                execute: () => {},
            };
            graph.add(shared);

            const plan = graph.compile("raster");
            expect(plan.view).toHaveLength(1);
            expect(plan.view[0].name).toBe("overlay");
        });

        test("frame nodes always included regardless of sub-graph", () => {
            const frameNode: ComputeNode = {
                name: "data",
                scope: "frame",
                inputs: [],
                outputs: ["data"],
                execute: () => {},
            };
            graph.add(frameNode);

            const rasterNode: ComputeNode = {
                name: "forward",
                inputs: [],
                outputs: [],
                execute: () => {},
            };
            graph.subGraph("raster").add(rasterNode);

            const plan = graph.compile("raster");
            expect(plan.frame).toHaveLength(1);
            expect(plan.frame[0].name).toBe("data");
        });

        test("cache per sub-graph name", () => {
            const shared: ComputeNode = {
                name: "overlay",
                inputs: [],
                outputs: [],
                execute: () => {},
            };
            graph.add(shared);

            graph.subGraph("raster").add({
                name: "forward",
                inputs: [],
                outputs: [],
                execute: () => {},
            });
            graph.subGraph("raytracing").add({
                name: "rt-render",
                inputs: [],
                outputs: [],
                execute: () => {},
            });

            const plan1 = graph.compile("raster");
            const plan2 = graph.compile("raster");
            expect(plan1).toBe(plan2);

            const plan3 = graph.compile("raytracing");
            expect(plan3).not.toBe(plan1);

            const plan4 = graph.compile("raytracing");
            expect(plan4).toBe(plan3);
        });

        test("cache invalidation on sub-graph node add", () => {
            graph.subGraph("raster").add({
                name: "forward",
                inputs: [],
                outputs: [],
                execute: () => {},
            });

            const plan1 = graph.compile("raster");

            graph.subGraph("raster").add({
                name: "shadows",
                inputs: [],
                outputs: [],
                execute: () => {},
            });

            const plan2 = graph.compile("raster");
            expect(plan2).not.toBe(plan1);
            expect(plan2.view).toHaveLength(2);
        });

        test("cache invalidation on sub-graph node remove", () => {
            graph.subGraph("raster").add({
                name: "forward",
                inputs: [],
                outputs: [],
                execute: () => {},
            });

            const plan1 = graph.compile("raster");

            graph.subGraph("raster").remove("forward");

            const plan2 = graph.compile("raster");
            expect(plan2).not.toBe(plan1);
            expect(plan2.view).toHaveLength(0);
        });

        test("shared node mutation invalidates all sub-graph caches", () => {
            graph.subGraph("raster").add({
                name: "forward",
                inputs: [],
                outputs: [],
                execute: () => {},
            });
            graph.subGraph("raytracing").add({
                name: "rt-render",
                inputs: [],
                outputs: [],
                execute: () => {},
            });

            const rasterPlan = graph.compile("raster");
            const rtPlan = graph.compile("raytracing");

            graph.add({
                name: "overlay",
                inputs: [],
                outputs: [],
                execute: () => {},
            });

            const rasterPlan2 = graph.compile("raster");
            const rtPlan2 = graph.compile("raytracing");

            expect(rasterPlan2).not.toBe(rasterPlan);
            expect(rtPlan2).not.toBe(rtPlan);
        });

        test("sub-graph nodes can depend on shared node outputs", () => {
            graph.add({
                name: "data-upload",
                scope: "frame",
                inputs: [],
                outputs: ["batched"],
                execute: () => {},
            });

            graph.subGraph("raster").add({
                name: "forward",
                inputs: ["batched"],
                outputs: ["color"],
                execute: () => {},
            });

            graph.add({
                name: "postprocess",
                inputs: ["color"],
                outputs: [],
                execute: () => {},
            });

            const plan = graph.compile("raster");
            expect(plan.frame).toHaveLength(1);
            expect(plan.frame[0].name).toBe("data-upload");

            const viewIds = plan.view.map((n) => n.name);
            expect(viewIds.indexOf("forward")).toBeLessThan(viewIds.indexOf("postprocess"));
        });

        test("sub-graph mutation does not invalidate other sub-graph cache", () => {
            graph.subGraph("raster").add({
                name: "forward",
                inputs: [],
                outputs: [],
                execute: () => {},
            });
            graph.subGraph("raytracing").add({
                name: "rt-render",
                inputs: [],
                outputs: [],
                execute: () => {},
            });

            graph.compile("raytracing");

            graph.subGraph("raster").add({
                name: "shadows",
                inputs: [],
                outputs: [],
                execute: () => {},
            });

            const rtPlan = graph.compile("raytracing");
            expect(rtPlan.view).toHaveLength(1);
            expect(rtPlan.view[0].name).toBe("rt-render");
        });

        test("shadow passes execute before frustum-cull", () => {
            const sg = graph.subGraph("raster");

            sg.add({
                name: "shadow-cascade-upload",
                inputs: ["data"],
                outputs: ["shadow-cascades"],
                execute: () => {},
            });

            sg.add({
                name: "shadow-render",
                inputs: ["shadow-cascades"],
                outputs: ["shadow-atlas"],
                execute: () => {},
            });

            sg.add({
                name: "cluster-cull",
                inputs: ["point-light-data"],
                outputs: ["cluster-data"],
                execute: () => {},
            });

            sg.add({
                name: "point-shadow-upload",
                inputs: ["point-light-data"],
                outputs: ["point-shadow-data"],
                execute: () => {},
            });

            sg.add({
                name: "point-shadow-render",
                inputs: ["point-shadow-data", "batched"],
                outputs: ["point-shadow-atlas"],
                execute: () => {},
            });

            sg.add({
                name: "frustum-cull",
                inputs: ["shadow-atlas", "point-shadow-atlas"],
                outputs: ["culled"],
                execute: () => {},
            });

            sg.add({
                name: "cull-stats",
                inputs: ["culled"],
                outputs: ["cull-stats"],
                execute: () => {},
            });

            sg.add({
                name: "forward",
                inputs: ["culled", "shadow-atlas", "point-shadow-atlas", "cluster-data"],
                outputs: ["color"],
                execute: () => {},
            });

            const plan = graph.compile("raster");
            const order = plan.view.map((n) => n.name);

            expect(order.indexOf("shadow-render")).toBeLessThan(order.indexOf("frustum-cull"));
            expect(order.indexOf("point-shadow-render")).toBeLessThan(
                order.indexOf("frustum-cull"),
            );
            expect(order.indexOf("frustum-cull")).toBeLessThan(order.indexOf("forward"));
        });

        test("check callback on sub-graph", () => {
            const sg = graph.subGraph("raytracing");
            sg.check = (eid) => eid === 42;

            expect(sg.check(42)).toBe(true);
            expect(sg.check(99)).toBe(false);
        });

        test("compile without sub-graph name includes only shared nodes", () => {
            graph.add({
                name: "overlay",
                inputs: [],
                outputs: [],
                execute: () => {},
            });
            graph.subGraph("raster").add({
                name: "forward",
                inputs: [],
                outputs: [],
                execute: () => {},
            });

            const plan = graph.compile();
            expect(plan.view).toHaveLength(1);
            expect(plan.view[0].name).toBe("overlay");
        });

        test("mixed frame/view with sub-graph nodes", () => {
            graph.add({
                name: "data",
                scope: "frame",
                inputs: [],
                outputs: ["data-out"],
                execute: () => {},
            });
            graph.add({
                name: "overlay",
                inputs: [],
                outputs: [],
                execute: () => {},
            });

            graph.subGraph("raster").add({
                name: "forward",
                inputs: [],
                outputs: [],
                execute: () => {},
            });
            graph.subGraph("raytracing").add({
                name: "rt-render",
                inputs: [],
                outputs: [],
                execute: () => {},
            });

            const plan = graph.compile("raytracing");
            expect(plan.frame).toHaveLength(1);
            expect(plan.frame[0].name).toBe("data");
            expect(plan.view.map((n) => n.name)).toEqual(["overlay", "rt-render"]);
        });
    });
});
