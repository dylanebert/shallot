//! Bit-exact dynamic-tree gate against the Box3D C operation stream.
//! Replay starts with an empty pool; checkpoints compare allocated nodes and the free chain.
//! Separate reconstructed-state gates isolate rebuild and traversal failures.

use serde_json::Value;
use shallot_physics::math::Vec3;
use shallot_physics::tree::{self, Rebuild, STACK_SIZE, STRIDE};

fn empty_pool(capacity: usize) -> Pool {
    let mut pool = Pool {
        slots: vec![0; capacity * STRIDE],
        root: -1,
        node_count: 0,
        free_list: 0,
        proxy_count: 0,
    };
    for i in 0..capacity {
        pool.slots[i * STRIDE + 10] = if i + 1 == capacity {
            u32::MAX
        } else {
            (i + 1) as u32
        };
    }
    pool
}
fn insert_box(pool: &mut Pool, x: f32, hi: u32, lo: u32) -> i32 {
    reserve_insert(pool);
    let id = tree::create_proxy(
        &mut pool.slots,
        &mut pool.root,
        &mut pool.node_count,
        &mut pool.free_list,
        [x, 0.0, 0.0],
        [x + 1.0, 1.0, 1.0],
        hi,
        lo,
        pool.proxy_count as u64,
    );
    pool.proxy_count += 1;
    id
}
fn filtered_hits(pool: &Pool, hi: u32, lo: u32, all: bool) -> Vec<i32> {
    let mut hits = Vec::new();
    let mut stack = [0; STACK_SIZE];
    tree::query(
        &pool.slots,
        pool.root,
        pool.node_count,
        [-100.0; 3],
        [100.0; 3],
        hi,
        lo,
        all,
        &mut stack,
        |id, _| {
            hits.push(id);
            true
        },
    );
    hits
}
#[test]
fn empty_tree_has_no_visits() {
    let pool = empty_pool(31);
    let mut stack = [0; STACK_SIZE];
    assert_eq!(
        tree::query(
            &pool.slots,
            pool.root,
            pool.node_count,
            [-1.0; 3],
            [1.0; 3],
            u32::MAX,
            u32::MAX,
            false,
            &mut stack,
            |_, _| panic!("empty tree hit")
        ),
        (0, 0)
    );
}
#[test]
fn first_proxy_is_root_without_internal_parent() {
    let mut pool = empty_pool(31);
    let id = insert_box(&mut pool, 0.0, u32::MAX, u32::MAX);
    assert_eq!(pool.root, id);
    assert_eq!(pool.node_count, 1);
    assert_eq!(pool.proxy_count, 1);
    assert_eq!(pool.slots[id as usize * STRIDE + 11] >> 16, 0);
    assert_eq!(pool.slots[id as usize * STRIDE + 10], u32::MAX);
}
#[test]
fn destroying_all_proxies_frees_internal_parents() {
    let mut pool = empty_pool(31);
    let ids: Vec<_> = (0..10)
        .map(|i| insert_box(&mut pool, i as f32, u32::MAX, u32::MAX))
        .collect();
    for id in ids {
        tree::destroy_proxy(
            &mut pool.slots,
            &mut pool.root,
            &mut pool.node_count,
            &mut pool.free_list,
            id,
        );
        pool.proxy_count -= 1;
    }
    assert_eq!(pool.root, -1);
    assert_eq!(pool.proxy_count, 0);
    assert_eq!(pool.node_count, 0);
    let mut chain = Vec::new();
    let mut i = pool.free_list;
    while i != -1 {
        assert!(!chain.contains(&i));
        chain.push(i);
        i = pool.slots[i as usize * STRIDE + 10] as i32;
    }
    assert_eq!(chain.len(), 31);
}
#[test]
fn category_change_propagates_to_ancestors() {
    let mut pool = empty_pool(31);
    let a = insert_box(&mut pool, 0.0, 0, 1);
    insert_box(&mut pool, 0.5, 0, 2);
    insert_box(&mut pool, 10.0, 0, 4);
    assert_eq!(pool.slots[pool.root as usize * STRIDE + 7], 7);
    tree::set_category_bits(&mut pool.slots, a, 0, 16);
    assert_eq!(pool.slots[a as usize * STRIDE + 7], 16);
    assert_eq!(pool.slots[pool.root as usize * STRIDE + 7], 22);
    assert_eq!(pool.slots[pool.root as usize * STRIDE + 6], 0);
    assert_eq!(filtered_hits(&pool, 0, 16, false), vec![a]);
}
#[test]
fn any_bit_filter_keeps_high_and_low_words_distinct() {
    for (ch, cl, mh, ml, want) in [
        (256, 0, 256, 0, 1),
        (0x80000000, 0, 0x80000000, 0, 1),
        (0, 256, 256, 0, 0),
        (256, 0, 0, 256, 0),
    ] {
        let mut pool = empty_pool(31);
        insert_box(&mut pool, 0.0, ch, cl);
        assert_eq!(filtered_hits(&pool, mh, ml, false).len(), want);
    }
}
#[test]
fn pool_growth_preserves_category_words_and_all_proxies() {
    let mut pool = empty_pool(31);
    let mut ids: Vec<_> = (0..20)
        .map(|i| insert_box(&mut pool, i as f32, u32::MAX, u32::MAX))
        .collect();
    assert!(pool.slots.len() / STRIDE > 31);
    let mut hits = filtered_hits(&pool, u32::MAX, u32::MAX, true);
    ids.sort();
    hits.sort();
    assert_eq!(hits, ids);
}

fn load() -> Value {
    let raw = include_str!("../../../src/standard/physics/collision/tree.gold.json");
    serde_json::from_str(raw).expect("parse tree.gold.json")
}

fn hex_bits(s: &str) -> u32 {
    u32::from_str_radix(s.trim_start_matches("0x"), 16).expect("hex u32")
}

fn checkpoint<'a>(gold: &'a Value, name: &str) -> &'a Value {
    for op in gold["ops"].as_array().unwrap() {
        if op["op"] == "checkpoint" && op["name"] == name {
            return &op["tree"];
        }
    }
    panic!("checkpoint {name} not found");
}

struct Pool {
    slots: Vec<u32>,
    root: i32,
    node_count: usize,
    free_list: i32,
    proxy_count: usize,
}

fn reserve_insert(pool: &mut Pool) {
    let old = pool.slots.len() / STRIDE;
    if old - pool.node_count >= 2 {
        return;
    }
    let capacity = old + old / 2;
    pool.slots.resize(capacity * STRIDE, 0);
    for i in old..capacity {
        pool.slots[i * STRIDE + 10] = if i + 1 == capacity {
            u32::MAX
        } else {
            (i + 1) as u32
        };
    }
    if pool.free_list == -1 {
        pool.free_list = old as i32;
    } else {
        let mut i = pool.free_list;
        while pool.slots[i as usize * STRIDE + 10] as i32 != -1 {
            i = pool.slots[i as usize * STRIDE + 10] as i32;
        }
        pool.slots[i as usize * STRIDE + 10] = old as u32;
    }
}

#[test]
fn whole_operation_stream_matches_c() {
    let gold = load();
    let capacity = 2 * (gold["proxyCapacity"].as_u64().unwrap() as usize).max(16) - 1;
    let mut pool = Pool {
        slots: vec![0; capacity * STRIDE],
        root: -1,
        node_count: 0,
        free_list: 0,
        proxy_count: 0,
    };
    for i in 0..capacity {
        pool.slots[i * STRIDE + 10] = if i + 1 == capacity {
            -1i32 as u32
        } else {
            (i + 1) as u32
        };
    }
    let mut handles = Vec::new();
    let mut counts = [0; 8];
    for (index, op) in gold["ops"].as_array().unwrap().iter().enumerate() {
        let bounds = || {
            let lo = vector(&op["aabb"]["lo"]);
            let hi = vector(&op["aabb"]["hi"]);
            ([lo.x, lo.y, lo.z], [hi.x, hi.y, hi.z])
        };
        match op["op"].as_str().unwrap() {
            "create" => {
                counts[0] += 1;
                let (lo, hi) = bounds();
                let category: u64 = op["category"].as_str().unwrap().parse().unwrap();
                let user_data = op["userData"].as_str().unwrap().parse().unwrap();
                reserve_insert(&mut pool);
                handles.push(tree::create_proxy(
                    &mut pool.slots,
                    &mut pool.root,
                    &mut pool.node_count,
                    &mut pool.free_list,
                    lo,
                    hi,
                    (category >> 32) as u32,
                    category as u32,
                    user_data,
                ));
                pool.proxy_count += 1;
            }
            "move" => {
                counts[1] += 1;
                let (lo, hi) = bounds();
                tree::move_proxy(
                    &mut pool.slots,
                    &mut pool.root,
                    &mut pool.node_count,
                    &mut pool.free_list,
                    handles[op["handle"].as_u64().unwrap() as usize],
                    lo,
                    hi,
                );
            }
            "enlarge" => {
                counts[2] += 1;
                let (lo, hi) = bounds();
                tree::enlarge_proxy(
                    &mut pool.slots,
                    handles[op["handle"].as_u64().unwrap() as usize],
                    lo,
                    hi,
                );
            }
            "destroy" => {
                counts[3] += 1;
                tree::destroy_proxy(
                    &mut pool.slots,
                    &mut pool.root,
                    &mut pool.node_count,
                    &mut pool.free_list,
                    handles[op["handle"].as_u64().unwrap() as usize],
                );
                pool.proxy_count -= 1;
            }
            "rebuild" => {
                counts[4] += 1;
                run_rebuild(&mut pool, op["full"].as_bool().unwrap());
            }
            "checkpoint" => {
                counts[5] += 1;
                assert_matches(&pool, &op["tree"], pool.root, op["name"].as_str().unwrap());
                assert_invariants(&pool);
            }
            "query" => {
                counts[6] += check_queries(&pool, std::slice::from_ref(op));
            }
            "raycast" | "boxcast" | "closest" => {
                counts[7] += check_casts(&pool, std::slice::from_ref(op))
                    .iter()
                    .sum::<usize>();
            }
            other => panic!("operation {index}: unknown operation {other}"),
        }
    }
    assert_eq!(counts, [24, 3, 2, 3, 2, 5, 5, 8]);
    assert_invariants(&pool);
    for node in pool.slots.as_chunks::<STRIDE>().0 {
        assert_eq!(
            node[11] & 3,
            node[11] & 1,
            "allocated node remains enlarged"
        );
    }
}

fn assert_invariants(pool: &Pool) {
    fn walk(pool: &Pool, i: i32, parent: i32, seen: &mut Vec<i32>) -> u32 {
        assert!(!seen.contains(&i), "cycle or shared child");
        seen.push(i);
        let n = i as usize * STRIDE;
        let node = &pool.slots[n..n + STRIDE];
        assert_eq!(node[10] as i32, parent);
        assert_ne!(node[11] & 1, 0);
        if node[11] & 4 != 0 {
            assert_eq!(node[11] >> 16, 0);
            return 0;
        }
        let a = node[8] as i32;
        let b = node[9] as i32;
        let height = 1 + walk(pool, a, i, seen).max(walk(pool, b, i, seen));
        assert_eq!(node[11] >> 16, height);
        for k in 6..8 {
            assert_eq!(
                node[k],
                pool.slots[a as usize * STRIDE + k] | pool.slots[b as usize * STRIDE + k]
            );
        }
        let (lo, hi) = tree::node_aabb(&pool.slots, i);
        for child in [a, b] {
            let (cl, ch) = tree::node_aabb(&pool.slots, child);
            for k in 0..3 {
                assert!(lo[k] <= cl[k] && hi[k] >= ch[k]);
            }
        }
        if (pool.slots[a as usize * STRIDE + 11] | pool.slots[b as usize * STRIDE + 11]) & 2 != 0 {
            assert_ne!(node[11] & 2, 0);
        }
        height
    }
    let mut seen = Vec::new();
    if pool.root != -1 {
        walk(pool, pool.root, -1, &mut seen);
    }
    assert_eq!(seen.len(), pool.node_count);
}

/// Reconstruct a node pool from a checkpoint's captured state — the exact C memory image (allocated
/// records at their indices + the free-list chain), so a rebuild replays C's node-index allocation.
fn reconstruct(tree_json: &Value) -> Pool {
    let cap = tree_json["nodeCapacity"].as_u64().unwrap() as usize;
    let mut slots = vec![0u32; cap * STRIDE];

    for node in tree_json["nodes"].as_array().unwrap() {
        let i = node["i"].as_u64().unwrap() as usize;
        let n = i * STRIDE;
        let aabb = &node["aabb"];
        let lo = aabb["lo"].as_array().unwrap();
        let hi = aabb["hi"].as_array().unwrap();
        slots[n] = hex_bits(lo[0].as_str().unwrap());
        slots[n + 1] = hex_bits(lo[1].as_str().unwrap());
        slots[n + 2] = hex_bits(lo[2].as_str().unwrap());
        slots[n + 3] = hex_bits(hi[0].as_str().unwrap());
        slots[n + 4] = hex_bits(hi[1].as_str().unwrap());
        slots[n + 5] = hex_bits(hi[2].as_str().unwrap());

        let category: u64 = node["category"].as_str().unwrap().parse().unwrap();
        slots[n + 6] = (category >> 32) as u32;
        slots[n + 7] = (category & 0xffff_ffff) as u32;

        let leaf = node["leaf"].as_bool().unwrap();
        if leaf {
            let ud: u32 = node["userData"].as_str().unwrap().parse().unwrap();
            slots[n + 8] = ud;
            slots[n + 9] = 0;
        } else {
            slots[n + 8] = node["child1"].as_i64().unwrap() as i32 as u32;
            slots[n + 9] = node["child2"].as_i64().unwrap() as i32 as u32;
        }
        slots[n + 10] = node["parent"].as_i64().unwrap() as i32 as u32;
        let height = node["height"].as_u64().unwrap() as u32;
        let flags = node["flags"].as_u64().unwrap() as u32;
        slots[n + 11] = (height << 16) | flags;
    }

    // Free chain: each free node's `next` (slot 10) points at the next, tail at -1. flags = 0.
    let chain: Vec<i32> = tree_json["freeChain"]
        .as_array()
        .unwrap()
        .iter()
        .map(|v| v.as_i64().unwrap() as i32)
        .collect();
    for (k, &idx) in chain.iter().enumerate() {
        let n = idx as usize * STRIDE;
        let next = if k + 1 < chain.len() {
            chain[k + 1]
        } else {
            -1
        };
        slots[n + 10] = next as u32;
        slots[n + 11] = 0;
    }

    Pool {
        slots,
        root: tree_json["root"].as_i64().unwrap() as i32,
        node_count: tree_json["nodeCount"].as_u64().unwrap() as usize,
        free_list: tree_json["freeList"].as_i64().unwrap() as i32,
        proxy_count: tree_json["proxyCount"].as_u64().unwrap() as usize,
    }
}

/// Assert a rebuilt pool matches the expected checkpoint node-for-node (structure, aabb, category,
/// height, links) plus the tree scalars.
fn assert_matches(pool: &Pool, expect: &Value, new_root: i32, label: &str) {
    assert_eq!(
        new_root,
        expect["root"].as_i64().unwrap() as i32,
        "{label}: root"
    );
    assert_eq!(
        pool.node_count,
        expect["nodeCount"].as_u64().unwrap() as usize,
        "{label}: nodeCount"
    );
    assert_eq!(
        pool.free_list,
        expect["freeList"].as_i64().unwrap() as i32,
        "{label}: freeList"
    );

    assert_eq!(
        pool.slots.len() / STRIDE,
        expect["nodeCapacity"].as_u64().unwrap() as usize,
        "{label}: nodeCapacity"
    );
    assert_eq!(
        pool.proxy_count,
        expect["proxyCount"].as_u64().unwrap() as usize,
        "{label}: proxyCount"
    );
    let mut chain = Vec::new();
    let mut next = pool.free_list;
    while next != -1 {
        assert!(
            chain.len() < pool.slots.len() / STRIDE,
            "{label}: cyclic free chain"
        );
        chain.push(next);
        next = pool.slots[next as usize * STRIDE + 10] as i32;
    }
    let expected_chain: Vec<i32> = expect["freeChain"]
        .as_array()
        .unwrap()
        .iter()
        .map(|v| v.as_i64().unwrap() as i32)
        .collect();
    assert_eq!(chain, expected_chain, "{label}: freeChain");
    let allocated: Vec<usize> = (0..pool.slots.len() / STRIDE)
        .filter(|&i| pool.slots[i * STRIDE + 11] & 1 != 0)
        .collect();
    let expected_allocated: Vec<usize> = expect["nodes"]
        .as_array()
        .unwrap()
        .iter()
        .map(|v| v["i"].as_u64().unwrap() as usize)
        .collect();
    assert_eq!(allocated, expected_allocated, "{label}: allocated nodes");
    for node in expect["nodes"].as_array().unwrap() {
        let i = node["i"].as_u64().unwrap() as usize;
        let n = i * STRIDE;
        let s = &pool.slots;
        for (k, key) in ["lo", "hi"].iter().enumerate() {
            let arr = node["aabb"][key].as_array().unwrap();
            for c in 0..3 {
                let got = s[n + k * 3 + c];
                let want = hex_bits(arr[c].as_str().unwrap());
                assert_eq!(got, want, "{label}: node {i} aabb {key}[{c}]");
            }
        }
        let category: u64 = node["category"].as_str().unwrap().parse().unwrap();
        assert_eq!(
            s[n + 6],
            (category >> 32) as u32,
            "{label}: node {i} category hi"
        );
        assert_eq!(
            s[n + 7],
            (category & 0xffff_ffff) as u32,
            "{label}: node {i} category lo"
        );
        assert_eq!(
            s[n + 11] >> 16,
            node["height"].as_u64().unwrap() as u32,
            "{label}: node {i} height"
        );
        assert_eq!(
            s[n + 11] & 0xffff,
            node["flags"].as_u64().unwrap() as u32,
            "{label}: node {i} flags"
        );
        assert_eq!(
            s[n + 10] as i32,
            node["parent"].as_i64().unwrap() as i32,
            "{label}: node {i} parent"
        );
        if node["leaf"].as_bool().unwrap() {
            let ud: u32 = node["userData"].as_str().unwrap().parse().unwrap();
            assert_eq!(s[n + 8], ud, "{label}: node {i} userData");
            assert_eq!(s[n + 9], 0, "{label}: node {i} userData high");
        } else {
            assert_eq!(
                s[n + 8] as i32,
                node["child1"].as_i64().unwrap() as i32,
                "{label}: node {i} child1"
            );
            assert_eq!(
                s[n + 9] as i32,
                node["child2"].as_i64().unwrap() as i32,
                "{label}: node {i} child2"
            );
        }
    }
}

fn run_rebuild(pool: &mut Pool, full: bool) -> i32 {
    let mut leaf_indices = vec![0i32; pool.proxy_count.max(1)];
    let mut leaf_centers = vec![0f32; pool.proxy_count.max(1) * 3];
    let mut gather = vec![0i32; STACK_SIZE];
    let mut build = vec![0i32; STACK_SIZE * 5];
    let mut rb = Rebuild {
        node_count: pool.node_count,
        free_list: pool.free_list,
        leaf_indices: &mut leaf_indices,
        leaf_centers: &mut leaf_centers,
        gather_stack: &mut gather,
        build_stack: &mut build,
    };
    let new_root = tree::rebuild(&mut pool.slots, pool.root, pool.proxy_count, full, &mut rb);
    pool.node_count = rb.node_count;
    pool.free_list = rb.free_list;
    pool.root = new_root;
    new_root
}

#[test]
fn rebuild_partial_matches_c() {
    let gold = load();
    let mut pool = reconstruct(checkpoint(&gold, "afterDestroy"));
    let new_root = run_rebuild(&mut pool, false);
    assert_matches(
        &pool,
        checkpoint(&gold, "afterRebuild"),
        new_root,
        "afterRebuild",
    );
}

#[test]
fn rebuild_full_matches_c() {
    let gold = load();
    let mut pool = reconstruct(checkpoint(&gold, "afterRebuild"));
    let new_root = run_rebuild(&mut pool, true);
    assert_matches(
        &pool,
        checkpoint(&gold, "afterFullRebuild"),
        new_root,
        "afterFullRebuild",
    );
}

fn vector(value: &Value) -> Vec3 {
    let a = value.as_array().unwrap();
    Vec3::new(float(&a[0]), float(&a[1]), float(&a[2]))
}
fn float(value: &Value) -> f32 {
    f32::from_bits(hex_bits(value.as_str().unwrap()))
}

#[test]
fn casts_and_closest_match_c() {
    let gold = load();
    let pool = reconstruct(checkpoint(&gold, "afterFullRebuild"));
    assert_eq!(
        check_casts(&pool, gold["ops"].as_array().unwrap()),
        [3, 2, 3]
    );
}

fn check_casts(pool: &Pool, ops: &[Value]) -> [usize; 3] {
    let mut ran = [0; 3];
    for op in ops {
        let kind = op["op"].as_str().unwrap();
        if !["raycast", "boxcast", "closest"].contains(&kind) {
            continue;
        }
        let mask: u64 = op["mask"].as_str().unwrap().parse().unwrap();
        let (hi, lo) = ((mask >> 32) as u32, mask as u32);
        let all = op["requireAll"].as_bool().unwrap();
        let mut hits = Vec::new();
        let stats = match kind {
            "raycast" => {
                ran[0] += 1;
                let shrink = float(&op["shrink"]);
                tree::ray_cast(
                    &pool.slots,
                    pool.root,
                    pool.node_count,
                    vector(&op["origin"]),
                    vector(&op["translation"]),
                    float(&op["maxFraction"]),
                    hi,
                    lo,
                    all,
                    |fraction, id, _| {
                        hits.push(id as i64);
                        if shrink < 0.0 {
                            fraction
                        } else {
                            shrink
                        }
                    },
                )
            }
            "boxcast" => {
                ran[1] += 1;
                tree::box_cast(
                    &pool.slots,
                    pool.root,
                    pool.node_count,
                    vector(&op["box"]["lo"]),
                    vector(&op["box"]["hi"]),
                    vector(&op["translation"]),
                    float(&op["maxFraction"]),
                    hi,
                    lo,
                    all,
                    |fraction, id, _| {
                        hits.push(id as i64);
                        fraction
                    },
                )
            }
            _ => {
                ran[2] += 1;
                let point = vector(&op["point"]);
                let mut min_sqr = f32::MAX;
                let stats = tree::query_closest(
                    &pool.slots,
                    pool.root,
                    pool.node_count,
                    point,
                    hi,
                    lo,
                    all,
                    &mut min_sqr,
                    |distance, id, _| {
                        hits.push(id as i64);
                        if op["shrink"] == 0 {
                            return distance;
                        }
                        let n = id as usize * STRIDE;
                        let mut r = [0.0; 3];
                        for (i, p) in [point.x, point.y, point.z].into_iter().enumerate() {
                            let lower = f32::from_bits(pool.slots[n + i]);
                            let upper = f32::from_bits(pool.slots[n + 3 + i]);
                            r[i] = p - shallot_physics::math::clampf(p, lower, upper);
                        }
                        (r[0] * r[0] + r[1] * r[1]) + r[2] * r[2]
                    },
                );
                assert_eq!(
                    min_sqr.to_bits(),
                    hex_bits(op["minDistanceSqr"].as_str().unwrap())
                );
                stats
            }
        };
        let expected: Vec<i64> = op["hits"]
            .as_array()
            .unwrap()
            .iter()
            .map(|v| v.as_i64().unwrap())
            .collect();
        assert_eq!(hits, expected, "{kind} hits");
        assert_eq!(
            stats.0,
            op["nodeVisits"].as_u64().unwrap() as u32,
            "{kind} nodeVisits"
        );
        assert_eq!(
            stats.1,
            op["leafVisits"].as_u64().unwrap() as u32,
            "{kind} leafVisits"
        );
    }
    ran
}

#[test]
fn query_matches_c() {
    let gold = load();
    let pool = reconstruct(checkpoint(&gold, "afterFullRebuild"));
    assert_eq!(check_queries(&pool, gold["ops"].as_array().unwrap()), 5);
}

fn check_queries(pool: &Pool, ops: &[Value]) -> usize {
    let mut stack = vec![0i32; STACK_SIZE];
    let mut ran = 0;
    for op in ops {
        if op["op"] != "query" {
            continue;
        }
        ran += 1;
        let aabb = &op["aabb"];
        let lo_a = aabb["lo"].as_array().unwrap();
        let hi_a = aabb["hi"].as_array().unwrap();
        let lo = [
            f32::from_bits(hex_bits(lo_a[0].as_str().unwrap())),
            f32::from_bits(hex_bits(lo_a[1].as_str().unwrap())),
            f32::from_bits(hex_bits(lo_a[2].as_str().unwrap())),
        ];
        let hi = [
            f32::from_bits(hex_bits(hi_a[0].as_str().unwrap())),
            f32::from_bits(hex_bits(hi_a[1].as_str().unwrap())),
            f32::from_bits(hex_bits(hi_a[2].as_str().unwrap())),
        ];
        let mask: u64 = op["mask"].as_str().unwrap().parse().unwrap();
        let mask_hi = (mask >> 32) as u32;
        let mask_lo = (mask & 0xffff_ffff) as u32;
        let require_all = op["requireAll"].as_bool().unwrap();

        let mut hits: Vec<i64> = Vec::new();
        let (nv, lv) = tree::query(
            &pool.slots,
            pool.root,
            pool.node_count,
            lo,
            hi,
            mask_hi,
            mask_lo,
            require_all,
            &mut stack,
            |node_id, _ud| {
                hits.push(node_id as i64);
                true
            },
        );

        let want_hits: Vec<i64> = op["hits"]
            .as_array()
            .unwrap()
            .iter()
            .map(|v| v.as_i64().unwrap())
            .collect();
        assert_eq!(hits, want_hits, "query hits");
        assert_eq!(nv, op["nodeVisits"].as_u64().unwrap() as u32, "nodeVisits");
        assert_eq!(lv, op["leafVisits"].as_u64().unwrap() as u32, "leafVisits");
    }
    ran
}
