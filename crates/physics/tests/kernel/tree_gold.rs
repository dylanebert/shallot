//! Exact native checkpoints and traversal observations over the pinned operation stream.
use serde_json::Value;
use shallot_physics::dynamic_tree::{Node, Proxy, Tree, EMPTY, MOVED};
use shallot_physics::math::Vec3;
fn load() -> Value {
    serde_json::from_str(include_str!(
        "../../../../src/standard/physics/collision/tree.gold.json"
    ))
    .unwrap()
}
fn bits(v: &Value) -> u32 {
    u32::from_str_radix(v.as_str().unwrap().trim_start_matches("0x"), 16).unwrap()
}
fn float(v: &Value) -> f32 {
    f32::from_bits(bits(v))
}
fn vector(v: &Value) -> Vec3 {
    Vec3::new(float(&v[0]), float(&v[1]), float(&v[2]))
}
fn insert(tree: &mut Tree, x: f32, category: u64) -> usize {
    tree.create(
        Vec3::new(x, 0.0, 0.0),
        Vec3::new(x + 1.0, 1.0, 1.0),
        category,
        tree.count as u64,
        false,
    )
}
fn hits(tree: &Tree, mask: u64, all: bool) -> Vec<usize> {
    let mut hits = Vec::new();
    tree.query(
        Vec3::new(-100.0, -100.0, -100.0),
        Vec3::new(100.0, 100.0, 100.0),
        mask,
        all,
        |id, _| {
            hits.push(id);
            true
        },
    );
    hits
}
#[test]
fn empty_tree_has_no_visits() {
    let t = Tree::new(16);
    assert_eq!(
        t.query(Vec3::ZERO, Vec3::ZERO, u64::MAX, false, |_, _| panic!(
            "empty tree hit"
        )),
        (0, 0)
    );
    assert!(t.valid());
}
#[test]
fn first_proxy_is_root_without_internal_parent() {
    let mut t = Tree::new(16);
    let id = insert(&mut t, 0.0, u64::MAX);
    assert_eq!(t.proxies[id].node, 0);
    assert_eq!(t.end, 2);
    assert_eq!(t.count, 1);
    assert_eq!(t.nodes[0].height(), 0);
    assert_eq!(t.parents[0], -1);
    assert_eq!(t.nodes[1].flags, EMPTY);
    assert!(t.valid());
}
#[test]
fn destroying_all_proxies_frees_sibling_pairs_and_proxies() {
    let mut t = Tree::new(16);
    let ids: Vec<_> = (0..10)
        .map(|i| insert(&mut t, i as f32, u64::MAX))
        .collect();
    for id in ids {
        t.destroy(id);
        assert!(t.valid());
    }
    assert_eq!(t.count, 0);
    assert_eq!(t.nodes[0].flags, EMPTY);
    let mut count = 0;
    let mut i = t.pair_free;
    while i != -1 {
        count += 2;
        i = t.parents[i as usize];
    }
    assert_eq!(count, t.end - 2);
}
#[test]
fn category_changes_only_touch_proxy_storage() {
    let mut t = Tree::new(16);
    let a = insert(&mut t, 0.0, 1);
    insert(&mut t, 0.5, 2);
    insert(&mut t, 10.0, 4);
    let root = t.nodes[0];
    t.proxies[a].category = 16;
    assert_eq!(t.nodes[0].flags, root.flags);
    assert_eq!(hits(&t, 16, false), vec![a]);
    assert!(t.valid());
}
#[test]
fn any_bit_filter_keeps_high_and_low_words_distinct() {
    for (category, mask, want) in [
        (256u64 << 32, 256u64 << 32, 1),
        (1u64 << 63, 1u64 << 63, 1),
        (256, 256u64 << 32, 0),
        (256u64 << 32, 256, 0),
    ] {
        let mut t = Tree::new(16);
        insert(&mut t, 0.0, category);
        assert_eq!(hits(&t, mask, false).len(), want);
    }
}
#[test]
fn pool_growth_preserves_proxy_identity_and_categories() {
    let mut t = Tree::new(16);
    let mut ids: Vec<_> = (0..40)
        .map(|i| insert(&mut t, i as f32, u64::MAX))
        .collect();
    assert!(t.nodes.len() > 32);
    assert!(t.proxies.len() > 16);
    let mut actual = hits(&t, u64::MAX, true);
    ids.sort_unstable();
    actual.sort_unstable();
    assert_eq!(actual, ids);
    assert!(t.valid());
}
fn checkpoint<'a>(gold: &'a Value, name: &str) -> &'a Value {
    &gold["ops"]
        .as_array()
        .unwrap()
        .iter()
        .find(|op| op["op"] == "checkpoint" && op["name"] == name)
        .unwrap()["tree"]
}
fn compare(t: &Tree, g: &Value) {
    assert_eq!(t.end, g["end"].as_u64().unwrap() as usize, "node end");
    assert_eq!(
        t.nodes.len(),
        g["capacity"].as_u64().unwrap() as usize,
        "node capacity"
    );
    assert_eq!(
        t.count,
        g["count"].as_u64().unwrap() as usize,
        "proxy count"
    );
    assert_eq!(
        t.proxies.len(),
        g["proxyCapacity"].as_u64().unwrap() as usize
    );
    assert_eq!(t.pair_free, g["pairFree"].as_i64().unwrap() as i32);
    assert_eq!(t.proxy_free, g["proxyFree"].as_i64().unwrap() as i32);
    assert_eq!(t.ordered, g["ordered"].as_bool().unwrap());
    for (i, expected) in g["nodes"].as_array().unwrap().iter().enumerate() {
        let n = t.nodes[i];
        let actual = [
            n.lo.x.to_bits(),
            n.lo.y.to_bits(),
            n.lo.z.to_bits(),
            n.hi.x.to_bits(),
            n.hi.y.to_bits(),
            n.hi.z.to_bits(),
            n.flags,
            n.height as u32,
        ];
        for (k, b) in actual.into_iter().enumerate() {
            assert_eq!(b, expected[k].as_u64().unwrap() as u32, "node {i} word {k}");
        }
        assert_eq!(
            t.parents[i],
            expected[8].as_i64().unwrap() as i32,
            "parent {i}"
        );
    }
    for (i, expected) in g["proxies"].as_array().unwrap().iter().enumerate() {
        let p = t.proxies[i];
        assert_eq!(
            p.user.to_string(),
            expected[0].as_str().unwrap(),
            "proxy {i} user"
        );
        assert_eq!(
            p.category.to_string(),
            expected[1].as_str().unwrap(),
            "proxy {i} category"
        );
        assert_eq!(p.node, expected[2].as_i64().unwrap() as i32);
        assert_eq!(p.next, expected[3].as_i64().unwrap() as i32);
    }
    assert!(t.valid());
}
fn reconstruct(g: &Value) -> Tree {
    let mut t = Tree::new(16);
    t.end = g["end"].as_u64().unwrap() as usize;
    t.nodes
        .resize(g["capacity"].as_u64().unwrap() as usize, Node::empty());
    t.parents.resize(t.nodes.len(), -1);
    t.pair_free = g["pairFree"].as_i64().unwrap() as i32;
    t.count = g["count"].as_u64().unwrap() as usize;
    t.proxy_free = g["proxyFree"].as_i64().unwrap() as i32;
    t.ordered = g["ordered"].as_bool().unwrap();
    for (i, n) in g["nodes"].as_array().unwrap().iter().enumerate() {
        let f = |k: usize| f32::from_bits(n[k].as_u64().unwrap() as u32);
        t.nodes[i] = Node {
            lo: Vec3::new(f(0), f(1), f(2)),
            hi: Vec3::new(f(3), f(4), f(5)),
            flags: n[6].as_u64().unwrap() as u32,
            height: n[7].as_u64().unwrap() as u32 as i32,
        };
        t.parents[i] = n[8].as_i64().unwrap() as i32;
    }
    t.proxies = g["proxies"]
        .as_array()
        .unwrap()
        .iter()
        .map(|p| Proxy {
            user: p[0].as_str().unwrap().parse().unwrap(),
            category: p[1].as_str().unwrap().parse().unwrap(),
            node: p[2].as_i64().unwrap() as i32,
            next: p[3].as_i64().unwrap() as i32,
        })
        .collect();
    assert!(t.valid());
    t
}
fn queries(t: &Tree, ops: &[Value]) -> [usize; 4] {
    let mut ran = [0; 4];
    for op in ops {
        let kind = op["op"].as_str().unwrap();
        let index = match kind {
            "query" => 0,
            "raycast" => 1,
            "boxcast" => 2,
            "closest" => 3,
            _ => continue,
        };
        ran[index] += 1;
        let mask = op["mask"].as_str().unwrap().parse().unwrap();
        let all = op["requireAll"].as_bool().unwrap();
        let mut hits = Vec::new();
        let stats = match kind {
            "query" => t.query(
                vector(&op["aabb"]["lo"]),
                vector(&op["aabb"]["hi"]),
                mask,
                all,
                |id, _| {
                    hits.push(id as u64);
                    true
                },
            ),
            "raycast" | "boxcast" => {
                let is_box = kind == "boxcast";
                let lo = if is_box {
                    vector(&op["box"]["lo"])
                } else {
                    vector(&op["origin"])
                };
                let hi = if is_box { vector(&op["box"]["hi"]) } else { lo };
                t.cast(
                    lo,
                    hi,
                    vector(&op["translation"]),
                    float(&op["maxFraction"]),
                    mask,
                    all,
                    is_box,
                    |fraction, id, _| {
                        hits.push(id as u64);
                        if !is_box && float(&op["shrink"]) >= 0.0 {
                            float(&op["shrink"])
                        } else {
                            fraction
                        }
                    },
                )
            }
            _ => {
                let point = vector(&op["point"]);
                let mut distance = f32::MAX;
                let stats = t.closest(point, mask, all, &mut distance, |limit, id, _| {
                    hits.push(id as u64);
                    if op["shrink"] == 0 {
                        return limit;
                    }
                    let n = t.nodes[t.proxies[id].node as usize];
                    let clamp = |p: f32, lo: f32, hi: f32| shallot_physics::math::clampf(p, lo, hi);
                    let d = point.sub(Vec3::new(
                        clamp(point.x, n.lo.x, n.hi.x),
                        clamp(point.y, n.lo.y, n.hi.y),
                        clamp(point.z, n.lo.z, n.hi.z),
                    ));
                    d.dot(d)
                });
                assert_eq!(distance.to_bits(), bits(&op["minDistanceSqr"]));
                stats
            }
        };
        let expected: Vec<_> = op["hits"]
            .as_array()
            .unwrap()
            .iter()
            .map(|v| v.as_u64().unwrap())
            .collect();
        assert_eq!(hits, expected, "{kind} hits");
        assert_eq!(
            stats.0,
            op["nodeVisits"].as_u64().unwrap() as u32,
            "{kind} node visits"
        );
        assert_eq!(
            stats.1,
            op["leafVisits"].as_u64().unwrap() as u32,
            "{kind} leaf visits"
        );
    }
    ran
}
#[test]
fn whole_operation_stream_matches_c() {
    let gold = load();
    let mut t = Tree::new(gold["proxyCapacity"].as_u64().unwrap() as usize);
    let mut handles = Vec::new();
    let mut counts = [0; 8];
    for op in gold["ops"].as_array().unwrap() {
        let bounds = || (vector(&op["aabb"]["lo"]), vector(&op["aabb"]["hi"]));
        match op["op"].as_str().unwrap() {
            "create" => {
                counts[0] += 1;
                let (lo, hi) = bounds();
                handles.push(t.create(
                    lo,
                    hi,
                    op["category"].as_str().unwrap().parse().unwrap(),
                    op["userData"].as_str().unwrap().parse().unwrap(),
                    false,
                ));
            }
            "move" => {
                counts[1] += 1;
                let (lo, hi) = bounds();
                t.move_proxy(
                    handles[op["handle"].as_u64().unwrap() as usize],
                    lo,
                    hi,
                    false,
                );
            }
            "enlarge" => {
                counts[2] += 1;
                let (lo, hi) = bounds();
                t.enlarge(handles[op["handle"].as_u64().unwrap() as usize], lo, hi);
            }
            "destroy" => {
                counts[3] += 1;
                t.destroy(handles[op["handle"].as_u64().unwrap() as usize]);
            }
            "rebuild" => {
                counts[4] += 1;
                t.rebuild(op["full"].as_bool().unwrap());
            }
            "checkpoint" => {
                counts[5] += 1;
                compare(&t, &op["tree"]);
            }
            "query" => {
                counts[6] += queries(&t, std::slice::from_ref(op))[0];
            }
            _ => {
                counts[7] += queries(&t, std::slice::from_ref(op))[1..]
                    .iter()
                    .sum::<usize>();
            }
        }
    }
    assert_eq!(counts, [24, 3, 2, 3, 2, 5, 5, 8]);
    assert!(t.valid());
    assert!(t.nodes[..t.end].iter().all(|n| n.flags & MOVED == 0));
}
#[test]
fn rebuild_partial_matches_c() {
    let g = load();
    let mut t = reconstruct(checkpoint(&g, "afterDestroy"));
    t.rebuild(false);
    compare(&t, checkpoint(&g, "afterRebuild"));
}
#[test]
fn rebuild_full_matches_c() {
    let g = load();
    let mut t = reconstruct(checkpoint(&g, "afterRebuild"));
    t.rebuild(true);
    compare(&t, checkpoint(&g, "afterFullRebuild"));
}
#[test]
fn casts_and_closest_match_c() {
    let g = load();
    let t = reconstruct(checkpoint(&g, "afterFullRebuild"));
    assert_eq!(queries(&t, g["ops"].as_array().unwrap())[1..], [3, 2, 3]);
}
#[test]
fn query_matches_c() {
    let g = load();
    let t = reconstruct(checkpoint(&g, "afterFullRebuild"));
    assert_eq!(queries(&t, g["ops"].as_array().unwrap())[0], 5);
}
