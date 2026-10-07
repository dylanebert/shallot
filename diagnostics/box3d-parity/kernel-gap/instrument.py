"""Diagnostic-only source instrumentation in the archive passed as argv[1]."""
import pathlib, re, sys
root = pathlib.Path(sys.argv[1])
src = root / 'crates/physics/src'
labels = ['support_scalar', 'support_wide', 'support_face', 'triangle_support', 'clip_input', 'clip_output', 'clip_final', 'hull_sat', 'hull_sat_hit', 'hull_sat_miss', 'face_axes', 'edge_candidates', 'contact_points', 'clip_copy_points', 'clip_zero_points', 'tree_queries', 'tree_nodes', 'tree_leaves', 'rebuild_leaves']
(src/'diagnostic.rs').write_text('use core::sync::atomic::{AtomicU32, Ordering};\nstatic COUNTS: [AtomicU32; %d] = [const { AtomicU32::new(0) }; %d];\npub fn add(i: usize, n: usize) { COUNTS[i].fetch_add(n as u32, Ordering::Relaxed); }\n#[no_mangle]\npub extern "C" fn diagnosticCount(i: usize) -> u32 { COUNTS[i].swap(0, Ordering::Relaxed) }\n' % (len(labels),len(labels)))
p=src/'lib.rs'; p.write_text(p.read_text()+'\nmod diagnostic;\n')
def wrap(file, name, before='', after=''):
    p=src/file; text=p.read_text(); m=re.search(r'\bfn '+name+r'(?:<[^\n]*>)?\s*\(',text)
    if not m: raise Exception(name)
    start=text.index('{',m.end()); depth=1; end=start+1
    while depth:
        if text[end]=='{': depth+=1
        if text[end]=='}': depth-=1
        end+=1
    body=text[start+1:end-1]
    text=text[:start+1]+before+'\nlet diagnostic_result = (|| {'+body+'})();\n'+after+'\ndiagnostic_result\n'+text[end-1:]
    p.write_text(text)
def add(label,n='1'): return 'crate::diagnostic::add(%d, %s);' % (labels.index(label), n)
for name,label in [('support_vertex','support_scalar'),('support_vertex_wide','support_wide'),('support_face','support_face')]: wrap('hull.rs',name,add(label))
wrap('triangle_manifold.rs','triangle_support',add('triangle_support'))
wrap('tree.rs','query',add('tree_queries'),add('tree_nodes','diagnostic_result.0 as usize')+add('tree_leaves','diagnostic_result.1 as usize'))
p=src/'tree.rs'; t=p.read_text(); t=t.replace('    build_tree(pool, rb, leaf_count)','    '+add('rebuild_leaves','leaf_count')+'\n    build_tree(pool, rb, leaf_count)'); p.write_text(t)
wrap('manifold.rs','clip_polygon',add('clip_input','count'),add('clip_output','diagnostic_result'))
wrap('manifold.rs','clip_segment',add('clip_input','2'),add('clip_output','diagnostic_result'))
wrap('manifold.rs','clip_segment_to_hull_face',after=add('clip_final','diagnostic_result'))
wrap('manifold.rs','collide_hulls',add('hull_sat'),add('hull_sat_hit','cache.hit as usize')+add('hull_sat_miss','(cache.hit == 0) as usize'))
p=src/'manifold.rs'; t=p.read_text(); t=t.replace('let plane = hull_a.planes[i];',add('face_axes')+'\nlet plane = hull_a.planes[i];').replace('let plane = hull_b.planes[i];',add('face_axes')+'\nlet plane = hull_b.planes[i];').replace('let dir = a_dir.load(i);',add('edge_candidates','(na - i).min(4)')+'\nlet dir = a_dir.load(i);').replace('point_count = point_count.min(MAX_CLIP_POINTS);',add('clip_final','point_count')+'\npoint_count = point_count.min(MAX_CLIP_POINTS);'); p.write_text(t)
wrap('triangle_manifold.rs','clip_segment',after=add('clip_final','if diagnostic_result { 2 } else { 0 }'))
p=src/'manifold.rs'; t=p.read_text(); t=t.replace('input[..point_count].copy_from_slice(&scratch[..point_count]);',add('clip_copy_points','point_count')+'\ninput[..point_count].copy_from_slice(&scratch[..point_count]);').replace('let mut input = [ClipVertex::ZERO; MAX_CLIP_POINTS];',add('clip_zero_points','MAX_CLIP_POINTS')+'\nlet mut input = [ClipVertex::ZERO; MAX_CLIP_POINTS];').replace('let mut scratch = [ClipVertex::ZERO; MAX_CLIP_POINTS];',add('clip_zero_points','MAX_CLIP_POINTS')+'\nlet mut scratch = [ClipVertex::ZERO; MAX_CLIP_POINTS];'); p.write_text(t)
p=src/'triangle_manifold.rs'; t=p.read_text(); t=t.replace('core::mem::swap(&mut input, &mut output);',add('clip_copy_points','256')+'\ncore::mem::swap(&mut input, &mut output);').replace('let mut input = [ClipVertex::ZERO; 128];',add('clip_zero_points','128')+'\nlet mut input = [ClipVertex::ZERO; 128];').replace('let mut output = [ClipVertex::ZERO; 128];',add('clip_zero_points','128')+'\nlet mut output = [ClipVertex::ZERO; 128];'); t=t.replace('let mut final_count = 0;',add('clip_final','count')+'\nlet mut final_count = 0;').replace('let [p1, p2] = *s;',add('clip_input','2')+'\nlet [p1, p2] = *s;').replace('if count != 2 {',add('clip_output','count')+'\nif count != 2 {'); p.write_text(t)
p=src/'arena.rs'; t=p.read_text(); t=t.replace('fn write_manifold(m: &Manifold, pool: Col<f32>, base: usize) {','fn write_manifold(m: &Manifold, pool: Col<f32>, base: usize) {'+add('contact_points','m.point_count')); p.write_text(t)
p=root/'diagnostics/box3d-parity/scenes.ts'; t=p.read_text(); t=t.replace('import { DIR_STRIDE }', 'import { DIR_STRIDE, contactPointCount }'); needle='    if (i === colors) {';
t=t.replace(needle, '''    { const g = kernel(state.ecsState); g.bodySetActiveWorld(state.worldId);
      const points = Array.from({length:24}, (_, c) => {
        let total = 0;
        for (const scalar of [0,1]) {
          const count = g.graphContactCount(c,scalar), stride = scalar ? 2 : 1;
          const ids = new Uint32Array(g.memory.buffer,g.graphContactPtr(c,scalar),count*stride);
          for (let j=0;j<count;++j) {
            const id=ids[j*stride];
            total += contactPointCount(state.manifoldStore.dirU,state.manifoldStore.poolU,id,contactField(state,id,ContactField.manifoldCount));
          }
        }
        return total;
      });
      lines.push(`P ${i} ${points.join(" ")}`);
    }
'''+needle);
t=t.replace(needle, '    { const g = kernel(state.ecsState); g.bodySetActiveWorld(state.worldId); lines.push(`G ${i} ${Array.from({length:24}, (_, c) => g.graphContactCount(c,0) + g.graphContactCount(c,1) + g.jointArrayCount(c)).join(" ")}`); }\n'+needle); t=t.replace(needle,'    { const diagnostic = kernel(state.ecsState) as unknown as { diagnosticCount(i: number): number };\n      lines.push(`D ${i} ${'+str(labels).replace("'",'"')+'.map((name, k) => `${name} ${diagnostic.diagnosticCount(k)}`).join(" ")}`); }\n'+needle); p.write_text(t)
