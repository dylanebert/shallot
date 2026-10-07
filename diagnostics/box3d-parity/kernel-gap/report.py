"""Regenerate tables from the adjacent evidence; no performance assertions."""
import json, pathlib, re
from statistics import median
p=pathlib.Path(__file__).parent
windows={'rain':(280,320),'junkyard':(180,200),'joint_grid':(20,40)}
fields=['step','pairs','collide','solve','solverSetup','constraints','prepareConstraints','integrateVelocities','warmStart','solveImpulses','integratePositions','relaxImpulses','applyRestitution','storeImpulses','splitIslands','transforms','sensorHits','jointEvents','hitEvents','refit','bullets','sleepIslands','sensors']
map_native={'hull::support_vertex':'support_scalar','hull::support_vertex_wide':'support_wide','hull::support_face':'support_face','triangle_manifold::triangle_support':'triangle_support','manifold::clip_polygon':'clip_polygon','manifold::compute_separating_axis':'compute_separating_axis','manifold::collide_hulls':'collide_hulls','manifold::build_face_a_contact':'build_face_a_contact','arena::contact_block':'update_contact','joint::prepare_world':'prepare_joint','joint::warm_start':'warm_joint','joint::solve':'solve_joint','tree::query':'tree_query','tree::rebuild':'tree_rebuild','tree::partition_mid':'tree_partition','tree::enlarge_proxy':'tree_enlarge','pairwork::record_child':'pair_callback','contact::solve':'solve_mesh','contact_wide::solve':'solve_convex','continuous::solve':'continuous','stages::execute_block':'execute_block'}
count_keys={'hull::support_vertex':'support_scalar','hull::support_vertex_wide':'support_wide','hull::support_face':'support_face','triangle_manifold::triangle_support':'triangle_support','manifold::collide_hulls':'hull_sat'}
def name(s):
    if '_15shallot_physics' not in s:return s
    rest=s.split('_15shallot_physics',1)[1]; parts=[]
    while rest and rest[0].isdigit():
        m=re.match(r'(\d+)',rest); n=int(m[1]); rest=rest[m.end():]; parts.append(rest[:n]); rest=rest[n:]
    if parts==['hull']:
        for member in ['support_vertex_wide','support_vertex','support_face']:
            if member in rest:return 'hull::'+member
    if parts==['pairwork'] and 'record_child' in rest:return 'pairwork::record_child'
    return '::'.join(parts) if parts else s

def rows(text,prefix,fr):return [l.split()[2:] for l in text.splitlines() if l.startswith(prefix+' ') and int(l.split()[1])>=fr]
def named(text,prefix,fr,scale=1):
    rs=rows(text,prefix,fr)
    return {rs[0][k]:median(float(r[k+1])*scale for r in rs) for k in range(0,len(rs[0]),2)} if rs else {}
def samples(text,steps):
    own={}; inclusive={}
    for line in text.splitlines():
        if line.startswith(('KS ','KI ')):
            tag,ms,fn=line.split(' ',2); dest=own if tag=='KS' else inclusive; key=name(fn); dest[key]=dest.get(key,0)+float(ms)
        if not line.startswith('WP '):continue
        _,worker,raw=line.split(' ',2); prof=json.loads(raw)
        nodes={n['id']:n for n in prof['nodes']}; parents={c:n['id'] for n in prof['nodes'] for c in n.get('children',[])}
        for sample,us in zip(prof['samples'],prof['timeDeltas']):
            frame=nodes[sample]['callFrame']
            if not frame['url'].startswith('wasm') or frame['functionName'].startswith('js-to-wasm'):continue
            ms=us/1000/steps; key=name(frame['functionName']); own[key]=own.get(key,0)+ms
            seen=set(); at=sample
            while at is not None:
                frame=nodes[at]['callFrame']
                if frame['url'].startswith('wasm') and not frame['functionName'].startswith('js-to-wasm'):
                    key=name(frame['functionName'])
                    if key not in seen:inclusive[key]=inclusive.get(key,0)+ms; seen.add(key)
                at=parents.get(at)
    return own,inclusive

def work(key):
    if key.startswith('manifold::build_face') or key=='manifold::clip_polygon':return 'clipped vertices; extra caller copies/initialization'
    if key in count_keys:return count_keys[key]
    if key.startswith('joint::'):return 'joint/color occupancy'
    if key.startswith(('contact::','contact_wide::')):return 'emitted points + contact/color occupancy'
    if key in ['arena::contact_block','manifold::collide_hulls']:return 'awake contacts, SAT/support/axes/points; extra clipping copies'
    if key.startswith(('tree::','pairwork::')):return 'tree_queries/tree_nodes/tree_leaves/rebuild_leaves' 
    return 'enclosing phase; no independent operation counter'

def classification(key):
    if key.startswith('manifold::build_face') or key in ['arena::contact_block','manifold::collide_hulls']:return 'more work in descendant clipping; residual same-work cost not isolated'
    if key.startswith(('hull::','manifold::','triangle_manifold::')):return 'same geometric work; native timer perturbation bounds helper rate'
    if key.startswith(('joint::','contact::','contact_wide::')):return 'same counted constraint work; narrowest routine shown'
    if key.startswith(('tree::','pairwork::')):return 'query/rebuild counts distinguish work; code-generation/layout residual'
    return 'no independent material gap established; narrowest sampled routine'

out=['# Measured tables','', 'Generated by `python3 diagnostics/box3d-parity/kernel-gap/report.py`. Read `ledger.md` for boundaries and limitations.','']
data={}
for scene,(fr,to) in windows.items():
    out+=['## '+scene,'']
    data[scene]={}
    for th in [1,4]:
        texts={side:(p/f'{scene}-{th}-{side}.txt').read_text() for side in ['native','kernel']}
        counted={side:(p/f'{scene}-{th}-counts-{side}.txt').read_text() for side in ['native','kernel']}
        timed=(p/f'{scene}-{th}-times-native.txt').read_text()
        phases={side:[list(map(float,r)) for r in rows(text,'F',fr)] for side,text in texts.items()}
        phase={side:{field:median(r[k] for r in rs) for k,field in enumerate(fields)} for side,rs in phases.items()}
        counts={side:named(text,'D',fr) for side,text in counted.items()}
        for prefix in ['D','G','P']:
            a=rows(counted['native'],prefix,fr); b=rows(counted['kernel'],prefix,fr)
            if prefix in ['G','P']:assert a==b
            else:
                for ra,rb in zip(a,b):
                    da=dict(zip(ra[::2],ra[1::2])); db=dict(zip(rb[::2],rb[1::2]))
                    assert all(da[k]==db[k] for k in da if k not in ['clip_copy_points','clip_zero_points','tree_queries','tree_nodes','tree_leaves','rebuild_leaves'])
        ntime=named(timed,'H',fr,1e-6); nself=named(timed,'HS',fr,1e-6); calls=named(timed,'Q',fr)
        self,inclusive=samples(texts['kernel'],to-fr)
        colors=rows(counted['kernel'],'G',fr); color_medians=[median(int(r[c]) for r in colors) for c in range(24)]
        out += [f'### {th} thread(s)','', '| field / substage | native ms | kernel ms | kernel − native ms |','|---|---:|---:|---:|']
        for field in fields:
            n,s=phase['native'][field],phase['kernel'][field]; out.append(f'| {field} | {n:.4f} | {s:.4f} | {s-n:+.4f} |')
        out+=['','| counter (median per step) | native | kernel |','|---|---:|---:|']
        for key,n in counts['native'].items():out.append(f'| {key} | {n:g} | {counts["kernel"][key]:g} |')
        point_rows=rows(counted['kernel'],'P',fr)
        point_medians=[median(int(r[c]) for r in point_rows) for c in range(24)]
        point_total=median(sum(map(int,r)) for r in point_rows)
        out+=['', f'Post-step graph-resident manifold points: {point_total:g} median total, equal per color at each measured step. Per-color medians: `'+', '.join(f'{v:g}' for v in point_medians)+'`.']
        out+=['', '24 color occupancies (contacts + joints), equal at **each** measured step: `'+', '.join(f'{v:g}' for v in color_medians)+'`.','', next(l for l in timed.splitlines() if l.startswith('B ')), '', '| symbol / routine | kernel self CPU ms | kernel inclusive CPU ms | native named inclusive elapsed ms | native calls | counted work / narrowest point |','|---|---:|---:|---:|---:|---|']
        functions={}
        for key,s in sorted(self.items(),key=lambda kv:-kv[1]):
            native_key=map_native.get(key); n=ntime.get(native_key); call=calls.get(native_key)
            nstr=f'{n:.4f}' if n is not None else 'not isolated; see phases'
            callstr=f'{call:g}' if call is not None else '—'
            if key in count_keys:callstr+=f'; kernel {counts["kernel"][count_keys[key]]:g}'
            out.append(f'| {key} | {s:.4f} | {inclusive.get(key,0):.4f} | {nstr} | {callstr} | {work(key)}; {classification(key)} |')
            functions[key]={'kernel_self_cpu_ms':s,'kernel_inclusive_cpu_ms':inclusive.get(key,0),'native_timer':native_key,'native_inclusive_ms':n,'native_self_ms':nself.get(native_key),'native_calls':call,'work':work(key),'classification':classification(key)}
        out+=['', 'Named native timers not necessarily emitted as separate WASM functions (inlining and boundaries differ):','', '| native timer | inclusive elapsed ms | self elapsed ms | calls |','|---|---:|---:|---:|---:|']
        for key,n in ntime.items():out.append(f'| {key} | {n:.4f} | {nself[key]:.4f} | {calls[key]:g} |')
        out+=['']
        data[scene][str(th)]={'phases':phase,'counts':counts,'colors':color_medians,'resident_points':point_medians,'resident_points_total':point_total,'functions':functions,'native_timers':ntime,'native_calls':calls}
(p/'tables.md').write_text('\n'.join(out)+'\n')
(p/'measurements.json').write_text(json.dumps(data,indent=2)+'\n')
