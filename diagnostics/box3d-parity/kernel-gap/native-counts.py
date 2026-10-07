"""Instrument an isolated Box3D archive and the diagnostic harness."""
import pathlib,re,sys
root=pathlib.Path(sys.argv[1]); harness=pathlib.Path(sys.argv[2])
labels=['support_scalar','support_wide','support_face','triangle_support','clip_input','clip_output','clip_final','hull_sat','hull_sat_hit','hull_sat_miss','face_axes','edge_candidates','contact_points','clip_copy_points','clip_zero_points','tree_queries','tree_nodes','tree_leaves','rebuild_leaves']
timers=['support_scalar','support_wide','support_face','triangle_support','clip_polygon','compute_separating_axis','collide_hulls','build_face_a_contact','update_contact','prepare_joint','warm_joint','solve_joint','tree_query','tree_rebuild','tree_partition','tree_enlarge','pair_callback','solve_mesh','solve_convex','continuous','finalize','execute_block']
header=(pathlib.Path(__file__).parent/'native-clock.h').read_text()
(root/'src/diagnostic.h').write_text(header)
def patch(file, callback):
    p=root/'src'/file; t=p.read_text(); p.write_text('#include "diagnostic.h"\n'+callback(t))
def entry(t,name,code):
    m=re.search(r'\b'+name+r'\s*\([^;]*?\)\s*\{',t,re.S)
    if not m: raise Exception(name)
    return t[:m.end()]+code+t[m.end():]
def body(t,name,change):
    m=re.search(r'\b'+name+r'\s*\([^;]*?\)\s*\{',t,re.S)
    if not m: raise Exception(name)
    start=m.end(); end=start; depth=1
    while depth:
        if t[end]=='{': depth+=1
        if t[end]=='}': depth-=1
        end+=1
    return t[:start]+change(t[start:end-1])+t[end-1:]
def dc(label,n='1'): return '\nDC(%d,%s);\n'%(labels.index(label),n)
patch('hull.c',lambda t:entry(entry(t,'b3FindHullSupportVertex',dc('support_scalar')),'b3FindHullSupportFace',dc('support_face')))
patch('convex_manifold.c',lambda t:entry(t,'b3GetSupportWide',dc('support_wide')).replace('for ( int i = 0; i < faceCountA; ++i )\n\t{','for ( int i = 0; i < faceCountA; ++i )\n\t{'+dc('face_axes')).replace('for ( int i = 0; i < faceCountB; ++i )\n\t{','for ( int i = 0; i < faceCountB; ++i )\n\t{'+dc('face_axes')).replace('for ( int i = 0; i < na; i += 4 )\n\t\t{','for ( int i = 0; i < na; i += 4 )\n\t\t{'+dc('edge_candidates','b3MinInt(4, na-i)')).replace('pointCount = b3MinInt( pointCount, B3_MAX_CLIP_POINTS );',dc('clip_final','pointCount')+'pointCount = b3MinInt( pointCount, B3_MAX_CLIP_POINTS );'))
patch('convex_manifold.c',lambda t:body(entry(t,'b3ClipSegment',dc('clip_input','2')),'b3ClipSegment',lambda b:b.replace('return vertexCount;',dc('clip_output','vertexCount')+'return vertexCount;')))
p=root/'src/convex_manifold.c'; t=p.read_text(); p.write_text(body(t,'b3ClipSegmentToHullFace',lambda b:b.replace('return 2;',dc('clip_final','2')+'return 2;')))
patch('triangle_manifold.c',lambda t:entry(t,'b3GetTriangleSupport',dc('triangle_support')).replace('int vertexCount = 0;',dc('clip_input','2')+'int vertexCount = 0;').replace('if ( vertexCount != 2 )',dc('clip_output','vertexCount')+'if ( vertexCount != 2 )').replace('return true;\n}\n\nstatic b3SeparatingAxis b3QueryTriangleFaceAndCapsule',dc('clip_final','2')+'return true;\n}\n\nstatic b3SeparatingAxis b3QueryTriangleFaceAndCapsule').replace('pointCount = b3MinInt( pointCount, pointCapacity );',dc('clip_final','pointCount')+'pointCount = b3MinInt( pointCount, pointCapacity );'))
patch('manifold.c',lambda t:entry(t,'b3ClipPolygon',dc('clip_input','count')).replace('return outCount;',dc('clip_output','outCount')+'return outCount;'))
patch('contact.c',lambda t:t.replace('world->taskContexts.data[workerIndex].satCallCount += 1;',dc('hull_sat')+dc('hull_sat_hit','cache->satCache.hit')+dc('hull_sat_miss','!cache->satCache.hit')+'world->taskContexts.data[workerIndex].satCallCount += 1;').replace('if ( geomManifold.pointCount == 0 )',dc('contact_points','geomManifold.pointCount')+'if ( geomManifold.pointCount == 0 )'))
patch('dynamic_tree.c',lambda t:body(body(entry(t,'b3DynamicTree_Query',dc('tree_queries')),'b3DynamicTree_Query',lambda b:b.replace('return result;',dc('tree_nodes','result.nodeVisits')+dc('tree_leaves','result.leafVisits')+'return result;')),'b3DynamicTree_Rebuild',lambda b:b.replace('return leafCount;',dc('rebuild_leaves','leafCount')+'return leafCount;')))
patch('mesh_contact.c',lambda t:t.replace('manifold->pointCount = pointCount;', dc('contact_points','pointCount')+'manifold->pointCount = pointCount;'))
for file, pairs in [('hull.c', [('b3FindHullSupportVertex',0),('b3FindHullSupportFace',2)]),('convex_manifold.c',[('b3GetSupportWide',1),('b3ComputeSeparatingAxis',5),('b3CollideHulls',6),('b3BuildFaceAContact',7)]),('manifold.c',[('b3ClipPolygon',4)]),('triangle_manifold.c',[('b3GetTriangleSupport',3)]),('contact.c',[('b3UpdateContact',8)]),('joint.c',[('b3PrepareJoint',9),('b3WarmStartJoint',10),('b3SolveJoint',11)]),('dynamic_tree.c',[('b3DynamicTree_Query',12),('b3DynamicTree_Rebuild',13),('b3PartitionMid',14),('b3DynamicTree_EnlargeProxy',15)]),('broad_phase.c',[('b3PairQueryCallback',16)]),('contact_solver.c',[('b3SolveContacts_Mesh',17),('b3SolveContacts_Convex',18)]),('solver.c',[('b3SolveContinuous',19),('b3FinalizeBodiesTask',20),('b3ExecuteBlock',21)])]:
    p=root/'src'/file; t=p.read_text()
    if '#include "diagnostic.h"' not in t: t='#include "diagnostic.h"\n'+t
    for name,index in pairs: t=entry(t,name,'DT(%d)'%index)
    p.write_text(t)
t=harness.read_text(); t='''#include "diagnostic.h"
int diagnostic_timing;
unsigned long long diagnostic_bias;
_Atomic unsigned diagnostic_counts[19];
unsigned long long diagnostic_times[22][16], diagnostic_self[22][16];
unsigned diagnostic_calls[22][16];
_Atomic unsigned diagnostic_next_thread;
_Thread_local DiagnosticTimer* diagnostic_top;
_Thread_local unsigned diagnostic_thread = UINT_MAX;
_Thread_local unsigned long long diagnostic_overhead;
'''+t
t=t.replace('b3WorldDef worldDef = b3DefaultWorldDef();', '''diagnostic_timing = getenv("TIMERS") != NULL;
if (diagnostic_timing) {
    unsigned long long batches[31];
    for(int batch=0;batch<31;++batch) {
        diagnostic_times[0][0]=0;
        for(int k=0;k<1000;++k) { DT(0) __asm__ volatile("" ::: "memory"); }
        batches[batch]=diagnostic_times[0][0]/1000;
    }
    for(int a=0;a<31;++a) for(int b=a+1;b<31;++b) if(batches[b]<batches[a]) { unsigned long long x=batches[a]; batches[a]=batches[b]; batches[b]=x; }
    diagnostic_bias=batches[15];
    printf("B timerBiasNs %llu\\n",diagnostic_bias);
    for(int k=0;k<22;++k) { diagnostic_times[k][0]=diagnostic_self[k][0]=diagnostic_calls[k][0]=0; }
}
b3WorldDef worldDef = b3DefaultWorldDef();''')
printing='printf("D %d", i);\n'
for k,label in enumerate(labels): printing+='printf(" '+label+' %u", atomic_exchange_explicit(&diagnostic_counts['+str(k)+'],0,memory_order_relaxed));\n'
printing+='printf("\\n");\n'
printing+='b3Counters dc = b3World_GetCounters(worldId); printf("G %d",i); for(int c=0;c<24;++c) printf(" %d",dc.colorCounts[c]); printf("\\n");\n'
printing+='''printf("P %d",i);
for(int c=0;c<24;++c) {
    b3GraphColor* color=world->constraintGraph.colors+c;
    int points=0;
    for(int scalar=0;scalar<2;++scalar) {
        int count=scalar ? color->contacts.count : color->convexContacts.count;
        for(int j=0;j<count;++j) {
            int id=scalar ? color->contacts.data[j].contactId : color->convexContacts.data[j];
            b3Contact* contact=world->contacts.data+id;
            for(int m=0;m<contact->manifoldCount;++m) points+=contact->manifolds[m].pointCount;
        }
    }
    printf(" %d",points);
}
printf("\\n");
'''
for prefix,array in [('H','diagnostic_times'),('HS','diagnostic_self'),('Q','diagnostic_calls')]:
    printing+='printf("'+prefix+' %d",i);\n'
    for k,label in enumerate(timers): printing+=' { unsigned long long sum=0; for(unsigned thread=0;thread<16;++thread) {sum+='+array+'['+str(k)+'][thread]; '+array+'['+str(k)+'][thread]=0;} printf(" '+label+' %llu",sum); }\n'
    printing+='printf("\\n");\n'
t=t.replace('if ( profileFrom >= 0 && i >= profileFrom )',printing+'if ( profileFrom >= 0 && i >= profileFrom )')
harness.write_text(t)
