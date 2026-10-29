// SPDX-License-Identifier: MIT
// Durations are host observations, not parity records. Bounds use the shim's zero draw origin.
void RecordTreeBounds(TreeSample::TreeBenchmark* s, FILE* file) {
    meshBounds.clear(); s->Render();
    fprintf(file, "[");
    for (size_t i=0;i<meshBounds.size();i++) {
        auto& b=meshBounds[i];
        fprintf(file, "%s[%.9g,%.9g,%.9g,%.9g,%.9g,%.9g]",i ? "," : "",b[0],b[1],b[2],b[3],b[4],b[5]);
    }
    fprintf(file, "]\n");
}
void RecordTree(TreeSample::TreeBenchmark* s, FILE* file) {
    fprintf(file, "[[%d,%d,%.9g,%d,%d,%d,%.9g,%.9g,%d,%d,%.9g,%.9g,%.9g,%d,%d,%d]",
        s->m_tree.proxyCount,s->m_height,s->m_areaRatio,s->m_fileIndex,s->m_testIndex,s->m_drawLevel,s->m_drawDistance,s->m_loadScale,
        s->m_timeStamp,s->m_haveClosest,s->m_closestPoint.x,s->m_closestPoint.y,s->m_closestPoint.z,s->m_doRay,s->m_doOverlap,s->m_doClosest);
    for (int id=0;id<s->m_tree.nodeCapacity;id++) {
        const b3TreeNode* n=s->m_tree.nodes+id;
        if (!(n->flags & b3_allocatedNode)) continue;
        bool leaf=(n->flags & b3_leafNode)!=0;
        fprintf(file, ",[%d,%d,%d,%d,%d,%d,%d,%d,%.9g,%.9g,%.9g,%.9g,%.9g,%.9g]",id,n->parent,
            leaf ? -1 : n->children.child1,leaf ? -1 : n->children.child2,s->m_depths[id],n->height,leaf,leaf ? (int)n->userData : -1,
            n->aabb.lowerBound.x,n->aabb.lowerBound.y,n->aabb.lowerBound.z,n->aabb.upperBound.x,n->aabb.upperBound.y,n->aabb.upperBound.z);
    }
    for (size_t id=0;id<s->m_proxies.size();id++) {
        auto& p=s->m_proxies.at(id);
        fprintf(file, ",[%d,%d,%d,%d]",(int)id,p.proxyId,p.queryTimeStamp,p.rayTimeStamp);
    }
    if (s->m_tree.proxyCount) {
        auto& r=s->m_rays[s->m_testIndex]; auto& b=s->m_overlapQueries[s->m_testIndex]; auto& q=s->m_closestPointQueries[s->m_testIndex];
        fprintf(file, ",[%.9g,%.9g,%.9g,%.9g,%.9g,%.9g,%.9g,%.9g,%.9g,%.9g,%.9g,%.9g,%.9g,%.9g,%.9g,%.9g]",
            r.origin.x,r.origin.y,r.origin.z,r.translation.x,r.translation.y,r.translation.z,b.lowerBound.x,b.lowerBound.y,b.lowerBound.z,
            b.upperBound.x,b.upperBound.y,b.upperBound.z,q.center.x,q.center.y,q.center.z,q.radius);
    }
    fprintf(file, "]\n");
}
