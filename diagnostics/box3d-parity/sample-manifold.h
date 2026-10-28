FILE* manifoldRecords = nullptr;
void ManifoldVector(std::vector<double>& row, b3Vec3 v) { row.insert(row.end(), {v.x,v.y,v.z}); }
void ManifoldPose(std::vector<std::vector<double>>& rows, b3WorldTransform t) { rows.push_back({0,t.p.x,t.p.y,t.p.z,t.q.v.x,t.q.v.y,t.q.v.z,t.q.s}); }
void ManifoldSphere(std::vector<std::vector<double>>& rows, b3Sphere s) { rows.push_back({1,s.center.x,s.center.y,s.center.z,s.radius}); }
void ManifoldCapsule(std::vector<std::vector<double>>& rows, b3Capsule c) { rows.push_back({2,c.center1.x,c.center1.y,c.center1.z,c.center2.x,c.center2.y,c.center2.z,c.radius}); }
void ManifoldHull(std::vector<std::vector<double>>& rows, const b3HullData* h, int role) {
    rows.push_back({double(role)});
    for(int i=0;i<h->vertexCount;i++) ManifoldVector(rows.back(), b3GetHullPoints(h)[i]);
}
template<class T> void ManifoldResult(std::vector<std::vector<double>>& rows, T* s) {
    ManifoldPose(rows,s->m_transformA); ManifoldPose(rows,s->m_transformB);
    const auto& m=s->m_manifold;
    // A separated call does not initialize the unused normal or contact feature.
    rows.push_back(m.pointCount ? std::vector<double>{6,double(m.pointCount),m.normal.x,m.normal.y,m.normal.z,double(m.feature),m.squaredDistance} : std::vector<double>{6,0,0,0,0,0,0});
    for(int i=0;i<m.pointCount;i++) {
        const auto& p=m.points[i];
        auto& row=rows.back();
        row.insert(row.end(),{p.point.x,p.point.y,p.point.z,p.separation,double(p.pair.owner1),double(p.pair.index1),double(p.pair.owner2),double(p.pair.index2),double(p.triangleIndex)});
    }
    const auto& c=s->m_simplexCache; const auto& sat=s->m_satCache;
    rows.push_back({7,c.metric,double(c.count)});
    for(int i=0;i<4;i++)rows.back().push_back(c.indexA[i]);
    for(int i=0;i<4;i++)rows.back().push_back(c.indexB[i]);
    rows.back().insert(rows.back().end(),{sat.separation,double(sat.type),double(sat.indexA),double(sat.indexB),double(sat.hit)});
}
void RecordManifold(Sample* sample) {
    using namespace ManifoldSamples;
    std::vector<std::vector<double>> rows;
    if(auto* s=dynamic_cast<Manifold*>(sample))ManifoldResult(rows,s);
    else if(auto* s=dynamic_cast<TriangleManifold*>(sample)) {
        ManifoldResult(rows,s);rows.push_back({5});
        for(const auto& p:s->m_triangle)ManifoldVector(rows.back(),p);
    } else return;
    if(auto* s=dynamic_cast<SphereAndSphere*>(sample))ManifoldSphere(rows,s->m_sphere);
    if(auto* s=dynamic_cast<CapsuleAndSphere*>(sample)){ManifoldCapsule(rows,s->m_capsule);ManifoldSphere(rows,s->m_sphere);}
    if(auto* s=dynamic_cast<HullAndSphere*>(sample)){ManifoldHull(rows,&s->m_hull.base,3);ManifoldSphere(rows,s->m_sphere);}
    if(auto* s=dynamic_cast<TriangleAndSphere*>(sample))ManifoldSphere(rows,s->m_sphere);
    if(auto* s=dynamic_cast<CapsuleAndCapsule*>(sample))ManifoldCapsule(rows,s->m_capsule);
    if(auto* s=dynamic_cast<CapsuleAndHull*>(sample)){ManifoldHull(rows,&s->m_hull.base,3);ManifoldCapsule(rows,s->m_capsule);}
    if(auto* s=dynamic_cast<TriangleAndCapsule*>(sample))ManifoldCapsule(rows,s->m_capsule);
    if(auto* s=dynamic_cast<HullAndHull*>(sample)){ManifoldHull(rows,s->m_hullA,3);ManifoldHull(rows,s->m_hullB,4);}
    if(auto* s=dynamic_cast<TriangleAndHull*>(sample))ManifoldHull(rows,s->m_hull,4);
    for(const auto& mark:characterMarks) if(mark[0]==0 || mark[0]==1) {
        rows.push_back(mark);rows.back()[0] = mark[0]==1 ? 8 : 9;
    }
    fprintf(manifoldRecords,"[");
    for(size_t j=0;j<rows.size();j++) {
        fprintf(manifoldRecords,"%s[",j?",":"");
        for(size_t k=0;k<rows[j].size();k++)fprintf(manifoldRecords,"%s%.9g",k?",":"",rows[j][k]);
        fprintf(manifoldRecords,"]");
    }
    fprintf(manifoldRecords,"]\n");
}
