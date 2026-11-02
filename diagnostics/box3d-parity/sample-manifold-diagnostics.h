FILE* manifoldDiagnosticRecords = nullptr;
void DiagnosticHull(std::vector<std::vector<double>>& rows,const b3HullData* h,int role) {
    rows.push_back({20,double(role),double(h->vertexCount),double(h->edgeCount),double(h->faceCount),h->surfaceArea,h->volume,h->innerRadius});
    auto& row=rows.back(); ManifoldVector(row,h->center);ManifoldVector(row,h->aabb.lowerBound);ManifoldVector(row,h->aabb.upperBound);ManifoldVector(row,h->centralInertia.cx);ManifoldVector(row,h->centralInertia.cy);ManifoldVector(row,h->centralInertia.cz);
    for(int i=0;i<4;i++)row.push_back((h->hash>>(16*i))&0xffff);
    const auto* points=b3GetHullPoints(h);const auto* vertices=b3GetHullVertices(h);const auto* edges=b3GetHullEdges(h);const auto* faces=b3GetHullFaces(h);const auto* planes=b3GetHullPlanes(h);const auto* cosines=b3GetHullEdgeCosines(h);
    for(int i=0;i<h->vertexCount;i++)rows.push_back({21,double(role),double(i),points[i].x,points[i].y,points[i].z,double(vertices[i].edge)});
    for(int i=0;i<h->edgeCount;i++)rows.push_back({22,double(role),double(i),double(edges[i].next),double(edges[i].twin),double(edges[i].origin),double(edges[i].face)});
    for(int i=0;i<h->faceCount;i++)rows.push_back({23,double(role),double(i),double(faces[i].edge),planes[i].normal.x,planes[i].normal.y,planes[i].normal.z,planes[i].offset});
    rows.push_back({24,double(role)});for(int i=0;i<h->edgeCount/2;i++)rows.back().push_back(cosines[i]);
}
void RecordManifoldDiagnostics(Sample* sample) {
    using namespace ManifoldSamples;
    auto* base=static_cast<Manifold*>(sample);
    std::vector<std::vector<double>> rows={{-1,double(base->m_useCache),double(base->m_manualFeature),double(base->m_drawPoints)}};
    if(auto* s=dynamic_cast<HullCulling*>(sample)) {
        DiagnosticHull(rows,s->m_hullA,0);DiagnosticHull(rows,s->m_hullB,1);
        rows.push_back({30,double(s->m_useCylinders),s->m_cylinderHeight,s->m_cylinderRadius,double(s->m_cylinderSides),double(s->m_showSpheres),double(s->m_showCulled),double(s->m_showNormals),double(s->m_edgeBound),double(s->m_seedA),double(s->m_seedB),double(s->m_probeVertexA),double(s->m_probeVertexB),double(s->m_keptEdgeCountA),double(s->m_keptEdgeCountB),double(s->m_sphereKeptEdgeCountA),double(s->m_sphereKeptEdgeCountB),s->m_separationA,s->m_separationB,s->m_gap,double(s->m_separatedFeature),double(s->m_probeApplied)});
        auto states=[&](int tag,const uint8_t* data,int count){rows.push_back({double(tag)});for(int i=0;i<count;i++)rows.back().push_back(data[i]);};
        states(31,s->m_faceStateA,s->m_hullA->faceCount);states(32,s->m_faceStateB,s->m_hullB->faceCount);states(33,s->m_edgeStateA,s->m_hullA->edgeCount/2);states(34,s->m_edgeStateB,s->m_hullB->edgeCount/2);
    } else if(auto* s=dynamic_cast<EdgeAxisWedge*>(sample)) {
        DiagnosticHull(rows,s->m_hullA,0);DiagnosticHull(rows,s->m_hullB,1);
        rows.push_back({40,double(s->m_hullType),double(s->m_seed&0xffff),double(s->m_seed>>16),double(s->m_searchCount),double(s->m_searchFound),double(s->m_edgeWins),s->m_faceSeparationA,s->m_faceSeparationB,s->m_radius});
        ManifoldVector(rows.back(),{float(s->m_gaussCenter.x),float(s->m_gaussCenter.y),float(s->m_gaussCenter.z)});rows.back().push_back(s->m_gaussRadius);ManifoldVector(rows.back(),s->m_centerOffset);ManifoldVector(rows.back(),s->m_edge.normal);rows.back().insert(rows.back().end(),{s->m_edge.separation,double(s->m_edge.edgeA),double(s->m_edge.edgeB)});
        auto arc=[&](int tag,const EdgeAxisWedge::Arc& a){rows.push_back({double(tag)});ManifoldVector(rows.back(),a.u);ManifoldVector(rows.back(),a.v);ManifoldVector(rows.back(),a.peak);rows.back().insert(rows.back().end(),{a.a,a.b,a.c,a.separationU,a.separationV,a.outsideDegrees,double(a.inside)});};
        arc(41,s->m_arcA);arc(42,s->m_arcB);
    } else assert(false);
    ManifoldWriteRows(manifoldDiagnosticRecords,rows);
}
