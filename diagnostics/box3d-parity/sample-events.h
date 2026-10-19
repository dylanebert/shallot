// SPDX-License-Identifier: MIT
struct RecordedShape { b3ShapeId id; int body; };
std::vector<b3BodyId> recordedBodies;
std::vector<RecordedShape> recordedShapes;
std::vector<b3JointId> recordedJoints;
bool recordEvents = false;
int BodyTag( b3BodyId id ) {
    for ( int i = (int)recordedBodies.size() - 1; i >= 0; --i )
        if ( B3_ID_EQUALS( recordedBodies[i], id ) ) return i + 1;
    return 0;
}
int ShapeTag( b3ShapeId id ) {
    for ( int i = (int)recordedShapes.size() - 1; i >= 0; --i )
        if ( B3_ID_EQUALS( recordedShapes[i].id, id ) ) return i + 1;
    return 0;
}
int JointTag( b3JointId id ) {
    for ( int i = (int)recordedJoints.size() - 1; i >= 0; --i )
        if ( B3_ID_EQUALS( recordedJoints[i], id ) ) return i + 1;
    return 0;
}
b3BodyId RecordBody( b3BodyId id ) { recordedBodies.push_back( id ); return id; }
b3ShapeId RecordShape( b3ShapeId id ) { recordedShapes.push_back( { id, BodyTag( b3Shape_GetBody( id ) ) } ); return id; }
b3JointId RecordJoint( b3JointId id ) { recordedJoints.push_back( id ); return id; }
void PrintPair( b3ShapeId a, b3ShapeId b ) {
    int ia = ShapeTag( a ), ib = ShapeTag( b );
    assert( ia && ib );
    printf( "%d,%d,%d,%d", ia, recordedShapes[ia - 1].body, ib, recordedShapes[ib - 1].body );
}
void PrintEvents( b3WorldId world ) {
    b3SensorEvents sensors = b3World_GetSensorEvents( world );
    b3ContactEvents contacts = b3World_GetContactEvents( world );
    b3JointEvents joints = b3World_GetJointEvents( world );
    if ( sensors.beginCount + sensors.endCount + contacts.beginCount + contacts.endCount + contacts.hitCount + joints.count == 0 ) return;
    printf( " [[" );
    for ( int i = 0; i < sensors.beginCount; ++i ) {
        if ( i ) printf( "," );
        printf( "[" ); PrintPair( sensors.beginEvents[i].sensorShapeId, sensors.beginEvents[i].visitorShapeId ); printf( "]" );
    }
    printf( "],[" );
    for ( int i = 0; i < sensors.endCount; ++i ) {
        if ( i ) printf( "," );
        printf( "[" ); PrintPair( sensors.endEvents[i].sensorShapeId, sensors.endEvents[i].visitorShapeId ); printf( "]" );
    }
    printf( "],[" );
    for ( int i = 0; i < contacts.beginCount; ++i ) {
        if ( i ) printf( "," );
        printf( "[" ); PrintPair( contacts.beginEvents[i].shapeIdA, contacts.beginEvents[i].shapeIdB ); printf( "]" );
    }
    printf( "],[" );
    for ( int i = 0; i < contacts.endCount; ++i ) {
        if ( i ) printf( "," );
        printf( "[" ); PrintPair( contacts.endEvents[i].shapeIdA, contacts.endEvents[i].shapeIdB ); printf( "]" );
    }
    printf( "],[" );
    for ( int i = 0; i < contacts.hitCount; ++i ) {
        if ( i ) printf( "," );
        b3ContactHitEvent e = contacts.hitEvents[i];
        printf( "[" ); PrintPair( e.shapeIdA, e.shapeIdB );
        printf( ",%.17g,%.17g,%.17g,%.17g,%.17g,%.17g,%.17g,\"%llu\",\"%llu\"]",
            e.point.x, e.point.y, e.point.z, e.normal.x, e.normal.y, e.normal.z, e.approachSpeed,
            (unsigned long long)e.userMaterialIdA, (unsigned long long)e.userMaterialIdB );
    }
    printf( "],[" );
    for ( int i = 0; i < joints.count; ++i ) { if ( i ) printf( "," ); printf( "[%d]", JointTag( joints.jointEvents[i].jointId ) ); }
    printf( "]]" );
}
#define b3CreateBody(...) RecordBody(b3CreateBody(__VA_ARGS__))
#define b3CreateHullShape(...) RecordShape(b3CreateHullShape(__VA_ARGS__))
#define b3CreateTransformedHullShape(...) RecordShape(b3CreateTransformedHullShape(__VA_ARGS__))
#define b3CreateSphereShape(...) RecordShape(b3CreateSphereShape(__VA_ARGS__))
#define b3CreateCapsuleShape(...) RecordShape(b3CreateCapsuleShape(__VA_ARGS__))
#define b3CreateMeshShape(...) RecordShape(b3CreateMeshShape(__VA_ARGS__))
#define b3CreateDistanceJoint(...) RecordJoint(b3CreateDistanceJoint(__VA_ARGS__))
#define b3CreateMotorJoint(...) RecordJoint(b3CreateMotorJoint(__VA_ARGS__))
#define b3CreatePrismaticJoint(...) RecordJoint(b3CreatePrismaticJoint(__VA_ARGS__))
#define b3CreateRevoluteJoint(...) RecordJoint(b3CreateRevoluteJoint(__VA_ARGS__))
#define b3CreateWeldJoint(...) RecordJoint(b3CreateWeldJoint(__VA_ARGS__))
#define b3CreateWheelJoint(...) RecordJoint(b3CreateWheelJoint(__VA_ARGS__))
