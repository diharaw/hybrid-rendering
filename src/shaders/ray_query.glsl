#ifndef RAY_QUERY_GLSL
#define RAY_QUERY_GLSL

// ------------------------------------------------------------------------

float query_visibility(vec3 world_pos, vec3 direction, float t_max, uint ray_flags)
{
    float t_min = 0.01f;
 
    // Initializes a ray query object but does not start traversal
    rayQueryEXT ray_query;

    rayQueryInitializeEXT(ray_query,
                          u_TopLevelAS,
                          ray_flags,
                          0xFF,
                          world_pos,
                          t_min,
                          direction,
                          t_max);

    // Start traversal: return false if traversal is complete
    while (rayQueryProceedEXT(ray_query)) {}

    // Returns type of committed (true) intersection
    if (rayQueryGetIntersectionTypeEXT(ray_query, true) != gl_RayQueryCommittedIntersectionNoneEXT)
        return 0.0f;

    return 1.0f;
}

// ------------------------------------------------------------------------

float query_distance(vec3 world_pos, vec3 direction, float t_max)
{
    float t_min     = 0.01f;
    uint  ray_flags = gl_RayFlagsOpaqueEXT | gl_RayFlagsTerminateOnFirstHitEXT;

    // Initializes a ray query object but does not start traversal
    rayQueryEXT ray_query;

    rayQueryInitializeEXT(ray_query,
                          u_TopLevelAS,
                          ray_flags,
                          0xFF,
                          world_pos,
                          t_min,
                          direction,
                          t_max);

    // Start traversal: return false if traversal is complete
    while (rayQueryProceedEXT(ray_query)) {}

    // Returns type of committed (true) intersection
    if (rayQueryGetIntersectionTypeEXT(ray_query, true) != gl_RayQueryCommittedIntersectionNoneEXT)
        return rayQueryGetIntersectionTEXT(ray_query, true) < t_max ? 0.0f : 1.0f;

    return 1.0f;
}

// ------------------------------------------------------------------------

#endif