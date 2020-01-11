#version 460
#extension GL_NV_ray_tracing : require
#extension GL_GOOGLE_include_directive : require
#extension GL_EXT_nonuniform_qualifier : require

#include "common.glsl"

layout (location = 0) rayPayloadInNV RayPayload ray_payload;
layout (location = 1) rayPayloadInNV ShadowRayPayload shadow_ray_payload;

hitAttributeNV vec3 hit_attribs;

layout(set = 0, binding = 0) uniform accelerationStructureNV topLevelAS;

layout (set = 1, binding = 0) readonly buffer MaterialBuffer 
{
    uint id[];
} Material[];

layout (set = 1, binding = 1, std430) readonly buffer VertexBuffer 
{
    Vertex vertices[];
} VertexArray[];

layout (set = 1, binding = 2) readonly buffer IndexBuffer 
{
    uint indices[];
} IndexArray[];

layout(set = 2, binding = 0) uniform sampler2D s_Albedo[];

layout(set = 3, binding = 0) uniform sampler2D s_Normal[];

layout(set = 4, binding = 0) uniform sampler2D s_Roughness[];

layout(set = 5, binding = 0) uniform sampler2D s_Metallic[];

Vertex get_vertex(uint mesh_idx, uint vertex_idx)
{
    return VertexArray[nonuniformEXT(mesh_idx)].vertices[vertex_idx];
}

Triangle fetch_triangle(uint mesh_idx)
{
    Triangle tri;

    uvec3 idx = uvec3(IndexArray[nonuniformEXT(mesh_idx)].indices[3 * gl_PrimitiveID], 
                      IndexArray[nonuniformEXT(mesh_idx)].indices[3 * gl_PrimitiveID + 1],
                      IndexArray[nonuniformEXT(mesh_idx)].indices[3 * gl_PrimitiveID + 2]);

    tri.v0 = get_vertex(mesh_idx, idx.x);
    tri.v1 = get_vertex(mesh_idx, idx.y);
    tri.v2 = get_vertex(mesh_idx, idx.z);

    tri.mat_idx = Material[nonuniformEXT(mesh_idx)].id[uint(tri.v0.position.w)];

    return tri;
}

Vertex interpolated_vertex(in Triangle tri)
{
    const vec3 barycentrics = vec3(1.0 - hit_attribs.x - hit_attribs.y, hit_attribs.x, hit_attribs.y);

    Vertex o;

    o.position.xyz = tri.v0.position.xyz * barycentrics.x + tri.v1.position.xyz * barycentrics.y + tri.v2.position.xyz * barycentrics.z;
    o.tex_coord.xy = tri.v0.tex_coord.xy * barycentrics.x + tri.v1.tex_coord.xy * barycentrics.y + tri.v2.tex_coord.xy * barycentrics.z;
    o.normal.xyz = normalize(tri.v0.normal.xyz * barycentrics.x + tri.v1.normal.xyz * barycentrics.y + tri.v2.normal.xyz * barycentrics.z);
    o.tangent.xyz = normalize(tri.v0.tangent.xyz * barycentrics.x + tri.v1.tangent.xyz * barycentrics.y + tri.v2.tangent.xyz * barycentrics.z);
    o.bitangent.xyz = normalize(tri.v0.bitangent.xyz * barycentrics.x + tri.v1.bitangent.xyz * barycentrics.y + tri.v2.bitangent.xyz * barycentrics.z);

    return o;
}

const float kShadowRayBias = 0.1f;

float get_shadow()
{
    uint ray_flags = gl_RayFlagsTerminateOnFirstHitNV | gl_RayFlagsOpaqueNV;
    uint cull_mask = 0xff;
    float tmin = 0.0;
    float tmax = 10000.0;
    vec3 light_dir = normalize(vec3(0.5f, 0.9770f, 0.5000f));

    vec3 origin = gl_WorldRayOriginNV + gl_WorldRayDirectionNV * gl_HitTNV;

    // Add shadow ray offset
    origin += light_dir * kShadowRayBias;

	traceNV(topLevelAS, ray_flags, cull_mask, 1, 1, 1, origin, tmin, light_dir, tmax, 1);

    float visible = 1.0;

    if (shadow_ray_payload.dist > 0.0)
        visible = 0.0f;

    return visible;
}

void main()
{
    const Triangle tri = fetch_triangle(gl_InstanceCustomIndexNV);
    const Vertex v = interpolated_vertex(tri);

    vec3 color = textureLod(s_Albedo[nonuniformEXT(tri.mat_idx)], v.tex_coord.xy, 0.0).rgb;

    ray_payload.color_dist = vec4(max(get_shadow(), 0.1f) * color, gl_HitTNV);
}
