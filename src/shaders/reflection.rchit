#version 460

#extension GL_EXT_ray_tracing : require
#extension GL_GOOGLE_include_directive : require
#extension GL_EXT_nonuniform_qualifier : require

#define RAY_TRACING
#include "common.glsl"
#include "scene_descriptor_set.glsl"

// ------------------------------------------------------------------------
// PAYLOADS ---------------------------------------------------------------
// ------------------------------------------------------------------------

layout(location = 0) rayPayloadInEXT RayPayload ray_payload;

// ------------------------------------------------------------------------
// HIT ATTRIBUTE ----------------------------------------------------------
// ------------------------------------------------------------------------

hitAttributeEXT vec2 hit_attribs;

// ------------------------------------------------------------------------
// DESCRIPTOR SETS --------------------------------------------------------
// ------------------------------------------------------------------------

layout(set = 2, binding = 0) uniform PerFrameUBO
{
    mat4 view_inverse;
    mat4 proj_inverse;
    mat4 view_proj_inverse;
    mat4 prev_view_proj;
    mat4 view_proj;
    vec4 cam_pos;
}
ubo;

// ------------------------------------------------------------------------
// MAIN -------------------------------------------------------------------
// ------------------------------------------------------------------------

void main()
{
    const Instance instance = Instances.data[gl_InstanceCustomIndexEXT];
    const HitInfo  hit_info = fetch_hit_info(instance, gl_PrimitiveID, gl_GeometryIndexEXT);
    const Triangle triangle = fetch_triangle(instance, hit_info);
    const Material material = Materials.data[hit_info.mat_idx];

    const vec3 barycentrics = vec3(1.0 - hit_attribs.x - hit_attribs.y, hit_attribs.x, hit_attribs.y);

    Vertex vertex = interpolated_vertex(triangle, barycentrics);

    transform_vertex(instance, vertex);

    vec4 albedo = fetch_albedo(material, vertex.tex_coord.xy);
    vec3 normal = fetch_normal(material, vertex.tangent.xyz, vertex.tangent.xyz, vertex.normal.xyz, vertex.tex_coord.xy);

    // TODO: TEMP
    vec3  dir          = normalize(vec3(1.0f, 1.0f, 0.0f));
    vec3  color        = albedo.rgb * max(dot(normal, dir), 0.0) + albedo.rgb * 0.1;
    float hit_distance = gl_HitTEXT;

    ray_payload.color = color;
}

// ------------------------------------------------------------------------