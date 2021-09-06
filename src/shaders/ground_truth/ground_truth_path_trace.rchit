#version 460

#extension GL_EXT_ray_tracing : require
#extension GL_EXT_ray_query : enable
#extension GL_GOOGLE_include_directive : require
#extension GL_EXT_nonuniform_qualifier : require

#define RAY_TRACING
#include "../brdf.glsl"
#include "../scene_descriptor_set.glsl"
#include "../ray_query.glsl"
#define SOFT_SHADOWS
#define RAY_THROUGHPUT
#define SAMPLE_SKY_LIGHT
#include "../lighting.glsl"

// ------------------------------------------------------------------------
// PAYLOADS ---------------------------------------------------------------
// ------------------------------------------------------------------------

layout(location = 0) rayPayloadInEXT PathTracePayload p_Payload;

layout(location = 1) rayPayloadEXT PathTracePayload p_IndirectPayload;

// ------------------------------------------------------------------------
// HIT ATTRIBUTE ----------------------------------------------------------
// ------------------------------------------------------------------------

hitAttributeEXT vec2 hit_attribs;

// ------------------------------------------------------------------------
// DESCRIPTOR SETS --------------------------------------------------------
// ------------------------------------------------------------------------

layout(set = 3, binding = 0) uniform PerFrameUBO
{
    mat4  view_inverse;
    mat4  proj_inverse;
    mat4  view_proj_inverse;
    mat4  prev_view_proj;
    mat4  view_proj;
    vec4  cam_pos;
    vec4  current_prev_jitter;
    Light light;
}
ubo;

layout(set = 4, binding = 0) uniform samplerCube s_Cubemap;

// ------------------------------------------------------------------------
// PUSH CONSTANTS ---------------------------------------------------------
// ------------------------------------------------------------------------

layout(push_constant) uniform PushConstants
{
    uint  num_frames;
    uint  max_ray_bounces;
    float roughness_multiplier;
}
u_PushConstants;

// ------------------------------------------------------------------------
// FUNCTIONS --------------------------------------------------------------
// ------------------------------------------------------------------------

vec3 indirect_lighting(vec3 Wo, vec3 N, vec3 P, vec3 F0, vec3 diffuse_color, float roughness, float metallic)
{
    vec3  Wi;
    float pdf;

    vec3 brdf = sample_uber_brdf(diffuse_color, F0, N, roughness, metallic, Wo, p_Payload.rng, Wi, pdf);

    float cos_theta = clamp(dot(N, Wi), 0.0, 1.0);

    p_IndirectPayload.L = vec3(0.0f);
    p_IndirectPayload.T = p_Payload.T * (brdf * cos_theta) / pdf;

    // Russian roulette
    float probability = max(p_IndirectPayload.T.r, max(p_IndirectPayload.T.g, p_IndirectPayload.T.b));
    if (next_float(p_Payload.rng) > probability)
        return vec3(0.0f);

    // Add the energy we 'lose' by randomly terminating paths
    p_IndirectPayload.T *= 1.0f / probability;
    p_IndirectPayload.depth = p_Payload.depth + 1;
    p_IndirectPayload.rng   = p_Payload.rng;

    uint  ray_flags = gl_RayFlagsOpaqueEXT;
    uint  cull_mask = 0xFF;
    float tmin      = 0.0001;
    float tmax      = 10000.0;
    vec3  origin    = P.xyz;

    // Trace Ray
    traceRayEXT(u_TopLevelAS,
                ray_flags,
                cull_mask,
                0,
                0,
                0,
                origin,
                tmin,
                Wi,
                tmax,
                1);

    return p_IndirectPayload.L;
}

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

    const vec3  albedo    = fetch_albedo(material, vertex.tex_coord.xy).rgb;
    const float roughness = fetch_roughness(material, vertex.tex_coord.xy) * u_PushConstants.roughness_multiplier;
    const float metallic  = fetch_metallic(material, vertex.tex_coord.xy);

    const vec3 N  = normalize(fetch_normal(material, vertex.tangent.xyz, vertex.tangent.xyz, vertex.normal.xyz, vertex.tex_coord.xy));
    const vec3 Wo = -gl_WorldRayDirectionEXT;
    const vec3 R  = reflect(-Wo, N);

    const vec3 F0        = mix(vec3(0.04f), albedo, metallic);
    const vec3 c_diffuse = mix(albedo * (vec3(1.0f) - F0), vec3(0.0f), metallic);

    p_Payload.L += direct_lighting(ubo.light, Wo, N, vertex.position.xyz, F0, c_diffuse, roughness, p_Payload.T, next_vec2(p_Payload.rng), next_vec2(p_Payload.rng), s_Cubemap);

    if ((p_Payload.depth + 1) < u_PushConstants.max_ray_bounces)
        p_Payload.L += indirect_lighting(Wo, N, vertex.position.xyz, F0, c_diffuse, roughness, metallic);
}

// ------------------------------------------------------------------------