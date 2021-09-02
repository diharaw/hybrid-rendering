#version 460

#extension GL_EXT_scalar_block_layout : enable
#extension GL_EXT_ray_tracing : require
#extension GL_EXT_ray_query : enable
#extension GL_GOOGLE_include_directive : require
#extension GL_EXT_nonuniform_qualifier : require

#define RAY_TRACING
#include "../brdf.glsl"
#include "../scene_descriptor_set.glsl"
#include "gi_common.glsl"

// ------------------------------------------------------------------------
// PAYLOADS ---------------------------------------------------------------
// ------------------------------------------------------------------------

layout(location = 0) rayPayloadInEXT GIPayload p_GIPayload;

// ------------------------------------------------------------------------
// HIT ATTRIBUTE ----------------------------------------------------------
// ------------------------------------------------------------------------

hitAttributeEXT vec2 hit_attribs;

// ------------------------------------------------------------------------
// DESCRIPTOR SETS --------------------------------------------------------
// ------------------------------------------------------------------------

layout(set = 2, binding = 0) uniform PerFrameUBO
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

layout(set = 3, binding = 0) uniform samplerCube s_Cubemap;

layout(set = 4, binding = 0) uniform sampler2D s_Irradiance;
layout(set = 4, binding = 1) uniform sampler2D s_Depth;
layout(set = 4, binding = 2, scalar) uniform DDGIUBO
{
    DDGIUniforms ddgi;
};

// ------------------------------------------------------------------------
// PUSH CONSTANTS ---------------------------------------------------------
// ------------------------------------------------------------------------

layout(push_constant) uniform PushConstants
{
    mat4  random_orientation;
    uint  num_frames;
    uint  infinite_bounces;
    float gi_intensity;
}
u_PushConstants;

// ------------------------------------------------------------------------
// FUNCTIONS --------------------------------------------------------------
// ------------------------------------------------------------------------

float query_visibility(vec3 world_pos, vec3 direction)
{
    float t_min     = 0.01f;
    float t_max     = 100000.0f;
    uint  ray_flags = gl_RayFlagsOpaqueEXT;

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

vec3 direct_lighting(vec3 Wo, vec3 N, vec3 P, vec3 F0, vec3 diffuse_color, float roughness)
{
    vec3 L = vec3(0.0f);

    uint  ray_flags  = gl_RayFlagsOpaqueEXT | gl_RayFlagsTerminateOnFirstHitEXT;
    uint  cull_mask  = 0xff;
    float tmin       = 0.001;
    float tmax       = 10000.0;
    vec3  ray_origin = P + N * 0.1f;

    // Punctual Light
    {
        const Light light = ubo.light;
        const int   type  = light_type(light);

        if (type == LIGHT_TYPE_DIRECTIONAL)
        {
            vec3 Li = light_color(light) * light_intensity(light);
            vec3 Wi = light_direction(light);
            vec3 Wh = normalize(Wo + Wi);

            Li *= query_visibility(ray_origin, Wi);

            vec3  brdf      = evaluate_uber_brdf(diffuse_color, roughness, N, F0, Wo, Wh, Wi);
            float cos_theta = clamp(dot(N, Wi), 0.0, 1.0);

            L += p_GIPayload.T * brdf * cos_theta * Li;
        }
        else if (type == LIGHT_TYPE_POINT)
        {
            vec3  to_light       = light_position(light) - P;
            float light_distance = length(to_light);
            float attenuation    = (1.0f / (light_distance * light_distance));

            vec3 Li = light_color(light) * light_intensity(light);
            vec3 Wi = normalize(to_light);
            vec3 Wh = normalize(Wo + Wi);

            Li *= query_distance(ray_origin, Wi, light_distance);

            vec3  brdf      = evaluate_uber_brdf(diffuse_color, roughness, N, F0, Wo, Wh, Wi);
            float cos_theta = clamp(dot(N, Wi), 0.0, 1.0);

            L += p_GIPayload.T * brdf * cos_theta * Li * attenuation;
        }
        else
        {
            vec3  to_light       = light_position(light) - P;
            float light_distance = length(to_light);

            vec3 Li = light_color(light) * light_intensity(light);
            vec3 Wi = normalize(to_light);
            vec3 Wh = normalize(Wo + Wi);

            float angle_attenuation = dot(Wi, light_direction(light));
            angle_attenuation       = smoothstep(light_cos_theta_outer(light), light_cos_theta_inner(light), angle_attenuation);

            float attenuation = (angle_attenuation / (light_distance * light_distance));

            Li *= query_distance(ray_origin, Wi, light_distance);

            vec3  brdf      = evaluate_uber_brdf(diffuse_color, roughness, N, F0, Wo, Wh, Wi);
            float cos_theta = clamp(dot(N, Wi), 0.0, 1.0);

            L += p_GIPayload.T * brdf * cos_theta * Li * attenuation;
        }
    }

    // Sky Light
    {
        vec2  rand_value = next_vec2(p_GIPayload.rng);
        vec3  Wi         = sample_cosine_lobe(N, rand_value);
        vec3  Li         = texture(s_Cubemap, Wi).rgb;
        float pdf        = pdf_cosine_lobe(dot(N, Wi));
        vec3  Wh         = normalize(Wo + Wi);

        // fire shadow ray for visiblity
        Li *= query_visibility(ray_origin, Wi);

        vec3  brdf      = evaluate_uber_brdf(diffuse_color, roughness, N, F0, Wo, Wh, Wi);
        float cos_theta = clamp(dot(N, Wi), 0.0, 1.0);

        L += (p_GIPayload.T * brdf * cos_theta * Li) / pdf;
    }

    return L;
}

// ----------------------------------------------------------------------------

vec3 fresnel_schlick_roughness(float cosTheta, vec3 F0, float roughness)
{
    return F0 + (max(vec3(1.0 - roughness), F0) - F0) * pow(max(1.0 - cosTheta, 0.0), 5.0);
}

// ----------------------------------------------------------------------------

vec3 indirect_lighting(vec3 Wo, vec3 N, vec3 P, vec3 F0, vec3 diffuse_color, float roughness, float metallic)
{
    vec3 F = fresnel_schlick_roughness(max(dot(N, Wo), 0.0), F0, roughness);

    vec3 kS = F;
    vec3 kD = 1.0 - kS;
    kD *= 1.0 - metallic;

    return u_PushConstants.gi_intensity * kD * diffuse_color * sample_irradiance(ddgi, P, N, Wo, s_Irradiance, s_Depth);
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
    const float roughness = fetch_roughness(material, vertex.tex_coord.xy);
    const float metallic  = fetch_metallic(material, vertex.tex_coord.xy);

    const vec3 N  = fetch_normal(material, vertex.tangent.xyz, vertex.tangent.xyz, vertex.normal.xyz, vertex.tex_coord.xy);
    const vec3 Wo = -gl_WorldRayDirectionEXT;
    const vec3 R  = reflect(-Wo, N);

    const vec3 F0 = mix(vec3(0.04f), albedo, metallic);
    const vec3 c_diffuse = mix(albedo * (vec3(1.0f) - F0),  vec3(0.0f), metallic);

    vec3 Li = direct_lighting(Wo, N, vertex.position.xyz, F0, c_diffuse, roughness);

    if (u_PushConstants.infinite_bounces == 1)
        Li += indirect_lighting(Wo, N, vertex.position.xyz, F0, c_diffuse, roughness, metallic);

    p_GIPayload.L            = Li;
    p_GIPayload.hit_distance = gl_RayTminEXT + gl_HitTEXT;
}

// ------------------------------------------------------------------------