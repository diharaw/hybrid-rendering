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

layout(location = 0) rayPayloadInEXT IndirectDiffusePayload p_PathTracePayload;

layout(location = 1) rayPayloadEXT IndirectDiffusePayload p_IndirectPayload;

layout(location = 2) rayPayloadEXT bool p_Visibility;

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
    Light light;
}
ubo;

layout(set = 4, binding = 0) uniform sampler2D s_IrradianceSH;
layout(set = 4, binding = 1) uniform samplerCube s_Prefiltered;
layout(set = 4, binding = 2) uniform sampler2D s_BRDF;

layout(set = 5, binding = 0) uniform samplerCube s_Cubemap;

// ------------------------------------------------------------------------
// PUSH CONSTANTS ---------------------------------------------------------
// ------------------------------------------------------------------------

layout(push_constant) uniform PushConstants
{
    float bias;
    uint  num_frames;
    uint  max_ray_depth;
    uint  sample_sky;
    int   g_buffer_mip;
}
u_PushConstants;

// ------------------------------------------------------------------------
// FUNCTIONS --------------------------------------------------------------
// ------------------------------------------------------------------------

mat3 make_rotation_matrix(vec3 z)
{
    const vec3 ref = abs(dot(z, vec3(0, 1, 0))) > 0.99f ? vec3(0, 0, 1) : vec3(0, 1, 0);

    const vec3 x = normalize(cross(ref, z));
    const vec3 y = cross(z, x);

    return mat3(x, y, z);
}

// ------------------------------------------------------------------------

vec3 sample_cosine_lobe(in vec3 n, in vec2 r)
{
    vec2 rand_sample = max(vec2(0.00001f), r);

    const float phi = 2.0f * M_PI * rand_sample.y;

    const float cos_theta = sqrt(rand_sample.x);
    const float sin_theta = sqrt(1 - rand_sample.x);

    vec3 t = vec3(sin_theta * cos(phi), sin_theta * sin(phi), cos_theta);

    return normalize(make_rotation_matrix(n) * t);
}

// ------------------------------------------------------------------------

float D_ggx(in float ndoth, in float alpha)
{
    float a2    = alpha * alpha;
    float denom = (ndoth * ndoth) * (a2 - 1.0) + 1.0;

    return a2 / max(EPSILON, (M_PI * denom * denom));
}

// ------------------------------------------------------------------------

float G1_schlick_ggx(in float roughness, in float ndotv)
{
    float k = ((roughness + 1) * (roughness + 1)) / 8.0;

    return ndotv / max(EPSILON, (ndotv * (1 - k) + k));
}

// ------------------------------------------------------------------------

float G_schlick_ggx(in float ndotl, in float ndotv, in float roughness)
{
    return G1_schlick_ggx(roughness, ndotl) * G1_schlick_ggx(roughness, ndotv);
}

// ------------------------------------------------------------------------

vec3 F_schlick(in vec3 f0, in float vdoth)
{
    return f0 + (vec3(1.0) - f0) * (pow(1.0 - vdoth, 5.0));
}

// ------------------------------------------------------------------------

vec3 evaluate_ggx(in float roughness, in vec3 F, in float ndoth, in float ndotl, in float ndotv)
{
    float alpha = roughness * roughness;
    return (D_ggx(ndoth, alpha) * F * G_schlick_ggx(ndotl, ndotv, roughness)) / max(EPSILON, (4.0 * ndotl * ndotv));
}

// ------------------------------------------------------------------------

float pdf_D_ggx(in float alpha, in float ndoth, in float vdoth)
{
    return D_ggx(ndoth, alpha) * ndoth / max(EPSILON, (4.0 * vdoth));
}

// ------------------------------------------------------------------------

float pdf_cosine_lobe(in float ndotl)
{
    return ndotl / M_PI;
}

// ------------------------------------------------------------------------

vec3 evaluate_lambert(in vec3 albedo)
{
    return albedo / M_PI;
}

// ------------------------------------------------------------------------

vec3 sample_lambert(in vec3 albedo, in vec3 N, in vec3 Wo, in RNG rng, out vec3 Wi, out float pdf, out float NdotL)
{
    vec3 Wh;

    Wi = sample_cosine_lobe(N, next_vec2(rng));
    Wh = normalize(Wo + Wi);

    NdotL = max(dot(N, Wi), 0.0);
    pdf   = pdf_cosine_lobe(NdotL);

    return evaluate_lambert(albedo);
}

// ------------------------------------------------------------------------

vec3 evaluate_uber(in vec3 albedo, in float roughness, in vec3 N, in vec3 F0, in vec3 Wo, in vec3 Wh, in vec3 Wi)
{
    float NdotL = max(dot(N, Wi), 0.0);
    float NdotV = max(dot(N, Wo), 0.0);
    float NdotH = max(dot(N, Wh), 0.0);
    float VdotH = max(dot(Wi, Wh), 0.0);

    vec3 F        = F_schlick(F0, VdotH);
    vec3 specular = evaluate_ggx(roughness, F, NdotH, NdotL, NdotV);
    vec3 diffuse  = evaluate_lambert(albedo.xyz);

    return (vec3(1.0) - F) * diffuse + specular;
}

// ------------------------------------------------------------------------

vec3 direct_lighting(vec3 Wo, vec3 N, vec3 P, vec3 F0, vec3 albedo, float roughness)
{
    vec3 L = vec3(0.0f);

    uint  ray_flags  = gl_RayFlagsOpaqueEXT | gl_RayFlagsTerminateOnFirstHitEXT;
    uint  cull_mask  = 0xff;
    float tmin       = 0.001;
    float tmax       = 10000.0;
    vec3  ray_origin = P + N * 0.1f;

    // Directional Light
    {
        const Light light = ubo.light;

        vec3 Li = light_color(light) * light_intensity(light);

        vec3 light_tangent   = normalize(cross(light_direction(light), vec3(0.0f, 1.0f, 0.0f)));
        vec3 light_bitangent = normalize(cross(light_tangent, light_direction(light)));

        // calculate disk point
        vec2  rnd_sample   = next_vec2(p_PathTracePayload.rng);
        float point_radius = light_radius(light) * sqrt(rnd_sample.x);
        float point_angle  = rnd_sample.y * 2.0f * M_PI;
        vec2  disk_point   = vec2(point_radius * cos(point_angle), point_radius * sin(point_angle));
        vec3  Wi           = normalize(light_direction(light) + disk_point.x * light_tangent + disk_point.y * light_bitangent);

        vec3 Wh = normalize(Wo + Wi);

        // fire shadow ray for visiblity
        traceRayEXT(u_TopLevelAS,
                    ray_flags,
                    cull_mask,
                    1,
                    0,
                    1,
                    ray_origin,
                    tmin,
                    Wi,
                    tmax,
                    2);

        Li *= float(p_Visibility);

        vec3  brdf      = evaluate_uber(albedo, roughness, N, F0, Wo, Wh, Wi);
        float cos_theta = clamp(dot(N, Wi), 0.0, 1.0);

        L += p_PathTracePayload.T * brdf * cos_theta * Li;
    }

    // Sky Light
    if (u_PushConstants.sample_sky == 1)
    {
        vec2  rand_value = next_vec2(p_PathTracePayload.rng);
        vec3  Wi         = sample_cosine_lobe(N, rand_value);
        vec3  Li         = texture(s_Cubemap, Wi).rgb;
        float pdf        = pdf_cosine_lobe(dot(N, Wi));
        vec3  Wh         = normalize(Wo + Wi);

        // fire shadow ray for visiblity
        traceRayEXT(u_TopLevelAS,
                    ray_flags,
                    cull_mask,
                    1,
                    0,
                    1,
                    ray_origin,
                    tmin,
                    Wi,
                    tmax,
                    2);

        Li *= float(p_Visibility);

        vec3  brdf      = evaluate_uber(albedo, roughness, N, F0, Wo, Wh, Wi);
        float cos_theta = clamp(dot(N, Wi), 0.0, 1.0);

        L += (p_PathTracePayload.T * brdf * cos_theta * Li) / pdf;
    }

    return L;
}

// ----------------------------------------------------------------------------

vec3 indirect_lighting(vec3 Wo, vec3 N, vec3 P, vec3 R, vec3 F0, vec3 albedo)
{
    vec3  Wi;
    float pdf;
    float NdotL;

    vec3 brdf = sample_lambert(albedo, N, Wo, p_PathTracePayload.rng, Wi, pdf, NdotL);

    p_IndirectPayload.L     = vec3(0.0f);
    p_IndirectPayload.T     = p_PathTracePayload.T * (brdf * NdotL) / pdf;
    p_IndirectPayload.depth = p_PathTracePayload.depth + 1;
    p_IndirectPayload.rng   = p_PathTracePayload.rng;

    uint  ray_flags = gl_RayFlagsOpaqueEXT;
    uint  cull_mask = 0xFF;
    float tmin      = 0.0001;
    float tmax      = 10000.0;
    vec3  origin    = P;

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
    const float roughness = fetch_roughness(material, vertex.tex_coord.xy);
    const float metallic  = fetch_metallic(material, vertex.tex_coord.xy);

    const vec3 N  = fetch_normal(material, vertex.tangent.xyz, vertex.tangent.xyz, vertex.normal.xyz, vertex.tex_coord.xy);
    const vec3 Wo = -gl_WorldRayDirectionEXT;
    const vec3 R  = reflect(-Wo, N);

    vec3 F0 = mix(vec3(0.04f), albedo, metallic);

    vec3 Li = direct_lighting(Wo, N, vertex.position.xyz, F0, albedo, roughness);

    if (p_PathTracePayload.depth < u_PushConstants.max_ray_depth)
        Li += indirect_lighting(Wo, N, vertex.position.xyz, R, F0, albedo);

    p_PathTracePayload.L = Li;
}

// ------------------------------------------------------------------------