#version 460

#extension GL_EXT_scalar_block_layout : enable
#extension GL_EXT_ray_tracing : require
#extension GL_EXT_ray_query : enable
#extension GL_GOOGLE_include_directive : require
#extension GL_EXT_nonuniform_qualifier : require

#define IBL_INDIRECT_SPECULAR
#define RAY_TRACING
#include "../brdf.glsl"
#include "../scene_descriptor_set.glsl"
#include "../ray_query.glsl"
#include "../gi/gi_common.glsl"

// ------------------------------------------------------------------------
// PAYLOADS ---------------------------------------------------------------
// ------------------------------------------------------------------------

layout(location = 0) rayPayloadInEXT ReflectionPayload p_ReflectionPayload;

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

layout(set = 4, binding = 0) uniform samplerCube s_Cubemap;
layout(set = 4, binding = 1) uniform sampler2D s_IrradianceSH;
layout(set = 4, binding = 2) uniform samplerCube s_Prefiltered;
layout(set = 4, binding = 3) uniform sampler2D s_BRDF;

layout(set = 6, binding = 0) uniform sampler2D s_Irradiance;
layout(set = 6, binding = 1) uniform sampler2D s_Depth;
layout(set = 6, binding = 2, scalar) uniform DDGIUBO
{
    DDGIUniforms ddgi;
};

// ------------------------------------------------------------------------
// PUSH CONSTANTS ---------------------------------------------------------
// ------------------------------------------------------------------------

layout(push_constant) uniform PushConstants
{
    float bias;
    float trim;
    uint  num_frames;
    int   g_buffer_mip;
    int   sample_gi;
    int   approximate_with_ddgi;
    float gi_intensity;
    float rough_ddgi_intensity;
    float ibl_indirect_specular_intensity;
}
u_PushConstants;

// ------------------------------------------------------------------------
// FUNCTIONS --------------------------------------------------------------
// ------------------------------------------------------------------------

vec3 direct_lighting(vec3 Wo, vec3 N, vec3 P, vec3 F0, vec3 diffuse_color, float roughness)
{
    vec3 L = vec3(0.0f);

    uint  ray_flags  = gl_RayFlagsOpaqueEXT | gl_RayFlagsTerminateOnFirstHitEXT;
    uint  cull_mask  = 0xff;
    float tmin       = 0.001;
    float tmax       = 10000.0;
    vec3  ray_origin = P + N * u_PushConstants.bias;

    // Punctual Light
    {
        const Light light = ubo.light;
        const int   type  = light_type(light);

        if (type == LIGHT_TYPE_DIRECTIONAL)
        {
            vec3 Li = light_color(light) * light_intensity(light);
            vec3 Wi = light_direction(light);
            vec3 Wh = normalize(Wo + Wi);

            Li *= query_visibility(ray_origin, Wi, tmax, ray_flags);

            vec3  brdf      = evaluate_uber_brdf(diffuse_color, roughness, N, F0, Wo, Wh, Wi);
            float cos_theta = clamp(dot(N, Wi), 0.0, 1.0);

            L += brdf * cos_theta * Li;
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

            L += brdf * cos_theta * Li * attenuation;
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

            L += brdf * cos_theta * Li * attenuation;
        }
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
    const vec3 R = reflect(-Wo, N);

    vec3 F = fresnel_schlick_roughness(max(dot(N, Wo), 0.0), F0, roughness);

    vec3 kS = F;
    vec3 kD = 1.0 - kS;
    kD *= 1.0 - metallic;

#if defined(IBL_INDIRECT_SPECULAR)
    const float MAX_REFLECTION_LOD = 4.0;

    vec3 prefiltered_color = textureLod(s_Prefiltered, R, roughness * MAX_REFLECTION_LOD).rgb;
    vec2 brdf              = texture(s_BRDF, vec2(max(dot(N, Wo), 0.0), roughness)).rg;

    vec3 specular = prefiltered_color * (F * brdf.x + brdf.y) * u_PushConstants.ibl_indirect_specular_intensity;
#else
    vec3 specular = vec3(0.0f);
#endif

    vec3 diffuse = u_PushConstants.gi_intensity * diffuse_color * sample_irradiance(ddgi, P, N, Wo, s_Irradiance, s_Depth);

    return kD * diffuse + specular;
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

    const vec3 F0        = mix(vec3(0.04f), albedo, metallic);
    const vec3 c_diffuse = mix(albedo * (vec3(1.0f) - F0), vec3(0.0f), metallic);

    vec3 Li = direct_lighting(Wo, N, vertex.position.xyz, F0, c_diffuse, roughness);

    if (u_PushConstants.sample_gi == 1)
        Li += indirect_lighting(Wo, N, vertex.position.xyz, F0, c_diffuse, roughness, metallic);

    p_ReflectionPayload.color      = Li;
    p_ReflectionPayload.ray_length = gl_RayTminEXT + gl_HitTEXT;
}

// ------------------------------------------------------------------------