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

layout(location = 0) rayPayloadInEXT ReflectionPayload ray_payload;

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

// ------------------------------------------------------------------------
// FUNCTIONS --------------------------------------------------------------
// ------------------------------------------------------------------------

float distribution_ggx(vec3 N, vec3 H, float roughness)
{
    float a      = roughness * roughness;
    float a2     = a * a;
    float NdotH  = max(dot(N, H), 0.0);
    float NdotH2 = NdotH * NdotH;

    float nom   = a2;
    float denom = (NdotH2 * (a2 - 1.0) + 1.0);
    denom       = M_PI * denom * denom;

    return nom / max(EPSILON, denom);
}

// ------------------------------------------------------------------

float geometry_schlick_ggx(float NdotV, float roughness)
{
    float r = (roughness + 1.0);
    float k = (r * r) / 8.0;

    float nom   = NdotV;
    float denom = NdotV * (1.0 - k) + k;

    return nom / max(EPSILON, denom);
}

// ------------------------------------------------------------------

float geometry_smith(vec3 N, vec3 V, vec3 L, float roughness)
{
    float NdotV = max(dot(N, V), 0.0);
    float NdotL = max(dot(N, L), 0.0);
    float ggx2  = geometry_schlick_ggx(NdotV, roughness);
    float ggx1  = geometry_schlick_ggx(NdotL, roughness);

    return ggx1 * ggx2;
}
// ----------------------------------------------------------------------------
vec3 fresnel_schlick(float cosTheta, vec3 F0)
{
    return F0 + (1.0 - F0) * pow(max(1.0 - cosTheta, 0.0), 5.0);
}
// ----------------------------------------------------------------------------
vec3 fresnel_schlick_roughness(float cosTheta, vec3 F0, float roughness)
{
    return F0 + (max(vec3(1.0 - roughness), F0) - F0) * pow(max(1.0 - cosTheta, 0.0), 5.0);
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

    const vec3 albedo = fetch_albedo(material, vertex.tex_coord.xy).rgb;
    const float roughness  = fetch_roughness(material, vertex.tex_coord.xy);
    const float metallic   = fetch_metallic(material, vertex.tex_coord.xy);

    const vec3 N  = fetch_normal(material, vertex.tangent.xyz, vertex.tangent.xyz, vertex.normal.xyz, vertex.tex_coord.xy);
    const vec3 Wo = normalize(ubo.cam_pos.xyz - vertex.position.xyz);
    const vec3 R  = reflect(-Wo, N);

    vec3 F0 = mix(vec3(0.04f), albedo, metallic);

    vec3 direct = vec3(0.0f);

    // Direct Lighting
    {
        Light light = ubo.light;

        vec3 Li = light_color(light) * light_intensity(light);
        vec3 Wi = light_direction(light);
        vec3 Wh = normalize(Wo + Wi);

        // Cook-Torrance BRDF
        float NDF = distribution_ggx(N, Wh, roughness);
        float G   = geometry_smith(N, Wo, Wi, roughness);
        vec3  F   = fresnel_schlick(max(dot(Wh, Wo), 0.0), F0);

        vec3  nominator   = NDF * G * F;
        float denominator = 4 * max(dot(N, Wo), 0.0) * max(dot(N, Wi), 0.0); // 0.001 to prevent divide by zero.
        vec3  specular    = nominator / max(EPSILON, denominator);

        // kS is equal to Fresnel
        vec3 kS = F;
        vec3 kD = vec3(1.0) - kS;
        kD *= 1.0 - metallic;

        // scale light by NdotL
        float NdotL = max(dot(N, Wi), 0.0);

        // add to outgoing radiance Lo
        direct += (kD * albedo / M_PI + specular) * Li * NdotL;
    }

    ray_payload.color = direct;
    ray_payload.hit_position = vertex.position.xyz;
}

// ------------------------------------------------------------------------