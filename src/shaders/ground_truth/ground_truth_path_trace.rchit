#version 460

#extension GL_EXT_ray_tracing : require
#extension GL_EXT_ray_query : enable
#extension GL_GOOGLE_include_directive : require
#extension GL_EXT_nonuniform_qualifier : require

#define RAY_TRACING
#include "../common.glsl"
#include "../scene_descriptor_set.glsl"

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

vec3 sample_ggx(in vec3 n, in float alpha, in vec2 Xi)
{
    float phi       = 2.0 * M_PI * Xi.x;
    float cos_theta = sqrt((1.0 - Xi.y) / (1.0 + (alpha * alpha - 1.0) * Xi.y));
    float sin_theta = sqrt(1.0 - cos_theta * cos_theta);

    vec3 d;

    d.x = sin_theta * cos(phi);
    d.y = sin_theta * sin(phi);
    d.z = cos_theta;

    return normalize(make_rotation_matrix(n) * d);
}

// ------------------------------------------------------------------------

float pdf_uber(in vec3 N, in float roughness, in vec3 Wo, in vec3 Wh, in vec3 Wi)
{
    float NdotL = max(dot(N, Wi), 0.0);
    float NdotV = max(dot(N, Wo), 0.0);
    float NdotH = max(dot(N, Wh), 0.0);
    float VdotH = max(dot(Wi, Wh), 0.0);

    float pd = pdf_cosine_lobe(NdotL);
    float ps = pdf_D_ggx(roughness * roughness, NdotH, VdotH);

    return mix(pd, ps, 0.5);
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

vec3 sample_uber(in vec3 albedo, in vec3 F0, in vec3 N, in float roughness, in vec3 Wo, in RNG rng, out vec3 Wi, out float pdf)
{
    float alpha = roughness * roughness;

    vec3 Wh;

    vec3 rand_value = next_vec3(rng);

    bool is_specular = false;

    if (rand_value.x < 0.5)
    {
        Wh = sample_ggx(N, alpha, rand_value.yz);

        if (roughness < MIRROR_REFLECTIONS_ROUGHNESS_THRESHOLD)
            Wi = reflect(-Wo, N);
        else
            Wi = reflect(-Wo, Wh);

        float NdotL = max(dot(N, Wi), 0.0);
        float NdotV = max(dot(N, Wo), 0.0);

        if (NdotL > 0.0f && NdotV > 0.0f)
            is_specular = true;
    }

    if (!is_specular)
    {
        Wi = sample_cosine_lobe(N, rand_value.yz);
        Wh = normalize(Wo + Wi);
    }

    pdf = pdf_uber(N, roughness, Wo, Wh, Wi);

    return evaluate_uber(albedo, roughness, N, F0, Wo, Wh, Wi);
}

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

vec3 direct_lighting(vec3 Wo, vec3 N, vec3 P, vec3 F0, vec3 albedo, float roughness)
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

            // calculate disk point
            vec2  rng          = next_vec2(p_Payload.rng);
            float point_radius = light_radius(light) * sqrt(rng.x);
            float point_angle  = rng.y * 2.0f * M_PI;
            vec2  disk_point   = vec2(point_radius * cos(point_angle), point_radius * sin(point_angle));

            vec3 light_dir       = light_direction(light);
            vec3 light_tangent   = normalize(cross(light_dir, vec3(0.0f, 1.0f, 0.0f)));
            vec3 light_bitangent = normalize(cross(light_tangent, light_dir));
            vec3 Wi              = normalize(light_dir + disk_point.x * light_tangent + disk_point.y * light_bitangent);
            vec3 Wh              = normalize(Wo + Wi);

            Li *= query_visibility(ray_origin, Wi);

            vec3  brdf      = evaluate_uber(albedo, roughness, N, F0, Wo, Wh, Wi);
            float cos_theta = clamp(dot(N, Wi), 0.0, 1.0);

            L += p_Payload.T * brdf * cos_theta * Li;
        }
        else if (type == LIGHT_TYPE_POINT)
        {
            vec3  to_light             = light_position(light) - P;
            float light_distance       = length(to_light);
            float attenuation          = (1.0f / (light_distance * light_distance));
            float current_light_radius = light_radius(light) / light_distance;

            vec3 Li = light_color(light) * light_intensity(light);

            vec3 light_dir       = normalize(to_light);
            vec3 light_tangent   = normalize(cross(light_dir, vec3(0.0f, 1.0f, 0.0f)));
            vec3 light_bitangent = normalize(cross(light_tangent, light_dir));

            // calculate disk point
            vec2  rng          = next_vec2(p_Payload.rng);
            float point_radius = current_light_radius * sqrt(rng.x);
            float point_angle  = rng.y * 2.0f * M_PI;
            vec2  disk_point   = vec2(point_radius * cos(point_angle), point_radius * sin(point_angle));

            vec3 Wi = normalize(light_dir + disk_point.x * light_tangent + disk_point.y * light_bitangent);
            vec3 Wh = normalize(Wo + Wi);

            Li *= query_distance(ray_origin, Wi, light_distance);

            vec3  brdf      = evaluate_uber(albedo, roughness, N, F0, Wo, Wh, Wi);
            float cos_theta = clamp(dot(N, Wi), 0.0, 1.0);

            L += p_Payload.T * brdf * cos_theta * Li * attenuation;
        }
        else
        {
            vec3  to_light             = light_position(light) - P;
            float light_distance       = length(to_light);
            float current_light_radius = light_radius(light) / light_distance;

            vec3 Li = light_color(light) * light_intensity(light);

            vec3 light_dir       = normalize(to_light);
            vec3 light_tangent   = normalize(cross(light_dir, vec3(0.0f, 1.0f, 0.0f)));
            vec3 light_bitangent = normalize(cross(light_tangent, light_dir));

            // calculate disk point
            vec2  rng          = next_vec2(p_Payload.rng);
            float point_radius = current_light_radius * sqrt(rng.x);
            float point_angle  = rng.y * 2.0f * M_PI;
            vec2  disk_point   = vec2(point_radius * cos(point_angle), point_radius * sin(point_angle));

            vec3 Wi = normalize(light_dir + disk_point.x * light_tangent + disk_point.y * light_bitangent);
            vec3 Wh = normalize(Wo + Wi);

            float angle_attenuation = dot(Wi, light_direction(light));
            angle_attenuation       = smoothstep(light_cos_theta_outer(light), light_cos_theta_inner(light), angle_attenuation);

            float attenuation = (angle_attenuation / (light_distance * light_distance));

            Li *= query_distance(ray_origin, Wi, light_distance);

            vec3  brdf      = evaluate_uber(albedo, roughness, N, F0, Wo, Wh, Wi);
            float cos_theta = clamp(dot(N, Wi), 0.0, 1.0);

            L += p_Payload.T * brdf * cos_theta * Li * attenuation;
        }
    }

    // Sky Light
    {
        vec2  rng = next_vec2(p_Payload.rng);
        vec3  Wi  = sample_cosine_lobe(N, rng);
        vec3  Li  = texture(s_Cubemap, Wi).rgb;
        float pdf = pdf_cosine_lobe(dot(N, Wi));
        vec3  Wh  = normalize(Wo + Wi);

        // fire shadow ray for visiblity
        Li *= query_visibility(ray_origin, Wi);

        vec3  brdf      = evaluate_uber(albedo, roughness, N, F0, Wo, Wh, Wi);
        float cos_theta = clamp(dot(N, Wi), 0.0, 1.0);

        L += (p_Payload.T * brdf * cos_theta * Li) / pdf;
    }

    return L;
}

// ----------------------------------------------------------------------------

vec3 indirect_lighting(vec3 Wo, vec3 N, vec3 P, vec3 F0, vec3 albedo, float roughness, float metallic)
{
    vec3  Wi;
    float pdf;

    vec3 brdf = sample_uber(albedo, F0, N, roughness, Wo, p_Payload.rng, Wi, pdf);

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

    vec3 F0 = mix(vec3(0.04f), albedo, metallic);

    p_Payload.L += direct_lighting(Wo, N, vertex.position.xyz, F0, albedo, roughness);

    if ((p_Payload.depth + 1) < u_PushConstants.max_ray_bounces)
        p_Payload.L += indirect_lighting(Wo, N, vertex.position.xyz, F0, albedo, roughness, metallic);
}

// ------------------------------------------------------------------------