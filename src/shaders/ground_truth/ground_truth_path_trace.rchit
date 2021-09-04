#version 460

#extension GL_EXT_ray_tracing : require
#extension GL_EXT_ray_query : enable
#extension GL_GOOGLE_include_directive : require
#extension GL_EXT_nonuniform_qualifier : require

#define RAY_TRACING
#include "../brdf.glsl"
#include "../scene_descriptor_set.glsl"
#include "../ray_query.glsl"

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

            Li *= query_visibility(ray_origin, Wi, tmax, ray_flags);

            vec3  brdf      = evaluate_uber_brdf(diffuse_color, roughness, N, F0, Wo, Wh, Wi);
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

            vec3  brdf      = evaluate_uber_brdf(diffuse_color, roughness, N, F0, Wo, Wh, Wi);
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

            vec3  brdf      = evaluate_uber_brdf(diffuse_color, roughness, N, F0, Wo, Wh, Wi);
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
        Li *= query_visibility(ray_origin, Wi, tmax, ray_flags);

        vec3  brdf      = evaluate_uber_brdf(diffuse_color, roughness, N, F0, Wo, Wh, Wi);
        float cos_theta = clamp(dot(N, Wi), 0.0, 1.0);

        L += (p_Payload.T * brdf * cos_theta * Li) / pdf;
    }

    return L;
}

// ----------------------------------------------------------------------------

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

    p_Payload.L += direct_lighting(Wo, N, vertex.position.xyz, F0, c_diffuse, roughness);

    if ((p_Payload.depth + 1) < u_PushConstants.max_ray_bounces)
        p_Payload.L += indirect_lighting(Wo, N, vertex.position.xyz, F0, c_diffuse, roughness, metallic);
}

// ------------------------------------------------------------------------