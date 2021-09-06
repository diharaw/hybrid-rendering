#version 450

#extension GL_GOOGLE_include_directive : require

#include "brdf.glsl"
#include "lighting.glsl"

// ------------------------------------------------------------------------
// INPUTS -----------------------------------------------------------------
// ------------------------------------------------------------------------

layout(location = 0) in vec2 FS_IN_TexCoord;

// ------------------------------------------------------------------------
// OUTPUTS ----------------------------------------------------------------
// ------------------------------------------------------------------------

layout(location = 0) out vec4 FS_OUT_Color;

// ------------------------------------------------------------------------
// CONSTANTS --------------------------------------------------------------
// ------------------------------------------------------------------------

const float Pi                       = 3.141592654;
const float CosineA0                 = Pi;
const float CosineA1                 = (2.0 * Pi) / 3.0;
const float CosineA2                 = Pi * 0.25;
const float IndirectSpecularStrength = 2.0f;

// ------------------------------------------------------------------------
// DESCRIPTOR SETS --------------------------------------------------------
// ------------------------------------------------------------------------

layout(set = 0, binding = 0) uniform sampler2D s_GBuffer1; // RGB: Albedo, A: Metallic
layout(set = 0, binding = 1) uniform sampler2D s_GBuffer2; // RG: Normal, BA: Motion Vector
layout(set = 0, binding = 2) uniform sampler2D s_GBuffer3; // R: Roughness, G: Curvature, B: Mesh ID, A: Linear Z
layout(set = 0, binding = 3) uniform sampler2D s_GBufferDepth;

layout(set = 1, binding = 0) uniform sampler2D s_AO;

layout(set = 2, binding = 0) uniform sampler2D s_Shadow;

layout(set = 3, binding = 0) uniform sampler2D s_Reflections;

layout(set = 4, binding = 0) uniform sampler2D s_GI;

layout(set = 5, binding = 0) uniform PerFrameUBO
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
u_GlobalUBO;

layout(set = 6, binding = 1) uniform sampler2D s_IrradianceSH;
layout(set = 6, binding = 2) uniform samplerCube s_Prefiltered;
layout(set = 6, binding = 3) uniform sampler2D s_BRDF;

// ------------------------------------------------------------------------
// PUSH CONSTANTS ---------------------------------------------------------
// ------------------------------------------------------------------------

layout(push_constant) uniform PushConstants
{
    int shadow;
    int ao;
    int reflections;
    int gi;
}
u_PushConstants;

// ------------------------------------------------------------------------
// FUNCTIONS --------------------------------------------------------------
// ------------------------------------------------------------------------

struct SH9
{
    float c[9];
};

// ------------------------------------------------------------------

struct SH9Color
{
    vec3 c[9];
};

// ------------------------------------------------------------------

void project_onto_sh9(in vec3 dir, inout SH9 sh)
{
    // Band 0
    sh.c[0] = 0.282095;

    // Band 1
    sh.c[1] = -0.488603 * dir.y;
    sh.c[2] = 0.488603 * dir.z;
    sh.c[3] = -0.488603 * dir.x;

    // Band 2
    sh.c[4] = 1.092548 * dir.x * dir.y;
    sh.c[5] = -1.092548 * dir.y * dir.z;
    sh.c[6] = 0.315392 * (3.0 * dir.z * dir.z - 1.0);
    sh.c[7] = -1.092548 * dir.x * dir.z;
    sh.c[8] = 0.546274 * (dir.x * dir.x - dir.y * dir.y);
}

// ------------------------------------------------------------------

vec3 evaluate_sh9_irradiance(in vec3 direction)
{
    SH9 basis;

    project_onto_sh9(direction, basis);

    basis.c[0] *= CosineA0;
    basis.c[1] *= CosineA1;
    basis.c[2] *= CosineA1;
    basis.c[3] *= CosineA1;
    basis.c[4] *= CosineA2;
    basis.c[5] *= CosineA2;
    basis.c[6] *= CosineA2;
    basis.c[7] *= CosineA2;
    basis.c[8] *= CosineA2;

    vec3 color = vec3(0.0);

    for (int i = 0; i < 9; i++)
        color += texelFetch(s_IrradianceSH, ivec2(i, 0), 0).rgb * basis.c[i];

    color.x = max(0.0, color.x);
    color.y = max(0.0, color.y);
    color.z = max(0.0, color.z);

    return color / Pi;
}

// ------------------------------------------------------------------------

vec3 fresnel_schlick_roughness(float cosTheta, vec3 F0, float roughness)
{
    return F0 + (max(vec3(1.0 - roughness), F0) - F0) * pow(max(1.0 - cosTheta, 0.0), 5.0);
}

// ------------------------------------------------------------------------

vec3 indirect_lighting(vec3 N, vec3 diffuse_color, float roughness, float metallic, float ao, vec3 Wo, vec3 F0)
{
    const vec3 R = reflect(-Wo, N);

    vec3 F = fresnel_schlick_roughness(max(dot(N, Wo), 0.0), F0, roughness);

    vec3 kS = F;
    vec3 kD = 1.0 - kS;
    kD *= 1.0 - metallic;

    vec3 irradiance = u_PushConstants.gi == 1 ? textureLod(s_GI, FS_IN_TexCoord, 0.0f).rgb : evaluate_sh9_irradiance(N);
    vec3 diffuse    = irradiance * diffuse_color;

    const float MAX_REFLECTION_LOD = 4.0;
    vec3        prefilteredColor   = u_PushConstants.reflections == 1 ? textureLod(s_Reflections, FS_IN_TexCoord, 0.0f).rgb : textureLod(s_Prefiltered, R, roughness * MAX_REFLECTION_LOD).rgb;
    vec2        brdf               = texture(s_BRDF, vec2(max(dot(N, Wo), 0.0), roughness)).rg;
    vec3        specular           = prefilteredColor * (F * brdf.x + brdf.y) * IndirectSpecularStrength;

    return (kD * diffuse + specular) * ao;
}

// ------------------------------------------------------------------------
// MAIN -------------------------------------------------------------------
// ------------------------------------------------------------------------

void main()
{
    vec4 g_buffer_data_1 = texture(s_GBuffer1, FS_IN_TexCoord);
    vec4 g_buffer_data_2 = texture(s_GBuffer2, FS_IN_TexCoord);
    vec4 g_buffer_data_3 = texture(s_GBuffer3, FS_IN_TexCoord);

    const vec3  world_pos  = world_position_from_depth(FS_IN_TexCoord, texture(s_GBufferDepth, FS_IN_TexCoord).r, u_GlobalUBO.view_proj_inverse);
    const vec3  albedo     = g_buffer_data_1.rgb;
    const float metallic   = g_buffer_data_1.a;
    const float roughness  = g_buffer_data_3.r;
    const float visibility = u_PushConstants.shadow == 1 ? texture(s_Shadow, FS_IN_TexCoord).r : 1.0f;
    const float ao         = u_PushConstants.ao == 1 ? texture(s_AO, FS_IN_TexCoord).r : 1.0f;

    const vec3 N  = octohedral_to_direction(g_buffer_data_2.rg);
    const vec3 Wo = normalize(u_GlobalUBO.cam_pos.xyz - world_pos);

    const vec3 F0        = mix(vec3(0.04f), albedo, metallic);
    const vec3 c_diffuse = mix(albedo * (vec3(1.0f) - F0), vec3(0.0f), metallic);

    vec3 Lo = vec3(0.0f);

    // Direct Lighting
    Lo += direct_lighting(u_GlobalUBO.light, Wo, N, world_pos, F0, c_diffuse, roughness) * visibility;

    // Indirect lighting
    Lo += indirect_lighting(N, c_diffuse, roughness, metallic, ao, Wo, F0);

    FS_OUT_Color = vec4(Lo, 1.0);
}

// ------------------------------------------------------------------------