#version 460

#extension GL_GOOGLE_include_directive : require
#extension GL_EXT_nonuniform_qualifier : require

#include "scene_descriptor_set.glsl"

// ------------------------------------------------------------------------
// INPUTS -----------------------------------------------------------------
// ------------------------------------------------------------------------

layout(location = 0) in vec3 FS_IN_FragPos;
layout(location = 1) in vec2 FS_IN_Texcoord;
layout(location = 2) in vec3 FS_IN_Normal;
layout(location = 3) in vec3 FS_IN_Tangent;
layout(location = 4) in vec3 FS_IN_Bitangent;
layout(location = 5) in vec4 FS_IN_CSPos;
layout(location = 6) in vec4 FS_IN_PrevCSPos;
layout(location = 7) in vec3 FS_IN_OSNormal;

// ------------------------------------------------------------------------
// OUTPUTS ----------------------------------------------------------------
// ------------------------------------------------------------------------

layout(location = 0) out vec4 FS_OUT_GBuffer1; // RGB: Albedo, A: Roughness
layout(location = 1) out vec4 FS_OUT_GBuffer2; // RGB: Normal, A: Metallic
layout(location = 2) out vec4 FS_OUT_GBuffer3; // RG: Motion Vector, BA: -
layout(location = 3) out vec4 FS_OUT_LinearZ;

// ------------------------------------------------------------------------
// PUSH CONSTANTS ---------------------------------------------------------
// ------------------------------------------------------------------------

layout(push_constant) uniform PushConstants
{
    mat4 model;
    mat4 prev_model;
    uint material_idx;
}
u_PushConstants;

// ------------------------------------------------------------------------
// FUNCTIONS --------------------------------------------------------------
// ------------------------------------------------------------------------

vec2 motion_vector(vec4 prev_pos, vec4 current_pos)
{
    // Perspective division, covert clip space positions to NDC.
    vec2 current = (current_pos.xy / current_pos.w);
    vec2 prev    = (prev_pos.xy / prev_pos.w);

    // Remap to [0, 1] range
    current = current * 0.5 + 0.5;
    prev    = prev * 0.5 + 0.5;

    // Calculate velocity (current -> prev)
    return (prev - current);
}

// ------------------------------------------------------------------------

float compute_curvature(float depth)
{
    vec3 dx = dFdx(FS_IN_Normal);
    vec3 dy = dFdy(FS_IN_Normal);

    float x = dot(dx, dx);
    float y = dot(dy, dy);

    return pow(max(x, y), 0.5f);
}

// ------------------------------------------------------------------------

// A simple utility to convert a float to a 2-component octohedral representation packed into one uint
uint direction_to_octohedral(vec3 normal)
{
    vec2 p = normal.xy * (1.0 / dot(abs(normal), vec3(1.0)));
    vec2 e = normal.z > 0.0 ? p : (1.0 - abs(p.yx)) * (step(0.0, p) * 2.0 - vec2(1.0));
    return packSnorm2x16(e);
}

// ------------------------------------------------------------------------

// A simple utility to convert a float to a 2-component octohedral representation packed into one uint
vec2 direction_to_octohedral_2d(vec3 normal)
{
    vec2 p = normal.xy * (1.0 / dot(abs(normal), vec3(1.0)));
    return normal.z > 0.0 ? p : (1.0 - abs(p.yx)) * (step(0.0, p) * 2.0 - vec2(1.0));
}

// ------------------------------------------------------------------------
// MAIN -------------------------------------------------------------------
// ------------------------------------------------------------------------

void main()
{
    const Material material = Materials.data[u_PushConstants.material_idx];

    vec4 albedo = fetch_albedo(material, FS_IN_Texcoord);

    if (albedo.a < 0.1)
        discard;

    // Albedo
    FS_OUT_GBuffer1.rgb = albedo.rgb;

    // Metallic
    FS_OUT_GBuffer1.a = fetch_metallic(material, FS_IN_Texcoord);

    // Normal
    FS_OUT_GBuffer2.rg = direction_to_octohedral_2d(fetch_normal(material, normalize(FS_IN_Tangent), normalize(FS_IN_Bitangent), normalize(FS_IN_Normal), FS_IN_Texcoord));

    // Curvature
    float linear_z    = gl_FragCoord.z / gl_FragCoord.w;
    FS_OUT_GBuffer2.b = compute_curvature(linear_z);

    // Roughness
    FS_OUT_GBuffer2.a = fetch_roughness(material, FS_IN_Texcoord);

    // Motion Vector
    vec2 position_normal_fwidth = vec2(length(fwidth(FS_IN_FragPos)), length(fwidth(FS_IN_Normal)));
    vec2 motion_vec             = motion_vector(FS_IN_PrevCSPos, FS_IN_CSPos);

    FS_OUT_GBuffer3 = vec4(motion_vec, position_normal_fwidth);

    // Linear Z
    float max_change_z = max(abs(dFdx(linear_z)), abs(dFdy(linear_z)));
    float os_normal    = uintBitsToFloat(direction_to_octohedral(normalize(FS_IN_OSNormal)));

    FS_OUT_LinearZ = vec4(linear_z, max_change_z, FS_IN_PrevCSPos.z, os_normal);
}

// ------------------------------------------------------------------------