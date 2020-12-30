#version 450

// ------------------------------------------------------------------------
// INPUTS -----------------------------------------------------------------
// ------------------------------------------------------------------------

layout(location = 0) in vec2 FS_IN_TexCoord;

// ------------------------------------------------------------------------
// OUTPUTS ----------------------------------------------------------------
// ------------------------------------------------------------------------

layout(location = 0) out vec4 FS_OUT_Color;

// ------------------------------------------------------------------------
// DESCRIPTOR SETS --------------------------------------------------------
// ------------------------------------------------------------------------

layout(set = 0, binding = 0) uniform sampler2D s_GBuffer1; // RGB: Albedo, A: Roughness
layout(set = 0, binding = 1) uniform sampler2D s_GBuffer2; // RGB: Normal, A: Metallic
layout(set = 0, binding = 2) uniform sampler2D s_GBuffer3; // RG: Motion Vector, BA: -
layout(set = 0, binding = 3) uniform sampler2D s_GBufferDepth;

layout(set = 1, binding = 0) uniform sampler2D s_Shadow;

layout(set = 2, binding = 0) uniform PerFrameUBO
{
    mat4 view_inverse;
    mat4 proj_inverse;
    mat4 view_proj_inverse;
    mat4 prev_view_proj;
    mat4 model;
    mat4 view;
    mat4 projection;
    vec4 cam_pos;
    vec4 light_dir;
}
ubo;

// ------------------------------------------------------------------------
// MAIN -------------------------------------------------------------------
// ------------------------------------------------------------------------

void main()
{
    vec3 albedo = texture(s_GBuffer1, FS_IN_TexCoord).rgb;
    vec3 normal = texture(s_GBuffer2, FS_IN_TexCoord).rgb;
    //vec3  reflection = texture(s_Reflection, inUV).rgb;
    float shadow = texture(s_Shadow, FS_IN_TexCoord).r;

    // TODO: TEMP
    vec3 dir   = normalize(vec3(1.0f, 1.0f, 0.0f));
    vec3 color = shadow * albedo * max(dot(normal, dir), 0.0) + albedo * 0.1; // + reflection;
    //vec3 color = shadow * vec3(max(dot(normal, ubo.light_dir.xyz), 0.0)) + vec3(0.1);

    // Reinhard tone mapping
    color = color / (1.0 + color);

    // Gamma correction
    color = pow(color, vec3(1.0 / 2.2));

    FS_OUT_Color = vec4(color, 1.0);
}

// ------------------------------------------------------------------------