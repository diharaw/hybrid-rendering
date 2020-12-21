#version 450

layout(set = 0, binding = 0) uniform sampler2D s_GBuffer1; // RGB: Albedo, A: Roughness
layout(set = 0, binding = 1) uniform sampler2D s_GBuffer2; // RGB: Normal, A: Metallic
layout(set = 0, binding = 2) uniform sampler2D s_GBuffer3; // RGB: Position, A: -

layout(set = 1, binding = 0) uniform sampler2D s_Shadow;
layout(set = 2, binding = 0) uniform sampler2D s_Reflection;

layout(set = 3, binding = 0) uniform PerFrameUBO
{
    mat4 view_inverse;
    mat4 proj_inverse;
    mat4 model;
    mat4 view;
    mat4 projection;
    vec4 cam_pos;
    vec4 light_dir;
}
ubo;

layout(location = 0) in vec2 inUV;

layout(location = 0) out vec4 outFragColor;

void main()
{
    vec3 albedo = texture(s_GBuffer1, inUV).rgb;
    vec3 normal = texture(s_GBuffer2, inUV).rgb;
    vec3 reflection = texture(s_Reflection, inUV).rgb;
    float shadow = texture(s_Shadow, inUV).r;

    vec3 color = shadow * albedo * max(dot(normal, ubo.light_dir.xyz), 0.0) + albedo * 0.1 + reflection;

    // Reinhard tone mapping
    color = color / (1.0 + color);

    // Gamma correction
    color = pow(color, vec3(1.0/2.2));

    outFragColor = vec4(color, 1.0);
}