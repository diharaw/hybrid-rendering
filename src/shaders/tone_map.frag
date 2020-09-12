#version 450

layout(set = 0, binding = 0) uniform sampler2D samplerColor;

layout(location = 0) in vec2 inUV;

layout(location = 0) out vec4 outFragColor;

void main()
{
    vec3 color = texture(samplerColor, inUV).rgb;

    // Reinhard tone mapping
    color = color / (1.0 + color);

    // Gamma correction
    color = pow(color, vec3(1.0/2.2));

    outFragColor = vec4(color, 1.0);
}