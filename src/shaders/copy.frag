#version 450

layout(set = 0, binding = 0) uniform sampler2D samplerColor;

layout(location = 0) in vec2 inUV;

layout(location = 0) out vec4 outFragColor;

vec3 aces_film(vec3 x)
{
    float a = 2.51f;
    float b = 0.03f;
    float c = 2.43f;
    float d = 0.59f;
    float e = 0.14f;
    return clamp((x * (a * x + b)) / (x * (c * x + d) + e), 0.0f, 1.0f);
}

void main()
{
    vec3 color = texture(samplerColor, inUV).rgb;

    // Apply exposure
    color *= 0.7f;

    // HDR tonemap and gamma correct
    color = aces_film(color);
    color = pow(color, vec3(1.0 / 2.2));

    outFragColor = vec4(color, 1.0);
}