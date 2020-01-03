#version 450

layout (location = 0) in vec3 FS_IN_FragPos;
layout (location = 1) in vec2 FS_IN_Texcoord;
layout (location = 2) in vec3 FS_IN_Normal;

layout (location = 0) out vec3 FS_OUT_Color;

layout (set = 1, binding = 0) uniform sampler2D s_Diffuse;
layout (set = 1, binding = 1) uniform sampler2D s_Normal;
layout (set = 1, binding = 2) uniform sampler2D s_Metallic;
layout (set = 1, binding = 3) uniform sampler2D s_Roughness;

void main()
{
    vec3 light_pos = vec3(-200.0, 200.0, 0.0);

	vec3 n = normalize(FS_IN_Normal);
	vec3 l = normalize(light_pos - FS_IN_FragPos);

	float lambert = max(0.0f, dot(n, l));

    vec3 diffuse = texture(s_Diffuse, FS_IN_Texcoord).xyz;
	vec3 ambient = diffuse * 0.03;

	vec3 color = diffuse * lambert + ambient;

    FS_OUT_Color = color;
}