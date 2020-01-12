#version 460

layout (location = 0) in vec3 FS_IN_FragPos;
layout (location = 1) in vec2 FS_IN_Texcoord;
layout (location = 2) in vec3 FS_IN_Normal;
layout (location = 3) in vec3 FS_IN_Tangent;
layout (location = 4) in vec3 FS_IN_Bitangent;

layout (location = 0) out vec4 FS_OUT_GBuffer1; // RGB: Albedo, A: Roughness
layout (location = 1) out vec4 FS_OUT_GBuffer2; // RGB: Normal, A: Metallic
layout (location = 2) out vec4 FS_OUT_GBuffer3; // RGB: Position, A: -

layout (set = 1, binding = 0) uniform sampler2D s_Diffuse;
layout (set = 1, binding = 1) uniform sampler2D s_Normal;
layout (set = 1, binding = 2) uniform sampler2D s_Metallic;
layout (set = 1, binding = 3) uniform sampler2D s_Roughness;

vec3 get_normal_from_map(vec3 tangent, vec3 bitangent, vec3 normal, vec2 tex_coord, sampler2D normal_map)
{
	// Create TBN matrix.
    mat3 TBN = mat3(normalize(tangent), normalize(bitangent), normalize(normal));

    // Sample tangent space normal vector from normal map and remap it from [0, 1] to [-1, 1] range.
    vec3 n = normalize(texture(normal_map, tex_coord).xyz * 2.0 - 1.0);

    // Multiple vector by the TBN matrix to transform the normal from tangent space to world space.
    n = normalize(TBN * n);

    return n;
}

void main()
{
	// Albedo
	FS_OUT_GBuffer1.rgb = texture(s_Diffuse, FS_IN_Texcoord).rgb;

    // Normal.
    FS_OUT_GBuffer2.rgb = get_normal_from_map(FS_IN_Tangent, FS_IN_Bitangent, FS_IN_Normal, FS_IN_Texcoord, s_Normal);

	// Roughness
	FS_OUT_GBuffer1.a = texture(s_Roughness, FS_IN_Texcoord).a;

	// Metallic
	FS_OUT_GBuffer2.a = texture(s_Metallic, FS_IN_Texcoord).a;

	// World Pos
	FS_OUT_GBuffer3.rgb = FS_IN_FragPos;
}