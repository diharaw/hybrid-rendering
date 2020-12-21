#version 460

layout(location = 0) in vec3 FS_IN_FragPos;
layout(location = 1) in vec2 FS_IN_Texcoord;
layout(location = 2) in vec3 FS_IN_Normal;
layout(location = 3) in vec3 FS_IN_Tangent;
layout(location = 4) in vec3 FS_IN_Bitangent;
layout(location = 5) in vec4 FS_IN_CSPos;
layout(location = 6) in vec4 FS_IN_PrevCSPos;

layout(location = 0) out vec4 FS_OUT_GBuffer1; // RGB: Albedo, A: Roughness
layout(location = 1) out vec4 FS_OUT_GBuffer2; // RGB: Normal, A: Metallic
layout(location = 2) out vec4 FS_OUT_GBuffer3; // RG: Motion Vector, BA: -

layout(set = 1, binding = 0) uniform sampler2D s_Diffuse;
layout(set = 1, binding = 1) uniform sampler2D s_Normal;
layout(set = 1, binding = 2) uniform sampler2D s_Roughness;
layout(set = 1, binding = 3) uniform sampler2D s_Metallic;

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

vec2 motion_vector(vec4 prev_pos, vec4 current_pos)
{
    // Perspective division, covert clip space positions to NDC.
    vec2 current = (current_pos.xy / current_pos.w);
    vec2 prev    = (prev_pos.xy / prev_pos.w);

    // Remove jitter
    //current -= current_prev_jitter.xy;
    //prev -= current_prev_jitter.zw;

    // Remap to [0, 1] range
    current = current * 0.5 + 0.5;
    prev    = prev * 0.5 + 0.5;

    // Calculate velocity (prev -> current)
    return (current - prev);
}

void main()
{
    vec4 albedo = texture(s_Diffuse, FS_IN_Texcoord);

    if (albedo.a < 0.1)
        discard;

    // Albedo
    FS_OUT_GBuffer1.rgb = albedo.rgb;

    // Normal.
    FS_OUT_GBuffer2.rgb = get_normal_from_map(FS_IN_Tangent, FS_IN_Bitangent, FS_IN_Normal, FS_IN_Texcoord, s_Normal);

    // Roughness
    FS_OUT_GBuffer1.a = texture(s_Roughness, FS_IN_Texcoord).r;

    // Metallic
    FS_OUT_GBuffer2.a = texture(s_Metallic, FS_IN_Texcoord).r;

    // Motion Vector
    FS_OUT_GBuffer3.rg = motion_vector(FS_IN_PrevCSPos, FS_IN_CSPos);
}