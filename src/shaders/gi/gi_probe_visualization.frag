#version 460

// ------------------------------------------------------------------------
// INPUTS -----------------------------------------------------------------
// ------------------------------------------------------------------------

layout(location = 0) in vec3 FS_IN_FragPos;
layout(location = 1) in vec3 FS_IN_Normal;
layout(location = 2) in flat int FS_IN_ProbeIdx;

// ------------------------------------------------------------------------
// OUTPUTS ----------------------------------------------------------------
// ------------------------------------------------------------------------

layout(location = 0) out vec4 FS_OUT_Color;

// ------------------------------------------------------------------------
// DESCRIPTOR SET ---------------------------------------------------------
// ------------------------------------------------------------------------

layout(set = 1, binding = 0) uniform sampler2D s_Irradiance;
layout(set = 1, binding = 1) uniform sampler2D s_Depth;

// ------------------------------------------------------------------------
// PUSH CONSTANTS ---------------------------------------------------------
// ------------------------------------------------------------------------

layout(push_constant) uniform PushConstants
{
    vec3  grid_start_position;
    vec3  grid_step;
    ivec3 probe_counts;
    float scale;
    int   probe_side_length;
    int   texture_width;
    int   texture_height;
}
u_PushConstants;

// ------------------------------------------------------------------------
// FUNCTIONS --------------------------------------------------------------
// ------------------------------------------------------------------------

float sign_not_zero(in float k)
{
    return (k >= 0.0) ? 1.0 : -1.0;
}

// ------------------------------------------------------------------

vec2 sign_not_zero(in vec2 v)
{
    return vec2(sign_not_zero(v.x), sign_not_zero(v.y));
}

// ------------------------------------------------------------------

vec2 oct_encode(in vec3 v) 
{
    float l1norm = abs(v.x) + abs(v.y) + abs(v.z);
    vec2 result = v.xy * (1.0 / l1norm);
    if (v.z < 0.0)
        result = (1.0 - abs(result.yx)) * sign_not_zero(result.xy);
    return result;
}

// ------------------------------------------------------------------

vec2 texture_coord_from_direction(vec3 dir, int probe_index, int full_texture_width, int full_texture_height, int probe_side_length) 
{
    vec2 normalized_oct_coord = oct_encode(normalize(dir));
    vec2 normalized_oct_coord_zero_one = (normalized_oct_coord + vec2(1.0f)) * 0.5f;

    // Length of a probe side, plus one pixel on each edge for the border
    float probe_with_border_side = float(probe_side_length) + 2.0f;

    vec2 oct_coord_normalized_to_texture_dimensions = (normalized_oct_coord_zero_one * float(probe_side_length)) / vec2(float(full_texture_width), float(full_texture_height));

    int probes_per_row = (full_texture_width - 2) / int(probe_with_border_side);

    // Add (2,2) back to texCoord within larger texture. Compensates for 1 pix 
    // border around texture and further 1 pix border around top left probe.
    vec2 probe_top_left_position = vec2(mod(probe_index, probes_per_row) * probe_with_border_side,
        (probe_index / probes_per_row) * probe_with_border_side) + vec2(2.0f, 2.0f);

    vec2 normalized_probe_top_left_position = vec2(probe_top_left_position) / vec2(float(full_texture_width), float(full_texture_height));

    return vec2(normalized_probe_top_left_position + oct_coord_normalized_to_texture_dimensions);
}

// ------------------------------------------------------------------------
// MAIN -------------------------------------------------------------------
// ------------------------------------------------------------------------

void main()
{
    vec2 probe_coord = texture_coord_from_direction(normalize(FS_IN_Normal), 
                                                    FS_IN_ProbeIdx, 
                                                    u_PushConstants.texture_width, 
                                                    u_PushConstants.texture_height, 
                                                    u_PushConstants.probe_side_length);

    FS_OUT_Color = vec4(textureLod(s_Irradiance, probe_coord, 0.0f).rgb, 1.0f);
}

// ------------------------------------------------------------------------