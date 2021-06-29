// ------------------------------------------------------------------
// DEFINES ----------------------------------------------------------
// ------------------------------------------------------------------

#define NUM_THREADS_X 32
#define NUM_THREADS_Y 32
#if defined(DEPTH_PROBE_UPDATE)
    #define TEXTURE_WIDTH ddgi.depth_texture_width
    #define TEXTURE_HEIGHT ddgi.depth_texture_height
    #define PROBE_SIDE_LENGTH ddgi.depth_probe_side_length
#else
    #define TEXTURE_WIDTH ddgi.irradiance_texture_width
    #define TEXTURE_HEIGHT ddgi.irradiance_texture_height
    #define PROBE_SIDE_LENGTH ddgi.irradiance_probe_side_length
#endif

// ------------------------------------------------------------------
// INPUTS -----------------------------------------------------------
// ------------------------------------------------------------------

layout(local_size_x = NUM_THREADS_X, local_size_y = NUM_THREADS_Y, local_size_z = 1) in;

// ------------------------------------------------------------------
// DESCRIPTOR SETS --------------------------------------------------
// ------------------------------------------------------------------

layout(set = 0, binding = 0, rgba16f) uniform image2D i_OutputIrradiance;
layout(set = 0, binding = 1, rg16f) uniform image2D i_OutputDepth;

layout(set = 1, binding = 0) uniform sampler2D s_InputIrradiance;
layout(set = 1, binding = 1) uniform sampler2D s_InputDepth;
layout(set = 1, binding = 2, std140) uniform DDGIUBO
{
    DDGIUniforms ddgi; 
};

layout(set = 2, binding = 0) uniform sampler2D s_InputRadiance;
layout(set = 2, binding = 1) uniform sampler2D s_InputDirectionDepth;

// ------------------------------------------------------------------
// CONSTANTS --------------------------------------------------------
// ------------------------------------------------------------------

const float FLT_EPS = 0.00000001;

// ------------------------------------------------------------------
// FUNCTIONS --------------------------------------------------------
// ------------------------------------------------------------------

bool is_border_texel()
{
    if (gl_GlobalInvocationID.x == 0 || gl_GlobalInvocationID.y == 0)
        return true;

    if (gl_GlobalInvocationID.x == (TEXTURE_WIDTH - 1) || gl_GlobalInvocationID.y == (TEXTURE_HEIGHT - 1))
        return true;

    int probe_with_border_side = PROBE_SIDE_LENGTH + 2;

    if ((gl_GlobalInvocationID.x % probe_with_border_side) == 0 || (gl_GlobalInvocationID.x % probe_with_border_side) == 1)
        return true;

    if ((gl_GlobalInvocationID.y % probe_with_border_side) == 0 || (gl_GlobalInvocationID.y % probe_with_border_side) == 1)
        return true;

    return false;
}

// ------------------------------------------------------------------
// MAIN -------------------------------------------------------------
// ------------------------------------------------------------------

void main()
{
    const ivec2 current_coord = ivec2(gl_GlobalInvocationID.xy);

    if (!is_border_texel())
    {
        const int   relative_probe_id   = probe_id(current_coord, TEXTURE_WIDTH, PROBE_SIDE_LENGTH);
        const float energy_conservation = 0.95f;

        vec3  result       = vec3(0.0f);
        float total_weight = 0.0f;

        // For each ray
        for (int r = 0; r < ddgi.rays_per_probe; ++r)
        {
            ivec2 C = ivec2(r, relative_probe_id);

            vec4 ray_direction_depth = texelFetch(s_InputDirectionDepth, C, 0);

            vec3  ray_direction      = ray_direction_depth.xyz;
            vec3  ray_hit_radiance   = texelFetch(s_InputRadiance, C, 0).xyz * energy_conservation;

#if defined(DEPTH_PROBE_UPDATE)            
            float ray_probe_distance = min(ddgi.max_distance, ray_direction_depth.w - 0.01f);
            
            // Detect misses and force depth
            if (ray_probe_distance == -1.0f)
                ray_probe_distance = ddgi.max_distance;
#endif

            vec3 texel_direction = oct_decode(normalized_oct_coord(current_coord, PROBE_SIDE_LENGTH));

            float weight = 0.0f;

#if defined(DEPTH_PROBE_UPDATE)  
            weight = pow(max(0.0, dot(texel_direction, ray_direction)), ddgi.depth_sharpness);
#else
            weight = max(0.0, dot(texel_direction, ray_direction));
#endif

            if (weight >= FLT_EPS)
            {
#if defined(DEPTH_PROBE_UPDATE) 
                result += vec3(ray_probe_distance * weight, square(ray_probe_distance) * weight, 0.0);
#else
                result += vec3(ray_hit_radiance * weight);
#endif                
                    
                total_weight += weight;
            }
        }

        if (total_weight > FLT_EPS)
            result /= total_weight;

        // Temporal Accumulation
        vec3 prev_result;

#if defined(DEPTH_PROBE_UPDATE)
        prev_result = texelFetch(s_InputDepth, current_coord, 0).rgb;
#else
        prev_result = texelFetch(s_InputIrradiance, current_coord, 0).rgb;
#endif
            
        result = mix(result, prev_result, ddgi.hysteresis);

#if defined(DEPTH_PROBE_UPDATE)
        imageStore(i_OutputDepth, current_coord, vec4(result, 1.0));
#else
        imageStore(i_OutputIrradiance, current_coord, vec4(result, 1.0));
#endif

        // Update borders
        const ivec2 coord_without_outer_border = current_coord - ivec2(1);
        const ivec2 probe_coord                = coord_without_outer_border % ivec2(PROBE_SIDE_LENGTH + 2);
        const ivec2 probe_coord_without_border = probe_coord - ivec2(1);

        // Top row
        if (probe_coord_without_border.y == 0)
        {
            const ivec2 column_start_coord = current_coord - probe_coord_without_border - ivec2(1, 0);
            const ivec2 border_texel       = column_start_coord + ivec2((PROBE_SIDE_LENGTH - probe_coord_without_border.x), -1);
    
#if defined(DEPTH_PROBE_UPDATE)
            imageStore(i_OutputDepth, border_texel, vec4(result, 1.0));
#else
            imageStore(i_OutputIrradiance, border_texel, vec4(result, 1.0));
#endif
        }

        // Bottom row
        if (probe_coord_without_border.y == (PROBE_SIDE_LENGTH - 1))
        {
            const ivec2 column_start_coord = current_coord - probe_coord_without_border + ivec2(0, PROBE_SIDE_LENGTH - 1) - ivec2(1, 0);
            const ivec2 border_texel       = column_start_coord + ivec2((PROBE_SIDE_LENGTH - probe_coord_without_border.x), 1);
    
#if defined(DEPTH_PROBE_UPDATE)
            imageStore(i_OutputDepth, border_texel, vec4(result, 1.0));
#else
            imageStore(i_OutputIrradiance, border_texel, vec4(result, 1.0));
#endif
        }

        // Left column
        if (probe_coord_without_border.x == 0)
        {
            const ivec2 column_start_coord = current_coord - probe_coord_without_border - ivec2(0, 1);
            const ivec2 border_texel       = column_start_coord + ivec2(-1, (PROBE_SIDE_LENGTH - probe_coord_without_border.y));
    
#if defined(DEPTH_PROBE_UPDATE)
            imageStore(i_OutputDepth, border_texel, vec4(result, 1.0));
#else
            imageStore(i_OutputIrradiance, border_texel, vec4(result, 1.0));
#endif
        }

        // Right column
        if (probe_coord_without_border.x == (PROBE_SIDE_LENGTH - 1))
        {
            const ivec2 column_start_coord = current_coord - probe_coord_without_border + ivec2(PROBE_SIDE_LENGTH - 1, 0) - ivec2(0, 1);
            const ivec2 border_texel       = column_start_coord + ivec2(1, (PROBE_SIDE_LENGTH - probe_coord_without_border.y));
    
#if defined(DEPTH_PROBE_UPDATE)
            imageStore(i_OutputDepth, border_texel, vec4(result, 1.0));
#else
            imageStore(i_OutputIrradiance, border_texel, vec4(result, 1.0));
#endif
        }

        // Top left corners
        if (probe_coord_without_border.x == 0 && probe_coord_without_border.y == 0)
        {
            const ivec2 border_texel = current_coord + ivec2(PROBE_SIDE_LENGTH);
    
#if defined(DEPTH_PROBE_UPDATE)
            imageStore(i_OutputDepth, border_texel, vec4(result, 1.0));
#else
            imageStore(i_OutputIrradiance, border_texel, vec4(result, 1.0));
#endif
        }

        // Top right corner
        if (probe_coord_without_border.x == (PROBE_SIDE_LENGTH - 1) && probe_coord_without_border.y == 0)
        {
            const ivec2 border_texel = current_coord + ivec2(-PROBE_SIDE_LENGTH, PROBE_SIDE_LENGTH);
    
#if defined(DEPTH_PROBE_UPDATE)
            imageStore(i_OutputDepth, border_texel, vec4(result, 1.0));
#else
            imageStore(i_OutputIrradiance, border_texel, vec4(result, 1.0));
#endif
        }

        // Bottom left corner
        if (probe_coord_without_border.x == 0 && probe_coord_without_border.y == (PROBE_SIDE_LENGTH - 1))
        {
            const ivec2 border_texel = current_coord + ivec2(PROBE_SIDE_LENGTH, -PROBE_SIDE_LENGTH);
    
#if defined(DEPTH_PROBE_UPDATE)
            imageStore(i_OutputDepth, border_texel, vec4(result, 1.0));
#else
            imageStore(i_OutputIrradiance, border_texel, vec4(result, 1.0));
#endif
        }

        // Bottom right corner
        if (probe_coord_without_border.x == (PROBE_SIDE_LENGTH - 1) && probe_coord_without_border.y == (PROBE_SIDE_LENGTH - 1))
        {
            const ivec2 border_texel = current_coord + ivec2(-PROBE_SIDE_LENGTH);
    
#if defined(DEPTH_PROBE_UPDATE)
            imageStore(i_OutputDepth, border_texel, vec4(result, 1.0));
#else
            imageStore(i_OutputIrradiance, border_texel, vec4(result, 1.0));
#endif
        }
    }
}

// ------------------------------------------------------------------