// ------------------------------------------------------------------
// DEFINES ----------------------------------------------------------
// ------------------------------------------------------------------

#if defined(DEPTH_PROBE_UPDATE)
    #define NUM_THREADS_X 16
    #define NUM_THREADS_Y 16
    #define TEXTURE_WIDTH ddgi.depth_texture_width
    #define TEXTURE_HEIGHT ddgi.depth_texture_height
    #define PROBE_SIDE_LENGTH ddgi.depth_probe_side_length
#else
    #define NUM_THREADS_X 8
    #define NUM_THREADS_Y 8
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
layout(set = 1, binding = 2, scalar) uniform DDGIUBO
{
    DDGIUniforms ddgi; 
};

layout(set = 2, binding = 0) uniform sampler2D s_InputRadiance;
layout(set = 2, binding = 1) uniform sampler2D s_InputDirectionDepth;

// ------------------------------------------------------------------------
// PUSH CONSTANTS ---------------------------------------------------------
// ------------------------------------------------------------------------

layout(push_constant) uniform PushConstants
{
    uint first_frame;
}
u_PushConstants;

// ------------------------------------------------------------------
// CONSTANTS --------------------------------------------------------
// ------------------------------------------------------------------

const float FLT_EPS = 0.00000001;

// ------------------------------------------------------------------
// MAIN -------------------------------------------------------------
// ------------------------------------------------------------------

void main()
{
    const ivec2 current_coord = ivec2(gl_GlobalInvocationID.xy);

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
            
    if (u_PushConstants.first_frame == 0)            
        result = mix(result, prev_result, ddgi.hysteresis);

#if defined(DEPTH_PROBE_UPDATE)
    imageStore(i_OutputDepth, current_coord, vec4(result, 1.0));
#else
    imageStore(i_OutputIrradiance, current_coord, vec4(result, 1.0));
#endif
}

// ------------------------------------------------------------------