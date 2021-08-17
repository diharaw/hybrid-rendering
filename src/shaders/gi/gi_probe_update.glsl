// ------------------------------------------------------------------
// DEFINES ----------------------------------------------------------
// ------------------------------------------------------------------

#define CACHE_SIZE 64

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
// SHARED MEMORY ----------------------------------------------------
// ------------------------------------------------------------------

shared vec4 g_ray_direction_depth[CACHE_SIZE];
#if !defined(DEPTH_PROBE_UPDATE) 
shared vec3 g_ray_hit_radiance[CACHE_SIZE];
#endif

// ------------------------------------------------------------------
// CONSTANTS --------------------------------------------------------
// ------------------------------------------------------------------

const float FLT_EPS = 0.00000001;

// ------------------------------------------------------------------
// FUNCTIONS --------------------------------------------------------
// ------------------------------------------------------------------

void populate_cache(int relative_probe_id, uint offset, uint num_rays)
{
    if (gl_LocalInvocationIndex < num_rays)
    {
        ivec2 C = ivec2(offset + uint(gl_LocalInvocationIndex), relative_probe_id);

        g_ray_direction_depth[gl_LocalInvocationIndex] = texelFetch(s_InputDirectionDepth, C, 0);
    #if !defined(DEPTH_PROBE_UPDATE) 
        g_ray_hit_radiance[gl_LocalInvocationIndex] = texelFetch(s_InputRadiance, C, 0).xyz;
    #endif 
    }
}

// ------------------------------------------------------------------

void gather_rays(ivec2 current_coord, uint num_rays, inout vec3 result, inout float total_weight)
{
    const float energy_conservation = 0.95f;

    // For each ray
    for (int r = 0; r < num_rays; ++r)
    {
        vec4 ray_direction_depth = g_ray_direction_depth[r];

        vec3  ray_direction      = ray_direction_depth.xyz;

#if defined(DEPTH_PROBE_UPDATE)            
        float ray_probe_distance = min(ddgi.max_distance, ray_direction_depth.w - 0.01f);
            
        // Detect misses and force depth
        if (ray_probe_distance == -1.0f)
            ray_probe_distance = ddgi.max_distance;
#else        
        vec3  ray_hit_radiance   = g_ray_hit_radiance[r] * energy_conservation;
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
}

// ------------------------------------------------------------------
// MAIN -------------------------------------------------------------
// ------------------------------------------------------------------

void main()
{
    const ivec2 current_coord = ivec2(gl_GlobalInvocationID.xy) + (ivec2(gl_WorkGroupID.xy) * ivec2(2)) + ivec2(2);

    const int   relative_probe_id   = probe_id(current_coord, TEXTURE_WIDTH, PROBE_SIDE_LENGTH);
    
    vec3  result       = vec3(0.0f);
    float total_weight = 0.0f;

    uint remaining_rays = ddgi.rays_per_probe;
    uint offset = 0;

    while (remaining_rays > 0)
    {
        uint num_rays = min(CACHE_SIZE, remaining_rays);
        
        populate_cache(relative_probe_id, offset, num_rays);

        barrier();

        gather_rays(current_coord, num_rays, result, total_weight);

        barrier();

        remaining_rays -= num_rays;
        offset += num_rays;
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