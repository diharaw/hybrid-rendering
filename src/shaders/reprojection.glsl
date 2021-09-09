#ifndef REPROJECTION_GLSL
#define REPROJECTION_GLSL

// ------------------------------------------------------------------------

#define NORMAL_DISTANCE 0.1f
#define PLANE_DISTANCE 5.0f

// ------------------------------------------------------------------------

bool plane_distance_disocclusion_check(vec3 current_pos, vec3 history_pos, vec3 current_normal)
{
    vec3  to_current    = current_pos - history_pos;
    float dist_to_plane = abs(dot(to_current, current_normal));

    return dist_to_plane > PLANE_DISTANCE;
}

// ------------------------------------------------------------------------

bool out_of_frame_disocclusion_check(ivec2 coord, ivec2 image_dim)
{
    // check whether reprojected pixel is inside of the screen
    if (any(lessThan(coord, ivec2(0, 0))) || any(greaterThan(coord, image_dim - ivec2(1, 1))))
        return true;
    else
        return false;
}

// ------------------------------------------------------------------------

bool mesh_id_disocclusion_check(float mesh_id, float mesh_id_prev)
{
    if (mesh_id == mesh_id_prev)
        return false;
    else
        return true;
}

// ------------------------------------------------------------------------

bool normals_disocclusion_check(vec3 current_normal, vec3 history_normal)
{
    if (pow(abs(dot(current_normal, history_normal)), 2) > NORMAL_DISTANCE)
        return false;
    else
        return true;
}

// ------------------------------------------------------------------------

bool is_reprojection_valid(ivec2 coord, vec3 current_pos, vec3 history_pos, vec3 current_normal, vec3 history_normal, float current_mesh_id, float history_mesh_id, ivec2 image_dim)
{
    // check if the history sample is within the frame
    if (out_of_frame_disocclusion_check(coord, image_dim)) return false;

    // check if the history belongs to the same surface
    if (mesh_id_disocclusion_check(current_mesh_id, history_mesh_id)) return false;

    // check if history sample is on the same plane
    if (plane_distance_disocclusion_check(current_pos, history_pos, current_normal)) return false;

    // check normals for compatibility
    if (normals_disocclusion_check(current_normal, history_normal)) return false;

    return true;
}

// ------------------------------------------------------------------

vec2 surface_point_reprojection(ivec2 coord, vec2 motion_vector, ivec2 size)
{
    return vec2(coord) + motion_vector.xy * vec2(size);
}

// ------------------------------------------------------------------

vec2 virtual_point_reprojection(ivec2 current_coord, ivec2 size, float depth, float ray_length, vec3 cam_pos, mat4 view_proj_inverse, mat4 prev_view_proj)
{
    const vec2 tex_coord  = current_coord / vec2(size);
    vec3       ray_origin = world_position_from_depth(tex_coord, depth, view_proj_inverse);

    vec3 camera_ray = ray_origin - cam_pos.xyz;

    float camera_ray_length     = length(camera_ray);
    float reflection_ray_length = ray_length;

    camera_ray = normalize(camera_ray);

    vec3 parallax_hit_point = cam_pos.xyz + camera_ray * (camera_ray_length + reflection_ray_length);

    vec4 reprojected_parallax_hit_point = prev_view_proj * vec4(parallax_hit_point, 1.0f);

    reprojected_parallax_hit_point.xy /= reprojected_parallax_hit_point.w;

    return (reprojected_parallax_hit_point.xy * 0.5f + 0.5f) * vec2(size);
}

// ------------------------------------------------------------------------

vec2 compute_history_coord(ivec2 current_coord, ivec2 size, float depth, vec2 motion, float curvature, float ray_length, vec3 cam_pos, mat4 view_proj_inverse, mat4 prev_view_proj)
{
    const vec2 surface_history_coord = surface_point_reprojection(current_coord, motion, size);

    vec2 history_coord = surface_history_coord;

    if (ray_length > 0.0f && curvature == 0.0f)
        history_coord = virtual_point_reprojection(current_coord, size, depth, ray_length, cam_pos, view_proj_inverse, prev_view_proj);

    return history_coord;
}

// ------------------------------------------------------------------

bool reproject(in ivec2 frag_coord, 
               in float depth, 
               in int g_buffer_mip,
            #if defined(REPROJECTION_REFLECTIONS)
               in vec3 cam_pos,
            #endif               
               in mat4 view_proj_inverse,
            #if defined(REPROJECTION_REFLECTIONS)
               in mat4 prev_view_proj,
               in float ray_length,
            #endif   
               in sampler2D sampler_gbuffer_2,
               in sampler2D sampler_gbuffer_3,
               in sampler2D sampler_prev_gbuffer_2,
               in sampler2D sampler_prev_gbuffer_3,
               in sampler2D sampler_prev_gbuffer_depth,
               in sampler2D sampler_history_output,
            #if defined(REPROJECTION_MOMENTS)               
               in sampler2D sampler_history_moments_length,
            #else 
               in sampler2D sampler_history_length,
            #endif
            #if defined(REPROJECTION_SINGLE_COLOR_CHANNEL)
               out float history_color,
            #else
               out vec3  history_color, 
            #endif
            #if defined(REPROJECTION_MOMENTS)
               out vec2 history_moments, 
            #endif
               out float history_length)
{
    const vec2  image_dim    = vec2(textureSize(sampler_history_output, 0));
    const vec2  pixel_center = vec2(frag_coord) + vec2(0.5f);
    const vec2  tex_coord    = pixel_center / vec2(image_dim);

    const vec4 center_g_buffer_2 = texelFetch(sampler_gbuffer_2, frag_coord, g_buffer_mip);
    const vec4 center_g_buffer_3 = texelFetch(sampler_gbuffer_3, frag_coord, g_buffer_mip);

    const vec2  current_motion  = center_g_buffer_2.zw;
    const vec3  current_normal  = octohedral_to_direction(center_g_buffer_2.xy);
    const float current_mesh_id = center_g_buffer_3.z;
    const vec3  current_pos     = world_position_from_depth(tex_coord, depth, view_proj_inverse);

#if defined(REPROJECTION_REFLECTIONS)
    const float curvature = center_g_buffer_3.g;
    const vec2 history_tex_coord = tex_coord + current_motion;
    const vec2 reprojected_coord = compute_history_coord(frag_coord, 
                                                         ivec2(image_dim), 
                                                         depth, 
                                                         current_motion, 
                                                         curvature, 
                                                         ray_length, 
                                                         cam_pos, 
                                                         view_proj_inverse, 
                                                         prev_view_proj);   
    const ivec2 history_coord = ivec2(reprojected_coord);                                                      
    const vec2  history_coord_floor = floor(reprojected_coord);                                                                                                         
#else 
    // +0.5 to account for texel center offset
    const ivec2 history_coord = ivec2(vec2(frag_coord) + current_motion.xy * image_dim + vec2(0.5f));
    const vec2  history_coord_floor = floor(frag_coord.xy) + current_motion.xy * image_dim;
    const vec2  history_tex_coord = tex_coord + current_motion.xy;
#endif

#if defined(REPROJECTION_SINGLE_COLOR_CHANNEL)
    history_color = 0.0f;
#else
    history_color   = vec3(0.0f);
#endif
#if defined(REPROJECTION_MOMENTS)    
    history_moments = vec2(0.0f);
#endif

    bool        v[4];
    const ivec2 offset[4] = { ivec2(0, 0), ivec2(1, 0), ivec2(0, 1), ivec2(1, 1) };

    // check for all 4 taps of the bilinear filter for validity
    bool valid = false;
    for (int sample_idx = 0; sample_idx < 4; sample_idx++)
    {
        ivec2 loc = ivec2(history_coord_floor) + offset[sample_idx];

        vec4  sample_g_buffer_2 = texelFetch(sampler_prev_gbuffer_2, loc, g_buffer_mip);
        vec4  sample_g_buffer_3 = texelFetch(sampler_prev_gbuffer_3, loc, g_buffer_mip);
        float sample_depth      = texelFetch(sampler_prev_gbuffer_depth, loc, g_buffer_mip).r;

        vec3  history_normal  = octohedral_to_direction(sample_g_buffer_2.xy);
        float history_mesh_id = sample_g_buffer_3.z;
        vec3  history_pos     = world_position_from_depth(history_tex_coord, sample_depth, view_proj_inverse);

        v[sample_idx] = is_reprojection_valid(history_coord, current_pos, history_pos, current_normal, history_normal, current_mesh_id, history_mesh_id, ivec2(image_dim));

        valid = valid || v[sample_idx];
    }

    if (valid)
    {
        float sumw = 0;
        float x    = fract(history_coord_floor.x);
        float y    = fract(history_coord_floor.y);

        // bilinear weights
        float w[4] = { (1 - x) * (1 - y),
                       x * (1 - y),
                       (1 - x) * y,
                       x * y };

    #if defined(REPROJECTION_SINGLE_COLOR_CHANNEL)
        history_color = 0.0f;
    #else
        history_color = vec3(0.0f);
    #endif
    #if defined(REPROJECTION_MOMENTS)         
        history_moments = vec2(0.0f);
    #endif

        // perform the actual bilinear interpolation
        for (int sample_idx = 0; sample_idx < 4; sample_idx++)
        {
            ivec2 loc = ivec2(history_coord_floor) + offset[sample_idx];

            if (v[sample_idx])
            {
            #if defined(REPROJECTION_SINGLE_COLOR_CHANNEL)
                history_color += w[sample_idx] * texelFetch(sampler_history_output, loc, 0).r;
            #else
                history_color += w[sample_idx] * texelFetch(sampler_history_output, loc, 0).rgb;
            #endif            
            #if defined(REPROJECTION_MOMENTS) 
                history_moments += w[sample_idx] * texelFetch(sampler_history_moments_length, loc, 0).rg;
            #endif                
                sumw += w[sample_idx];
            }
        }

        // redistribute weights in case not all taps were used
        valid           = (sumw >= 0.01);
    #if defined(REPROJECTION_SINGLE_COLOR_CHANNEL)
        history_color   = valid ? history_color / sumw : 0.0f;
    #else
        history_color   = valid ? history_color / sumw : vec3(0.0f);
    #endif    
    #if defined(REPROJECTION_MOMENTS)         
        history_moments = valid ? history_moments / sumw : vec2(0.0f);
    #endif
    }
    if (!valid) // perform cross-bilateral filter in the hope to find some suitable samples somewhere
    {
        float cnt = 0.0;

        // this code performs a binary descision for each tap of the cross-bilateral filter
        const int radius = 1;
        for (int yy = -radius; yy <= radius; yy++)
        {
            for (int xx = -radius; xx <= radius; xx++)
            {
                ivec2 p = history_coord + ivec2(xx, yy);

                vec4  sample_g_buffer_2 = texelFetch(sampler_prev_gbuffer_2, p, g_buffer_mip);
                vec4  sample_g_buffer_3 = texelFetch(sampler_prev_gbuffer_3, p, g_buffer_mip);
                float sample_depth      = texelFetch(sampler_prev_gbuffer_depth, p, g_buffer_mip).r;

                vec3  history_normal  = octohedral_to_direction(sample_g_buffer_2.xy);
                float history_mesh_id = sample_g_buffer_3.z;
                vec3  history_pos     = world_position_from_depth(history_tex_coord, sample_depth, view_proj_inverse);

                if (is_reprojection_valid(history_coord, current_pos, history_pos, current_normal, history_normal, current_mesh_id, history_mesh_id, ivec2(image_dim)))
                {
                #if defined(REPROJECTION_SINGLE_COLOR_CHANNEL)
                    history_color += texelFetch(sampler_history_output, p, 0).r;
                #else
                    history_color += texelFetch(sampler_history_output, p, 0).rgb;
                #endif
                #if defined(REPROJECTION_MOMENTS) 
                    history_moments += texelFetch(sampler_history_moments_length, p, 0).rg;
                #endif                    
                    cnt += 1.0;
                }
            }
        }
        if (cnt > 0)
        {
            valid = true;
            history_color /= cnt;
        #if defined(REPROJECTION_MOMENTS)             
            history_moments /= cnt;
        #endif            
        }
    }

    if (valid)
    {
    #if defined(REPROJECTION_MOMENTS)        
        history_length = texelFetch(sampler_history_moments_length, history_coord, 0).b;
    #else
        history_length = texelFetch(sampler_history_length, history_coord, 0).r;
    #endif        
    }    
    else
    {
    #if defined(REPROJECTION_SINGLE_COLOR_CHANNEL)
        history_color = 0.0f;
    #else        
        history_color = vec3(0.0f);
    #endif
    #if defined(REPROJECTION_MOMENTS)          
        history_moments = vec2(0.0f);
    #endif
        history_length  = 0.0f;
    }

    return valid;
}

// ------------------------------------------------------------------

#endif