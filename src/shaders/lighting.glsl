#ifndef LIGHTING_GLSL
#define LIGHTING_GLSL

// ------------------------------------------------------------------------

void fetch_light_properties(
    in Light light,
#if !defined(SHADOW_RAY_ONLY)    
    in vec3 Wo,
#endif    
    in vec3 P,
    in vec3 N,
#if defined(SOFT_SHADOWS)
    in vec2 rng,
#endif 
#if !defined(SHADOW_RAY_ONLY)
    out vec3 Li,
#endif    
    out vec3 Wi,
#if !defined(SHADOW_RAY_ONLY)
    out vec3 Wh,
#endif 
#if defined(RAY_TRACING)
    out float t_max,
#endif    
    out float attenuation
)
{
    const int type = light_type(light);

#if !defined(SHADOW_RAY_ONLY)
    Li = light_color(light) * light_intensity(light);  
#endif

    if (type == LIGHT_TYPE_DIRECTIONAL)
    {
        vec3 light_dir       = light_direction(light);

#if defined(SOFT_SHADOWS)
        vec3 light_tangent   = normalize(cross(light_dir, vec3(0.0f, 1.0f, 0.0f)));
        vec3 light_bitangent = normalize(cross(light_tangent, light_dir));

        // calculate disk point
        float point_radius = light_radius(light) * sqrt(rng.x);
        float point_angle  = rng.y * 2.0f * M_PI;
        vec2  disk_point   = vec2(point_radius * cos(point_angle), point_radius * sin(point_angle));    
        Wi = normalize(light_dir + disk_point.x * light_tangent + disk_point.y * light_bitangent);
#else
        Wi = light_dir;
#endif
#if defined(RAY_TRACING)
        t_max = 10000.0f;
#endif  
        attenuation = 1.0f;
    }
    else if (type == LIGHT_TYPE_POINT)
    {        
        vec3  to_light       = light_position(light) - P;
        vec3  light_dir       = normalize(to_light);
        float light_distance = length(to_light);

#if defined(SOFT_SHADOWS)
        vec3 light_tangent   = normalize(cross(light_dir, vec3(0.0f, 1.0f, 0.0f)));
        vec3 light_bitangent = normalize(cross(light_tangent, light_dir));  
        
        // calculate disk point
        float current_light_radius = light_radius(light) / light_distance;  
        float point_radius = current_light_radius * sqrt(rng.x);
        float point_angle  = rng.y * 2.0f * M_PI;
        vec2  disk_point   = vec2(point_radius * cos(point_angle), point_radius * sin(point_angle));    
        Wi = normalize(light_dir + disk_point.x * light_tangent + disk_point.y * light_bitangent);
#else
        Wi = light_dir;
#endif
#if defined(RAY_TRACING)
        t_max = light_distance;
#endif  
        attenuation = (1.0f / (light_distance * light_distance));
    }
    else
    {
        vec3  to_light       = light_position(light) - P;
        vec3  light_dir      = normalize(to_light);
        float light_distance = length(to_light);

#if defined(SOFT_SHADOWS)    
        vec3 light_tangent   = normalize(cross(light_dir, vec3(0.0f, 1.0f, 0.0f)));
        vec3 light_bitangent = normalize(cross(light_tangent, light_dir));  
        
        // calculate disk point
        float current_light_radius = light_radius(light) / light_distance;  
        float point_radius = current_light_radius * sqrt(rng.x);
        float point_angle  = rng.y * 2.0f * M_PI;
        vec2  disk_point   = vec2(point_radius * cos(point_angle), point_radius * sin(point_angle));    
        Wi = normalize(light_dir + disk_point.x * light_tangent + disk_point.y * light_bitangent); 
#else
        Wi = light_dir;
#endif       
#if defined(RAY_TRACING)
        t_max = light_distance;
#endif  
        float angle_attenuation = dot(Wi, light_direction(light));
        angle_attenuation       = smoothstep(light_cos_theta_outer(light), light_cos_theta_inner(light), angle_attenuation);    
        attenuation = (angle_attenuation / (light_distance * light_distance));    
    }

#if !defined(SHADOW_RAY_ONLY)
    Wh  = normalize(Wo + Wi); 
#endif     
    attenuation *= clamp(dot(N, Wi), 0.0, 1.0);
}

// ------------------------------------------------------------------------

#if !defined(SHADOW_RAY_ONLY)

vec3 direct_lighting(
    in Light light,   
    in vec3 Wo, 
    in vec3 N, 
    in vec3 P, 
    in vec3 F0, 
    in vec3 diffuse_color, 
    in float roughness
#if defined(RAY_THROUGHPUT)
  , in vec3 T
#endif    
#if defined(SOFT_SHADOWS)
  , in vec2 rng1
#endif
#if defined(SAMPLE_SKY_LIGHT)
  , in vec2 rng2
  , in samplerCube sky_cubemap
#endif
)
{
    vec3 Lo = vec3(0.0f);

#if !defined(RAY_THROUGHPUT)
    vec3 T = vec3(1.0f);
#endif

    vec3 ray_origin = P + N * 0.1f;

    // Punctual Light
    {
        vec3 Li;
        vec3 Wi;
        vec3 Wh;
    #if defined(RAY_TRACING)    
        float t_max;
    #endif    
        float attenuation; 

        fetch_light_properties(light, 
                               Wo, 
                               P, 
                               N, 
                            #if defined(SOFT_SHADOWS)
                               rng1,
                            #endif 
                               Li, 
                               Wi, 
                               Wh, 
                            #if defined(RAY_TRACING)
                               t_max, 
                            #endif
                               attenuation);

#if defined(RAY_TRACING)
        if (attenuation > 0.0f)
            attenuation *= query_distance(ray_origin, Wi, t_max);
#endif 

        vec3  brdf      = evaluate_uber_brdf(diffuse_color, roughness, N, F0, Wo, Wh, Wi);
        Lo += T * brdf * attenuation * Li;  
    }

#if defined(SAMPLE_SKY_LIGHT)
    // Sky Light
    {
        vec3  Wi  = sample_cosine_lobe(N, rng2);
        vec3  Li  = texture(sky_cubemap, Wi).rgb;
        float pdf = pdf_cosine_lobe(dot(N, Wi));
        vec3  Wh  = normalize(Wo + Wi);

        // fire shadow ray for visiblity
        Li *= query_distance(ray_origin, Wi, 10000.0f);

        vec3  brdf = evaluate_uber_brdf(diffuse_color, roughness, N, F0, Wo, Wh, Wi);
        Lo += T * brdf * Li;  
    }
#endif    

    return Lo;
}

#endif

// ------------------------------------------------------------------------

#endif