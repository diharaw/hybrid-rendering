#ifndef BRDF_GLSL
#define BRDF_GLSL

#include "sampling.glsl"

vec3 sample_lambert(in vec3 n, in vec2 r)
{
    return sample_cosine_lobe(n, r);
}

vec3 evaluate_lambert(in vec3 albedo)
{
    return albedo / M_PI;
}

float pdf_lambert(in float ndotl)
{
    return ndotl / M_PI;
}

float D_ggx(in float ndoth, in float alpha)
{
    float a2 = alpha * alpha;
    float denom = (ndoth * ndoth) * (a2 - 1.0) + 1.0;

    return a2 / max(EPSILON, (M_PI * denom * denom));
}

float G1_schlick_ggx(in float roughness, in float ndotv)
{
    float k = ((roughness + 1) * (roughness + 1)) / 8.0;

    return ndotv / max(EPSILON, (ndotv * (1 - k) + k));
}

float G_schlick_ggx(in float ndotl, in float ndotv, in float roughness) 
{
    return G1_schlick_ggx(roughness, ndotl) * G1_schlick_ggx(roughness, ndotv);
}

vec3 F_schlick(in vec3 f0, in float vdoth)
{
    return f0 + (vec3(1.0) - f0) * (pow(1.0 - vdoth, 5.0)); 
}

vec3 sample_ggx(in vec3 n, in float alpha, in vec2 Xi)
{
    float phi = 2.0 * M_PI * Xi.x;
    float cos_theta = sqrt((1.0 - Xi.y) / (1.0 + (alpha * alpha - 1.0) * Xi.y));
    float sin_theta = sqrt(1.0 - cos_theta * cos_theta);

    vec3 d;

    d.x = sin_theta * cos(phi);
    d.y = sin_theta * sin(phi);
    d.z = cos_theta;

    return normalize(make_rotation_matrix(n) * d);
}

vec3 evaluate_ggx(in SurfaceProperties p, in vec3 F, in float ndoth, in float ndotl, in float ndotv)
{
    float alpha = p.roughness * p.roughness;
    return (D_ggx(ndoth, alpha) * F * G_schlick_ggx(ndotl, ndotv, p.roughness)) / max(EPSILON, (4.0 * ndotl * ndotv)); 
}

float pdf_D_ggx(in float alpha, in float ndoth, in float vdoth)
{
    return D_ggx(ndoth, alpha) * ndoth / max(EPSILON, (4.0 * vdoth));
}

vec3 evaluate_uber(in SurfaceProperties p, in vec3 Wo, in vec3 Wh, in vec3 Wi)
{
    float NdotL = max(dot(p.normal, Wi), 0.0);
    float NdotV = max(dot(p.normal, Wo), 0.0);
    float NdotH = max(dot(p.normal, Wh), 0.0);
    float VdotH = max(dot(Wi, Wh), 0.0);

    vec3 F = F_schlick(p.F0, VdotH);
    vec3 specular = evaluate_ggx(p, F, NdotH, NdotL, NdotV);
    vec3 diffuse = evaluate_lambert(p.albedo.xyz);

    return (vec3(1.0) - F) * diffuse + specular;
}

float pdf_uber(in SurfaceProperties p, in vec3 Wo, in vec3 Wh, in vec3 Wi)
{
    float NdotL = max(dot(p.normal, Wi), 0.0);
    float NdotV = max(dot(p.normal, Wo), 0.0);
    float NdotH = max(dot(p.normal, Wh), 0.0);
    float VdotH = max(dot(Wi, Wh), 0.0);

    float pd = pdf_lambert(NdotL);
    float ps = pdf_D_ggx(p.roughness * p.roughness, NdotH, VdotH);

    return mix(pd, ps, 0.5);
}

vec3 sample_uber(in SurfaceProperties p, in vec3 Wo, in RNG rng, out vec3 Wi, out float pdf)
{
    float alpha = p.roughness * p.roughness;

    vec3 Wh;

    vec3 rand_value = next_vec3(rng);

    bool is_specular = false;

    if (rand_value.x < 0.5)
    {
        Wh = sample_ggx(p.normal, alpha, rand_value.yz);
        Wi = reflect(-Wo, Wh);
        
        float NdotL = max(dot(p.normal, Wi), 0.0);
        float NdotV = max(dot(p.normal, Wo), 0.0);

        if (NdotL > 0.0f && NdotV > 0.0f)
            is_specular = true;
    }
    
    if (!is_specular)
    {
        Wi = sample_lambert(p.normal, rand_value.yz);
        Wh = normalize(Wo + Wi);
    }

    pdf = pdf_uber(p, Wo, Wh, Wi);

    return evaluate_uber(p, Wo, Wh, Wi);
}

#endif