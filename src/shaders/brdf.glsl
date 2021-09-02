#ifndef BRDF_GLSL
#define BRDF_GLSL

#include "common.glsl"

// ------------------------------------------------------------------------

mat3 make_rotation_matrix(vec3 z)
{
    const vec3 ref = abs(dot(z, vec3(0, 1, 0))) > 0.99f ? vec3(0, 0, 1) : vec3(0, 1, 0);

    const vec3 x = normalize(cross(ref, z));
    const vec3 y = cross(z, x);

    return mat3(x, y, z);
}

// ------------------------------------------------------------------------

vec3 sample_cosine_lobe(in vec3 n, in vec2 r)
{
    vec2 rand_sample = max(vec2(0.00001f), r);

    const float phi = 2.0f * M_PI * rand_sample.y;

    const float cos_theta = sqrt(rand_sample.x);
    const float sin_theta = sqrt(1 - rand_sample.x);

    vec3 t = vec3(sin_theta * cos(phi), sin_theta * sin(phi), cos_theta);

    return normalize(make_rotation_matrix(n) * t);
}

// ------------------------------------------------------------------------

float D_ggx(in float ndoth, in float alpha)
{
    float a2    = alpha * alpha;
    float denom = (ndoth * ndoth) * (a2 - 1.0) + 1.0;

    return a2 / max(EPSILON, (M_PI * denom * denom));
}

// ------------------------------------------------------------------------

float G1_schlick_ggx(in float roughness, in float ndotv)
{
    float k = ((roughness + 1) * (roughness + 1)) / 8.0;

    return ndotv / max(EPSILON, (ndotv * (1 - k) + k));
}

// ------------------------------------------------------------------------

float G_schlick_ggx(in float ndotl, in float ndotv, in float roughness)
{
    return G1_schlick_ggx(roughness, ndotl) * G1_schlick_ggx(roughness, ndotv);
}

// ------------------------------------------------------------------------

vec3 F_schlick(in vec3 f0, in float vdoth)
{
    return f0 + (vec3(1.0) - f0) * (pow(1.0 - vdoth, 5.0));
}

// ------------------------------------------------------------------------

vec3 evaluate_specular_brdf(in float roughness, in vec3 F, in float ndoth, in float ndotl, in float ndotv)
{
    float alpha = roughness * roughness;
    return (D_ggx(ndoth, alpha) * F * G_schlick_ggx(ndotl, ndotv, roughness)) / max(EPSILON, (4.0 * ndotl * ndotv));
}

// ------------------------------------------------------------------------

float pdf_specular_ggx_lobe(in float alpha, in float ndoth, in float vdoth)
{
    return D_ggx(ndoth, alpha) * ndoth / max(EPSILON, (4.0 * vdoth));
}

// ------------------------------------------------------------------------

float pdf_cosine_lobe(in float ndotl)
{
    return ndotl / M_PI;
}

// ------------------------------------------------------------------------

vec3 evaluate_diffuse_brdf(in vec3 diffuse_color)
{
    return diffuse_color / M_PI;
}

// ------------------------------------------------------------------------

vec3 sample_specular_ggx_lobe(in vec3 n, in float alpha, in vec2 Xi)
{
    float phi       = 2.0 * M_PI * Xi.x;
    float cos_theta = sqrt((1.0 - Xi.y) / (1.0 + (alpha * alpha - 1.0) * Xi.y));
    float sin_theta = sqrt(1.0 - cos_theta * cos_theta);

    vec3 d;

    d.x = sin_theta * cos(phi);
    d.y = sin_theta * sin(phi);
    d.z = cos_theta;

    return normalize(make_rotation_matrix(n) * d);
}

// ------------------------------------------------------------------------

float pdf_uber_brdf(in vec3 N, in float roughness, in vec3 Wo, in vec3 Wh, in vec3 Wi)
{
    float NdotL = max(dot(N, Wi), 0.0);
    float NdotV = max(dot(N, Wo), 0.0);
    float NdotH = max(dot(N, Wh), 0.0);
    float VdotH = max(dot(Wi, Wh), 0.0);

    float pd = pdf_cosine_lobe(NdotL);
    float ps = pdf_specular_ggx_lobe(roughness * roughness, NdotH, VdotH);

    return mix(pd, ps, 0.5);
}

// ------------------------------------------------------------------------

vec3 evaluate_uber_brdf(in vec3 diffuse_color, in float roughness, in vec3 N, in vec3 F0, in vec3 Wo, in vec3 Wh, in vec3 Wi)
{
    float NdotL = max(dot(N, Wi), 0.0);
    float NdotV = max(dot(N, Wo), 0.0);
    float NdotH = max(dot(N, Wh), 0.0);
    float VdotH = max(dot(Wi, Wh), 0.0);

    vec3 F        = F_schlick(F0, VdotH);
    vec3 specular = evaluate_specular_brdf(roughness, F, NdotH, NdotL, NdotV);
    vec3 diffuse  = evaluate_diffuse_brdf(diffuse_color.xyz);

    return (vec3(1.0) - F) * diffuse + specular;
}

// ------------------------------------------------------------------------

vec3 sample_uber_brdf(in vec3 diffuse_color, in vec3 F0, in vec3 N, in float roughness, in float metallic, in vec3 Wo, in RNG rng, out vec3 Wi, out float pdf)
{
    float alpha = roughness * roughness;

    vec3 Wh;

    vec3 rand_value = next_vec3(rng);

    if (rand_value.x < 0.5f)
    {
        Wh = sample_specular_ggx_lobe(N, alpha, rand_value.yz);

        if (roughness < MIRROR_REFLECTIONS_ROUGHNESS_THRESHOLD)
            Wi = reflect(-Wo, N);
        else
            Wi = reflect(-Wo, Wh);
    }
    else
    {
        Wi = sample_cosine_lobe(N, rand_value.yz);
        Wh = normalize(Wo + Wi);
    }

    pdf = pdf_uber_brdf(N, roughness, Wo, Wh, Wi);

    return evaluate_uber_brdf(diffuse_color, roughness, N, F0, Wo, Wh, Wi);
}

// ------------------------------------------------------------------------

#endif