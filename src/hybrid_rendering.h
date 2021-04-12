#include <application.h>
#include <camera.h>
#include <material.h>
#include <mesh.h>
#include <vk.h>
#include <profiler.h>
#include <assimp/scene.h>
#include <vk_mem_alloc.h>
#include <ray_traced_scene.h>
#include <brdf_preintegrate_lut.h>
#include <hosek_wilkie_sky_model.h>
#include <cubemap_sh_projection.h>
#include <cubemap_prefilter.h>
#include <stdexcept>

#define NUM_PILLARS 6
#define HALTON_SAMPLES 16

enum SceneType
{
    SCENE_PILLARS,
    SCENE_SPONZA,
    SCENE_PICA_PICA
};

enum VisualizationType
{
    VISUALIZATION_FINAL,
    VISUALIZATION_SHADOWS,
    VISUALIZATION_AMBIENT_OCCLUSION,
    VISUALIZATION_REFLECTIONS,
    VISUALIZATION_GLOBAL_ILLUIMINATION,
    VISUALIZATION_REFLECTIONS_TEMPORAL_VARIANCE
};

struct Light
{
    glm::vec4 data0;
    glm::vec4 data1;
};

const std::vector<std::string> visualization_types = { "Final", "Shadows", "Ambient Occlusion", "Reflections", "Global Illumination", "Reflections Temporal Variance" };
const std::vector<std::string> scene_types         = { "Pillars", "Sponza", "Pica Pica" };

void set_light_direction(Light& light, glm::vec3 value)
{
    light.data0.x = value.x;
    light.data0.y = value.y;
    light.data0.z = value.z;
}

void set_light_color(Light& light, glm::vec3 value)
{
    light.data1.x = value.x;
    light.data1.y = value.y;
    light.data1.z = value.z;
}

void set_light_intensity(Light& light, float value)
{
    light.data0.w = value;
}

void set_light_radius(Light& light, float value)
{
    light.data1.w = value;
}

// Uniform buffer data structure.
struct UBO
{
    DW_ALIGNED(16)
    glm::mat4 view_inverse;
    DW_ALIGNED(16)
    glm::mat4 proj_inverse;
    DW_ALIGNED(16)
    glm::mat4 view_proj_inverse;
    DW_ALIGNED(16)
    glm::mat4 prev_view_proj;
    DW_ALIGNED(16)
    glm::mat4 view_proj;
    DW_ALIGNED(16)
    glm::vec4 cam_pos;
    DW_ALIGNED(16)
    Light light;
};

struct GBufferPushConstants
{
    glm::mat4 model;
    glm::mat4 prev_model;
    uint32_t  material_index;
};

struct ShadowPushConstants
{
    float    bias;
    uint32_t num_frames;
};

struct ReflectionsPushConstants
{
    float    bias;
    uint32_t num_frames;
};

struct ReflectionsSpatialResolvePushConstants
{
    glm::vec4 z_buffer_params;
    uint32_t  bypass;
};

struct ReflectionsTemporalPushConstants
{
    uint32_t first_frame;
    uint32_t neighborhood_clamping;
    float    neighborhood_std_scale;
    float    alpha;
};

struct ReflectionsBlurPushConstants
{
    float alpha;
};

struct GIPushConstants
{
    float    bias;
    uint32_t num_frames;
    uint32_t max_ray_depth;
    uint32_t sample_sky;
};

struct AmbientOcclusionPushConstants
{
    uint32_t num_rays;
    uint32_t num_frames;
    float    ray_length;
    float    power;
    float    bias;
};

struct SVGFReprojectionPushConstants
{
    float alpha;
    float moments_alpha;
};

struct SVGFFilterMomentsPushConstants
{
    float phi_color;
    float phi_normal;
};

struct SVGFATrousFilterPushConstants
{
    int   radius;
    int   step_size;
    float phi_color;
    float phi_normal;
};

struct SkyboxPushConstants
{
    glm::mat4 projection;
    glm::mat4 view;
};

struct DeferredShadingPushConstants
{
    int shadows;
    int ao;
    int reflections;
};

struct TAAPushConstants
{
    glm::vec4 texel_size;
    glm::vec4 current_prev_jitter;
    glm::vec4 time_params;
    float     feedback_min;
    float     feedback_max;
    int       sharpen;
};

struct ToneMapPushConstants
{
    int   visualization;
    float exposure;
};

void pipeline_barrier(dw::vk::CommandBuffer::Ptr        cmd_buf,
                      std::vector<VkMemoryBarrier>      memory_barriers,
                      std::vector<VkImageMemoryBarrier> image_memory_barriers,
                      VkPipelineStageFlags              srcStageMask,
                      VkPipelineStageFlags              dstStageMask)
{
    vkCmdPipelineBarrier(
        cmd_buf->handle(),
        srcStageMask,
        dstStageMask,
        0,
        memory_barriers.size(),
        memory_barriers.data(),
        0,
        nullptr,
        image_memory_barriers.size(),
        image_memory_barriers.data());
}

VkImageMemoryBarrier image_memory_barrier(dw::vk::Image::Ptr      image,
                                          VkImageLayout           oldImageLayout,
                                          VkImageLayout           newImageLayout,
                                          VkImageSubresourceRange subresourceRange,
                                          VkAccessFlags           srcAccessFlags,
                                          VkAccessFlags           dstAccessFlags)
{
    // Create an image barrier object
    VkImageMemoryBarrier memory_barrier;
    DW_ZERO_MEMORY(memory_barrier);

    memory_barrier.sType            = VK_STRUCTURE_TYPE_IMAGE_MEMORY_BARRIER;
    memory_barrier.oldLayout        = oldImageLayout;
    memory_barrier.newLayout        = newImageLayout;
    memory_barrier.image            = image->handle();
    memory_barrier.subresourceRange = subresourceRange;
    memory_barrier.srcAccessMask    = srcAccessFlags;
    memory_barrier.dstAccessMask    = dstAccessFlags;

    return memory_barrier;
}

VkMemoryBarrier memory_barrier(VkAccessFlags srcAccessFlags, VkAccessFlags dstAccessFlags)
{
    // Create an image barrier object
    VkMemoryBarrier memory_barrier;
    DW_ZERO_MEMORY(memory_barrier);

    memory_barrier.sType         = VK_STRUCTURE_TYPE_MEMORY_BARRIER;
    memory_barrier.srcAccessMask = srcAccessFlags;
    memory_barrier.dstAccessMask = dstAccessFlags;

    return memory_barrier;
}

float halton_sequence(int base, int index)
{
    float result = 0;
    float f      = 1;
    while (index > 0)
    {
        f /= base;
        result += f * (index % base);
        index = floor(index / base);
    }

    return result;
}

class HybridRendering;

class TemporalReprojection
{
private:
    struct PushConstants
    {
        float    alpha;
        float    neighborhood_scale;
        uint32_t use_variance_clipping;
        uint32_t use_tonemap;
    };

public:
    TemporalReprojection(HybridRendering* sample, std::string name, uint32_t input_width, uint32_t input_height);
    ~TemporalReprojection();

    void                       reproject(dw::vk::CommandBuffer::Ptr cmd_buf, dw::vk::DescriptorSet::Ptr input, dw::vk::DescriptorSet::Ptr prev_input = nullptr);
    void                       gui();
    dw::vk::DescriptorSet::Ptr output_ds();

    inline bool                             variance_clipping() { return m_use_variance_clipping; }
    inline bool                             tone_map() { return m_use_tone_map; }
    inline float                            alpha() { return m_alpha; }
    inline float                            neighborhood_scale() { return m_neighborhood_scale; }
    inline dw::vk::DescriptorSetLayout::Ptr read_ds_layout() { return m_read_ds_layout; }
    inline void                             set_variance_clipping(bool value) { m_use_tone_map = value; }
    inline void                             set_tone_map(bool value) { m_use_variance_clipping = value; }
    inline void                             set_alpha(float value) { m_alpha = value; }
    inline void                             set_neighborhood_scale(float value) { m_neighborhood_scale = value; }

private:
    void clear_images(dw::vk::CommandBuffer::Ptr cmd_buf);

private:
    std::string      m_name;
    HybridRendering* m_sample;
    uint32_t         m_input_width;
    uint32_t         m_input_height;
    float            m_scale                 = 1.0f;
    bool             m_use_variance_clipping = true;
    bool             m_use_tone_map          = false;
    float            m_neighborhood_scale    = 1.0f;
    float            m_alpha                 = 0.01f;
    int32_t          m_read_idx              = 0;
    // Reprojection
    dw::vk::ComputePipeline::Ptr     m_pipeline;
    dw::vk::PipelineLayout::Ptr      m_pipeline_layout;
    dw::vk::DescriptorSetLayout::Ptr m_read_ds_layout;
    dw::vk::DescriptorSetLayout::Ptr m_write_ds_layout;
    dw::vk::Image::Ptr               m_color_image[2];
    dw::vk::ImageView::Ptr           m_color_view[2];
    dw::vk::Image::Ptr               m_history_length_image[2];
    dw::vk::ImageView::Ptr           m_history_length_view[2];
    dw::vk::DescriptorSet::Ptr       m_write_ds[2];
    dw::vk::DescriptorSet::Ptr       m_read_ds[2];
    dw::vk::DescriptorSet::Ptr       m_output_read_ds[2];
};

class SpatialReconstruction
{
private:
    struct PushConstants
    {
        glm::vec4 z_buffer_params;
        uint32_t  num_frames;
    };

public:
    SpatialReconstruction(HybridRendering* sample, std::string name, uint32_t input_width, uint32_t input_height);
    ~SpatialReconstruction();

    void                       reconstruct(dw::vk::CommandBuffer::Ptr cmd_buf, dw::vk::DescriptorSet::Ptr input);
    void                       gui();
    dw::vk::DescriptorSet::Ptr output_ds();

private:
    std::string      m_name;
    HybridRendering* m_sample;

    uint32_t m_input_width;
    uint32_t m_input_height;
    float    m_scale = 0.5f;

    // Reconstruction
    dw::vk::PipelineLayout::Ptr  m_layout;
    dw::vk::ComputePipeline::Ptr m_pipeline;
    dw::vk::Image::Ptr           m_image;
    dw::vk::ImageView::Ptr       m_image_view;
    dw::vk::DescriptorSet::Ptr   m_read_ds;
    dw::vk::DescriptorSet::Ptr   m_write_ds;
};

class BilateralBlur
{
private:
    struct PushConstants
    {
        glm::vec4 z_buffer_params;
        float     variance_threshold;
        float     roughness_sigma_min;
        float     roughness_sigma_max;
        int32_t   radius;
        uint32_t  roughness_weight;
        uint32_t  depth_weight;
        uint32_t  normal_weight;
    };

public:
    BilateralBlur(HybridRendering* sample, std::string name, uint32_t input_width, uint32_t input_height);
    ~BilateralBlur();

    void                       blur(dw::vk::CommandBuffer::Ptr cmd_buf, dw::vk::DescriptorSet::Ptr input);
    void                       prepare_first_frame(dw::vk::CommandBuffer::Ptr cmd_buf);
    void                       gui();
    dw::vk::DescriptorSet::Ptr output_ds();

    inline int32_t blur_radius() { return m_blur_radius; }
    inline float   variance_threshold() { return m_variance_threshold; }
    inline bool    depth_weight() { return m_use_depth_weight; }
    inline bool    normal_weight() { return m_use_normal_weight; }
    inline bool    roughness_weight() { return m_use_roughness_weight; }
    inline bool    reflections_sigma_min() { return m_roughness_sigma_min; }
    inline bool    reflections_sigma_max() { return m_roughness_sigma_max; }
    inline void    set_blur_radius(int32_t n) { m_blur_radius = glm::clamp(n, 1, 7); }
    inline void    set_variance_threshold(float v) { m_variance_threshold = glm::clamp(v, 0.0f, 1.0f); }
    inline void    set_depth_weight(bool v) { m_use_depth_weight = v; }
    inline void    set_normal_weight(bool v) { m_use_normal_weight = v; }
    inline void    set_roughness_weight(bool v) { m_use_roughness_weight = v; }
    inline void    set_reflections_sigma_min(bool v) { m_roughness_sigma_min = v; }
    inline void    set_reflections_sigma_max(bool v) { m_roughness_sigma_max = v; }

private:
    std::string      m_name;
    HybridRendering* m_sample;

    uint32_t m_input_width;
    uint32_t m_input_height;
    float    m_scale                = 1.0f;
    int32_t  m_blur_radius          = 5;
    float    m_variance_threshold   = 0.1f;
    float    m_roughness_sigma_min  = 0.001f;
    float    m_roughness_sigma_max  = 0.01f;
    bool     m_use_depth_weight     = true;
    bool     m_use_normal_weight    = true;
    bool     m_use_roughness_weight = true;

    // Reconstruction
    dw::vk::PipelineLayout::Ptr  m_layout;
    dw::vk::ComputePipeline::Ptr m_pipeline;
    dw::vk::Image::Ptr           m_image;
    dw::vk::ImageView::Ptr       m_image_view;
    dw::vk::DescriptorSet::Ptr   m_read_ds;
    dw::vk::DescriptorSet::Ptr   m_write_ds;
};

class SVGFDenoiser
{
public:
    struct ReprojectionPushConstants
    {
        float alpha;
        float moments_alpha;
    };

    struct FilterMomentsPushConstants
    {
        float phi_color;
        float phi_normal;
    };

    struct ATrousFilterPushConstants
    {
        int   radius;
        int   step_size;
        float phi_color;
        float phi_normal;
    };

public:
    SVGFDenoiser(HybridRendering* sample, std::string name, uint32_t input_width, uint32_t input_height, uint32_t filter_iterations);
    ~SVGFDenoiser();
    void                       denoise(dw::vk::CommandBuffer::Ptr cmd_buf, dw::vk::DescriptorSet::Ptr input);
    dw::vk::DescriptorSet::Ptr output_ds();

    inline uint32_t filter_iterations() { return m_a_trous_filter_iterations; }
    inline void     set_filter_iterations(uint32_t n) { m_a_trous_filter_iterations = glm::clamp(n, 1u, 5u); }

protected:
    void create_reprojection_resources();
    void create_filter_moments_resources();
    void create_a_trous_filter_resources();
    void clear_images(dw::vk::CommandBuffer::Ptr cmd_buf);
    void reprojection(dw::vk::CommandBuffer::Ptr cmd_buf, dw::vk::DescriptorSet::Ptr input);
    void filter_moments(dw::vk::CommandBuffer::Ptr cmd_buf);
    void a_trous_filter(dw::vk::CommandBuffer::Ptr cmd_buf);

private:
    std::string      m_name;
    HybridRendering* m_sample;
    bool             m_use_spatial_for_feedback = false;
    uint32_t         m_input_width;
    uint32_t         m_input_height;
    float            m_alpha                      = 0.01f;
    float            m_moments_alpha              = 0.2f;
    float            m_phi_color                  = 10.0f;
    float            m_phi_normal                 = 128.0f;
    int32_t          m_a_trous_radius             = 1;
    int32_t          m_a_trous_filter_iterations  = 4;
    int32_t          m_a_trous_feedback_iteration = 1;
    int32_t          m_read_idx                   = 0;

    // Reprojection
    dw::vk::ComputePipeline::Ptr     m_reprojection_pipeline;
    dw::vk::PipelineLayout::Ptr      m_reprojection_pipeline_layout;
    dw::vk::DescriptorSetLayout::Ptr m_reprojection_read_ds_layout;
    dw::vk::DescriptorSetLayout::Ptr m_reprojection_write_ds_layout;
    dw::vk::Image::Ptr               m_reprojection_image[2];
    dw::vk::ImageView::Ptr           m_reprojection_view[2];
    dw::vk::Image::Ptr               m_moments_image[2];
    dw::vk::ImageView::Ptr           m_moments_view[2];
    dw::vk::Image::Ptr               m_history_length_image[2];
    dw::vk::ImageView::Ptr           m_history_length_view[2];
    dw::vk::DescriptorSet::Ptr       m_reprojection_write_ds[2];
    dw::vk::DescriptorSet::Ptr       m_reprojection_read_ds[2];
    dw::vk::Image::Ptr               m_prev_reprojection_image;
    dw::vk::ImageView::Ptr           m_prev_reprojection_view;
    dw::vk::DescriptorSet::Ptr       m_prev_reprojection_read_ds;

    // Filter Moments
    dw::vk::ComputePipeline::Ptr m_filter_moments_pipeline;
    dw::vk::PipelineLayout::Ptr  m_filter_moments_pipeline_layout;
    dw::vk::Image::Ptr           m_filter_moments_image;
    dw::vk::ImageView::Ptr       m_filter_moments_view;
    dw::vk::DescriptorSet::Ptr   m_filter_moments_write_ds;
    dw::vk::DescriptorSet::Ptr   m_filter_moments_read_ds;

    // A-Trous Filter
    dw::vk::ComputePipeline::Ptr m_a_trous_filter_pipeline;
    dw::vk::PipelineLayout::Ptr  m_a_trous_filter_pipeline_layout;
    dw::vk::Image::Ptr           m_a_trous_image[2];
    dw::vk::ImageView::Ptr       m_a_trous_view[2];
    dw::vk::DescriptorSet::Ptr   m_a_trous_read_ds[2];
    dw::vk::DescriptorSet::Ptr   m_a_trous_write_ds[2];
};

class DiffuseDenoiser
{
public:
    DiffuseDenoiser(HybridRendering* sample, std::string name, uint32_t input_width, uint32_t input_height);
    ~DiffuseDenoiser();

    void                       denoise(dw::vk::CommandBuffer::Ptr cmd_buf, dw::vk::DescriptorSet::Ptr input);
    void                       gui();
    dw::vk::DescriptorSet::Ptr output_ds();

private:
    std::string      m_name;
    HybridRendering* m_sample;

    uint32_t m_input_width;
    uint32_t m_input_height;
    bool     m_use_blur_as_temporal_input = false;

    // Temporal
    std::unique_ptr<TemporalReprojection> m_temporal_reprojection;

    // Bilateral Blur
    std::unique_ptr<BilateralBlur> m_bilateral_blur;
};

class ReflectionDenoiser
{
public:
    ReflectionDenoiser(HybridRendering* sample, std::string name, uint32_t input_width, uint32_t input_height);
    ~ReflectionDenoiser();

    void                       denoise(dw::vk::CommandBuffer::Ptr cmd_buf, dw::vk::DescriptorSet::Ptr input);
    void                       gui();
    dw::vk::DescriptorSet::Ptr output_ds();

    inline bool temporal_pre_pass() { return m_use_temporal_pre_pass; }
    inline void set_temporal_pre_pass(bool value) { m_use_temporal_pre_pass = value; }

private:
    std::string      m_name;
    HybridRendering* m_sample;

    uint32_t m_input_width;
    uint32_t m_input_height;
    bool     m_use_temporal_pre_pass      = true;
    bool     m_use_blur_as_temporal_input = false;
    bool     m_use_bilateral_blur         = true;

    // Reconstruction
    std::unique_ptr<SpatialReconstruction> m_spatial_reconstruction;

    // Temporal
    std::unique_ptr<TemporalReprojection> m_temporal_pre_pass;
    std::unique_ptr<TemporalReprojection> m_temporal_main_pass;

    // Bilateral Blur
    std::unique_ptr<BilateralBlur> m_bilateral_blur;
};

class GroundTruthPathTracer
{
};

class RayTracedAO
{
};

class RayTracedShadows
{
};

class PathTracedGI
{
};

class DDGI
{
};

class StochasticReflections
{
};

class HybridRendering : public dw::Application
{
public:
    friend class TemporalReprojection;
    friend class SpatialReconstruction;
    friend class BilateralBlur;
    friend class SVGFDenoiser;
    friend class DiffuseDenoiser;
    friend class ReflectionDenoiser;
    friend class SSaReflections;

private:
    struct GPUResources
    {
        // Assets.
        std::vector<dw::Mesh::Ptr> meshes;
        dw::RayTracedScene::Ptr    pillars_scene;
        dw::RayTracedScene::Ptr    sponza_scene;
        dw::RayTracedScene::Ptr    pica_pica_scene;
        dw::RayTracedScene::Ptr    current_scene;

        // Common
        dw::vk::DescriptorSet::Ptr       per_frame_ds;
        dw::vk::DescriptorSetLayout::Ptr per_frame_ds_layout;
        dw::vk::DescriptorSetLayout::Ptr combined_sampler_ds_layout;
        dw::vk::DescriptorSetLayout::Ptr storage_image_ds_layout;
        dw::vk::Buffer::Ptr              ubo;
        dw::vk::Image::Ptr               blue_noise_image_1;
        dw::vk::ImageView::Ptr           blue_noise_view_1;
        dw::vk::Image::Ptr               blue_noise_image_2;
        dw::vk::ImageView::Ptr           blue_noise_view_2;

        // Denoisers
        std::unique_ptr<SVGFDenoiser>       svgf_shadow_denoiser;
        std::unique_ptr<SVGFDenoiser>       svgf_gi_denoiser;
        std::unique_ptr<ReflectionDenoiser> reflection_denoiser;
        std::unique_ptr<DiffuseDenoiser>    shadow_denoiser;

        // Ray-Traced Shadows
        dw::vk::RayTracingPipeline::Ptr shadow_mask_pipeline;
        dw::vk::PipelineLayout::Ptr     shadow_mask_pipeline_layout;
        dw::vk::ShaderBindingTable::Ptr shadow_mask_sbt;
        dw::vk::Image::Ptr              visibility_image;
        dw::vk::ImageView::Ptr          visibility_view;
        dw::vk::DescriptorSet::Ptr      visibility_write_ds;
        dw::vk::DescriptorSet::Ptr      visibility_read_ds;

        // RTAO
        dw::vk::RayTracingPipeline::Ptr rtao_pipeline;
        dw::vk::PipelineLayout::Ptr     rtao_pipeline_layout;
        dw::vk::ShaderBindingTable::Ptr rtao_sbt;

        // Reflection RT pass
        dw::vk::DescriptorSet::Ptr      reflection_rt_write_ds;
        dw::vk::DescriptorSet::Ptr      reflection_rt_read_ds;
        dw::vk::RayTracingPipeline::Ptr reflection_rt_pipeline;
        dw::vk::PipelineLayout::Ptr     reflection_rt_pipeline_layout;
        dw::vk::Image::Ptr              reflection_rt_color_image;
        dw::vk::ImageView::Ptr          reflection_rt_color_view;
        dw::vk::ShaderBindingTable::Ptr reflection_rt_sbt;

        // Global Illumination Ray Tracing pass
        dw::vk::DescriptorSet::Ptr      rtgi_write_ds;
        dw::vk::DescriptorSet::Ptr      rtgi_read_ds;
        dw::vk::RayTracingPipeline::Ptr rtgi_pipeline;
        dw::vk::PipelineLayout::Ptr     rtgi_pipeline_layout;
        dw::vk::Image::Ptr              rtgi_image;
        dw::vk::ImageView::Ptr          rtgi_view;
        dw::vk::ShaderBindingTable::Ptr rtgi_sbt;

        // Deferred pass
        dw::vk::RenderPass::Ptr       deferred_rp;
        dw::vk::Framebuffer::Ptr      deferred_fbo;
        dw::vk::Image::Ptr            deferred_image;
        dw::vk::ImageView::Ptr        deferred_view;
        dw::vk::GraphicsPipeline::Ptr deferred_pipeline;
        dw::vk::PipelineLayout::Ptr   deferred_pipeline_layout;
        dw::vk::DescriptorSet::Ptr    deferred_read_ds;

        // TAA pass
        std::vector<dw::vk::Image::Ptr>         taa_image;
        std::vector<dw::vk::ImageView::Ptr>     taa_view;
        dw::vk::ComputePipeline::Ptr            taa_pipeline;
        dw::vk::PipelineLayout::Ptr             taa_pipeline_layout;
        std::vector<dw::vk::DescriptorSet::Ptr> taa_read_ds;
        std::vector<dw::vk::DescriptorSet::Ptr> taa_write_ds;

        // Copy pass
        dw::vk::GraphicsPipeline::Ptr copy_pipeline;
        dw::vk::PipelineLayout::Ptr   copy_pipeline_layout;

        // G-Buffer pass
        dw::vk::Image::Ptr                      g_buffer_1; // RGB: Albedo, A: Metallic
        dw::vk::Image::Ptr                      g_buffer_2; // RGB: Normal, A: Roughness
        dw::vk::Image::Ptr                      g_buffer_3; // RGB: Position, A: -
        std::vector<dw::vk::Image::Ptr>         g_buffer_linear_z;
        dw::vk::Image::Ptr                      g_buffer_depth;
        dw::vk::ImageView::Ptr                  g_buffer_1_view;
        dw::vk::ImageView::Ptr                  g_buffer_2_view;
        dw::vk::ImageView::Ptr                  g_buffer_3_view;
        std::vector<dw::vk::ImageView::Ptr>     g_buffer_linear_z_view;
        dw::vk::ImageView::Ptr                  g_buffer_depth_view;
        dw::vk::Image::Ptr                      downsampled_g_buffer_1; // RGB: Albedo, A: Metallic
        dw::vk::Image::Ptr                      downsampled_g_buffer_2; // RGB: Normal, A: Roughness
        dw::vk::Image::Ptr                      downsampled_g_buffer_3; // RGB: Position, A: -
        std::vector<dw::vk::Image::Ptr>         downsampled_g_buffer_linear_z;
        dw::vk::Image::Ptr                      downsampled_g_buffer_depth;
        dw::vk::ImageView::Ptr                  downsampled_g_buffer_1_view;
        dw::vk::ImageView::Ptr                  downsampled_g_buffer_2_view;
        dw::vk::ImageView::Ptr                  downsampled_g_buffer_3_view;
        std::vector<dw::vk::ImageView::Ptr>     downsampled_g_buffer_linear_z_view;
        dw::vk::ImageView::Ptr                  downsampled_g_buffer_depth_view;
        std::vector<dw::vk::Framebuffer::Ptr>   g_buffer_fbo;
        dw::vk::RenderPass::Ptr                 g_buffer_rp;
        dw::vk::GraphicsPipeline::Ptr           g_buffer_pipeline;
        dw::vk::PipelineLayout::Ptr             g_buffer_pipeline_layout;
        dw::vk::DescriptorSetLayout::Ptr        g_buffer_ds_layout;
        std::vector<dw::vk::DescriptorSet::Ptr> g_buffer_ds;
        std::vector<dw::vk::DescriptorSet::Ptr> downsampled_g_buffer_ds;

        // Skybox
        dw::vk::Buffer::Ptr           cube_vbo;
        dw::vk::GraphicsPipeline::Ptr skybox_pipeline;
        dw::vk::PipelineLayout::Ptr   skybox_pipeline_layout;
        dw::vk::DescriptorSet::Ptr    skybox_ds;
        dw::vk::RenderPass::Ptr       skybox_rp;
        dw::vk::Framebuffer::Ptr      skybox_fbo;

        // PBR resources
        dw::vk::DescriptorSetLayout::Ptr pbr_ds_layout;
        dw::vk::DescriptorSet::Ptr       pbr_ds;

        // Helpers
        std::unique_ptr<dw::BRDFIntegrateLUT>    brdf_preintegrate_lut;
        std::unique_ptr<dw::HosekWilkieSkyModel> hosek_wilkie_sky_model;
        std::unique_ptr<dw::CubemapSHProjection> cubemap_sh_projection;
        std::unique_ptr<dw::CubemapPrefiler>     cubemap_prefilter;
    };

protected:
    bool init(int argc, const char* argv[]) override;
    void update(double delta) override;
    void shutdown() override;
    void key_pressed(int code) override;
    void key_released(int code) override;
    void mouse_pressed(int code) override;
    void mouse_released(int code) override;
    dw::AppSettings intial_app_settings() override;
    void            window_resized(int width, int height) override;

private:
    void create_output_images();
    void create_render_passes();
    void create_framebuffers();
    bool create_uniform_buffer();
    void create_descriptor_set_layouts();
    void create_descriptor_sets();
    void write_descriptor_sets();
    void create_deferred_pipeline();
    void create_tone_map_pipeline();
    void create_shadow_mask_ray_tracing_pipeline();
    void create_ambient_occlusion_ray_tracing_pipeline();
    void create_reflection_ray_tracing_pipeline();
    void create_gi_ray_tracing_pipeline();
    void create_gbuffer_pipeline();
    void create_skybox_pipeline();
    void create_taa_pipeline();
    void create_cube();
    bool load_mesh();
    void create_camera();
    void ray_trace_shadows(dw::vk::CommandBuffer::Ptr cmd_buf);
    void ray_trace_ambient_occlusion(dw::vk::CommandBuffer::Ptr cmd_buf);
    void ray_trace_reflection(dw::vk::CommandBuffer::Ptr cmd_buf);
    void ray_trace_gi(dw::vk::CommandBuffer::Ptr cmd_buf);
    void render_gbuffer(dw::vk::CommandBuffer::Ptr cmd_buf);
    void downsample_gbuffer(dw::vk::CommandBuffer::Ptr cmd_buf);
    void blitt_image(dw::vk::CommandBuffer::Ptr cmd_buf,
                     dw::vk::Image::Ptr         src,
                     dw::vk::Image::Ptr         dst,
                     VkImageLayout              src_img_src_layout,
                     VkImageLayout              src_img_dst_layout,
                     VkImageLayout              dst_img_src_layout,
                     VkImageLayout              dst_img_dst_layout,
                     VkImageAspectFlags         aspect_flags,
                     VkFilter                   filter);
    void render_skybox(dw::vk::CommandBuffer::Ptr cmd_buf);
    void deferred_shading(dw::vk::CommandBuffer::Ptr cmd_buf);
    void temporal_aa(dw::vk::CommandBuffer::Ptr cmd_buf);
    void tone_map(dw::vk::CommandBuffer::Ptr cmd_buf);
    void update_uniforms(dw::vk::CommandBuffer::Ptr cmd_buf);
    void update_ibl(dw::vk::CommandBuffer::Ptr cmd_buf);
    void update_light_animation();
    void update_camera();
    void clear_images(dw::vk::CommandBuffer::Ptr cmd_buf);
    void set_active_scene();

private:
    std::unique_ptr<GPUResources> m_gpu_resources;

    bool   m_first_frame = true;
    size_t m_ubo_size;

    // Camera.
    std::unique_ptr<dw::Camera> m_main_camera;
    glm::mat4                   m_projection;
    glm::mat4                   m_prev_view_proj;
    std::vector<glm::vec2>      m_jitter_samples;
    glm::vec2                   m_prev_jitter    = glm::vec2(0.0f);
    glm::vec2                   m_current_jitter = glm::vec2(0.0f);
    float                       m_near_plane     = 0.1f;
    float                       m_far_plane      = 1000.0f;

    // TAA
    bool  m_taa_enabled      = true;
    bool  m_taa_sharpen      = true;
    float m_taa_feedback_min = 0.88f;
    float m_taa_feedback_max = 0.97f;

    // Camera controls.
    bool  m_mouse_look         = false;
    float m_heading_speed      = 0.0f;
    float m_sideways_speed     = 0.0f;
    float m_camera_sensitivity = 0.05f;
    float m_camera_speed       = 0.2f;
    float m_offset             = 0.1f;
    bool  m_debug_gui          = false;

    // Camera orientation.
    float m_camera_x;
    float m_camera_y;

    // Light
    float     m_light_radius    = 0.1f;
    glm::vec3 m_light_direction = glm::normalize(glm::vec3(0.568f, 0.707f, -0.421f));
    glm::vec3 m_light_color     = glm::vec3(1.0f);
    float     m_light_intensity = 1.0f;
    bool      m_light_animation = false;

    // General Ray Tracing Settings
    bool m_quarter_resolution = true;
    bool m_downscaled_rt      = true;

    // Ray Traced Shadows
    bool    m_ping_pong                            = false;
    bool    m_svgf_shadow_denoise                  = true;
    bool    m_svgf_shadow_use_spatial_for_feedback = false;
    bool    m_rt_shadows_enabled                   = true;
    int32_t m_num_frames                           = 0;
    float   m_ray_traced_shadows_bias              = 0.1f;
    int32_t m_max_samples                          = 10000;
    float   m_svgf_alpha                           = 0.01f;
    float   m_svgf_moments_alpha                   = 0.2f;
    float   m_svgf_phi_color                       = 10.0f;
    float   m_svgf_phi_normal                      = 128.0f;
    int32_t m_a_trous_radius                       = 1;
    int32_t m_a_trous_filter_iterations            = 4;
    int32_t m_a_trous_feedback_iteration           = 1;
    int32_t m_visiblity_read_idx                   = 0;

    // Ray Traced Reflections
    float m_ray_traced_reflections_bias                        = 0.1f;
    bool  m_ray_traced_reflections_spatial_resolve             = true;
    bool  m_rt_reflections_enabled                             = true;
    bool  m_rt_reflections_neighborhood_clamping               = true;
    bool  m_rt_reflections_blur                                = true;
    float m_ray_traced_reflections_alpha                       = 0.01f;
    float m_ray_traced_reflections_std_scale                   = 5.0f;
    float m_ray_traced_reflections_temporal_variance_threshold = 0.002f;
    float m_ray_traced_reflections_sigma_min                   = 0.001f;
    float m_ray_traced_reflections_sigma_max                   = 0.01f;

    // Ray Traced Global Illumination
    float   m_ray_traced_gi_bias            = 0.1f;
    int32_t m_ray_traced_gi_max_ray_bounces = 1;
    bool    m_ray_traced_gi_sample_sky      = false;

    // Ambient Occlusion
    int32_t m_rtao_num_rays   = 2;
    float   m_rtao_ray_length = 30.0f;
    float   m_rtao_power      = 5.0f;
    float   m_rtao_bias       = 0.1f;
    bool    m_rtao_enabled    = true;

    // Uniforms.
    UBO               m_ubo_data;
    float             m_exposure              = 1.0f;
    SceneType         m_current_scene         = SCENE_PILLARS;
    VisualizationType m_current_visualization = VISUALIZATION_FINAL;
};