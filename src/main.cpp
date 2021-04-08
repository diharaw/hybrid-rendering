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
    uint32_t                   filter_iterations();
    void                       set_filter_iterations(uint32_t n);
    dw::vk::DescriptorSet::Ptr denoised_image_ds();

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

class TemporalDenoiser
{
private:
    struct ReprojectionPushConstants
    {
        float    alpha;
        float    neighborhood_scale;
        uint32_t use_variance_clipping;
    };

public:
    TemporalDenoiser(HybridRendering* sample, std::string name, uint32_t input_width, uint32_t input_height);
    ~TemporalDenoiser();

    void                       reproject(dw::vk::CommandBuffer::Ptr cmd_buf, dw::vk::DescriptorSet::Ptr input);
    dw::vk::DescriptorSet::Ptr output_image_ds();

    inline bool                             variance_clipping() { return m_use_variance_clipping; }
    inline float                            alpha() { return m_alpha; }
    inline float                            neighborhood_scale() { return m_neighborhood_scale; }
    inline dw::vk::DescriptorSetLayout::Ptr read_ds_layout() { return m_read_ds_layout; }
    inline void                             set_variance_clipping(bool value) { m_use_variance_clipping = value; }
    inline void                             set_alpha(float value) { m_alpha = value; }
    inline void                             set_neighborhood_scale(float value) { m_neighborhood_scale = value; }

private:
    void clear_images(dw::vk::CommandBuffer::Ptr cmd_buf);

private:
    std::string      m_name;
    HybridRendering* m_sample;
    uint32_t         m_input_width;
    uint32_t         m_input_height;
    bool             m_use_variance_clipping = true;
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
};

class ReflectionDenoiser
{
private:
    struct ReconstructionPushConstants
    {
        glm::vec4 z_buffer_params;
        uint32_t  num_frames;
    };

public:
    ReflectionDenoiser(HybridRendering* sample, std::string name, uint32_t input_width, uint32_t input_height, uint32_t blur_radius);
    ~ReflectionDenoiser();

    void                       denoise(dw::vk::CommandBuffer::Ptr cmd_buf, dw::vk::DescriptorSet::Ptr input);
    dw::vk::DescriptorSet::Ptr denoised_image_ds();

    inline uint32_t blur_radius() { return m_blur_radius; }
    inline bool     temporal_pre_pass() { return m_use_temporal_pre_pass; }
    inline void     set_temporal_pre_pass(bool value) { m_use_temporal_pre_pass = value; }
    inline void     set_blur_radius(uint32_t n) { m_blur_radius = glm::clamp(n, 1u, 7u); }

protected:
    void reconstruct_pass(dw::vk::CommandBuffer::Ptr cmd_buf, dw::vk::DescriptorSet::Ptr input);
    void temporal_pre_pass(dw::vk::CommandBuffer::Ptr cmd_buf);
    void temporal_main_pass(dw::vk::CommandBuffer::Ptr cmd_buf);
    void bilateral_blur();

private:
    std::string      m_name;
    HybridRendering* m_sample;

    uint32_t m_input_width;
    uint32_t m_input_height;
    uint32_t m_blur_radius;
    bool     m_use_temporal_pre_pass = true;

    dw::vk::DescriptorSet::Ptr m_blur_read_ds;
    dw::vk::DescriptorSet::Ptr m_blur_write_ds;

    // Reconstruction
    bool                         m_reconstruction;
    dw::vk::PipelineLayout::Ptr  m_reconstruction_layout;
    dw::vk::ComputePipeline::Ptr m_reconstruction_pipeline;
    dw::vk::Image::Ptr           m_reconstruction_image;
    dw::vk::ImageView::Ptr       m_reconstruction_image_view;
    dw::vk::DescriptorSet::Ptr   m_reconstruction_read_ds;
    dw::vk::DescriptorSet::Ptr   m_reconstruction_write_ds;

    // Temporal
    std::unique_ptr<TemporalDenoiser> m_temporal_pre_pass;
    std::unique_ptr<TemporalDenoiser> m_temporal_main_pass;
};

class HybridRendering : public dw::Application
{
public:
    friend class TemporalDenoiser;
    friend class SVGFDenoiser;
    friend class ReflectionDenoiser;

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
        std::unique_ptr<SVGFDenoiser> svgf_shadow_denoiser;
        std::unique_ptr<SVGFDenoiser> svgf_gi_denoiser;

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

        // Reflection spatial pass
        dw::vk::ComputePipeline::Ptr reflection_resolve_pipeline;
        dw::vk::PipelineLayout::Ptr  reflection_resolve_pipeline_layout;
        dw::vk::Image::Ptr           reflection_resolve_image;
        dw::vk::ImageView::Ptr       reflection_resolve_view;
        dw::vk::DescriptorSet::Ptr   reflection_resolve_write_ds;
        dw::vk::DescriptorSet::Ptr   reflection_resolve_read_ds;

        // Reflection blur pass
        dw::vk::ComputePipeline::Ptr            reflection_blur_pipeline;
        dw::vk::PipelineLayout::Ptr             reflection_blur_pipeline_layout;
        std::vector<dw::vk::DescriptorSet::Ptr> reflection_blur_write_ds;
        std::vector<dw::vk::DescriptorSet::Ptr> reflection_blur_read_ds;
        std::vector<dw::vk::Image::Ptr>         reflection_blur_image;
        std::vector<dw::vk::ImageView::Ptr>     reflection_blur_view;
        std::vector<dw::vk::DescriptorSet::Ptr> reflection_history_ds;
        std::vector<dw::vk::Image::Ptr>         reflection_history_image;
        std::vector<dw::vk::ImageView::Ptr>     reflection_history_view;

        // Reflection temporal pass
        dw::vk::ComputePipeline::Ptr reflection_temporal_pipeline;
        dw::vk::PipelineLayout::Ptr  reflection_temporal_pipeline_layout;
        dw::vk::Image::Ptr           reflection_temporal_image;
        dw::vk::ImageView::Ptr       reflection_temporal_view;
        dw::vk::DescriptorSet::Ptr   reflection_temporal_write_ds;
        dw::vk::DescriptorSet::Ptr   reflection_temporal_read_ds;

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
    // -----------------------------------------------------------------------------------------------------------------------------------

    bool init(int argc, const char* argv[]) override
    {
        m_gpu_resources = std::unique_ptr<GPUResources>(new GPUResources());

        if (!create_uniform_buffer())
            return false;

        // Load mesh.
        if (!load_mesh())
        {
            DW_LOG_INFO("Failed to load mesh");
            return false;
        }

        m_gpu_resources->brdf_preintegrate_lut  = std::unique_ptr<dw::BRDFIntegrateLUT>(new dw::BRDFIntegrateLUT(m_vk_backend));
        m_gpu_resources->hosek_wilkie_sky_model = std::unique_ptr<dw::HosekWilkieSkyModel>(new dw::HosekWilkieSkyModel(m_vk_backend));
        m_gpu_resources->cubemap_sh_projection  = std::unique_ptr<dw::CubemapSHProjection>(new dw::CubemapSHProjection(m_vk_backend, m_gpu_resources->hosek_wilkie_sky_model->image()));
        m_gpu_resources->cubemap_prefilter      = std::unique_ptr<dw::CubemapPrefiler>(new dw::CubemapPrefiler(m_vk_backend, m_gpu_resources->hosek_wilkie_sky_model->image()));
        m_gpu_resources->blue_noise_image_1     = dw::vk::Image::create_from_file(m_vk_backend, "texture/LDR_RGBA_0.png");
        m_gpu_resources->blue_noise_view_1      = dw::vk::ImageView::create(m_vk_backend, m_gpu_resources->blue_noise_image_1, VK_IMAGE_VIEW_TYPE_2D, VK_IMAGE_ASPECT_COLOR_BIT);
        m_gpu_resources->blue_noise_image_2     = dw::vk::Image::create_from_file(m_vk_backend, "texture/LDR_RGBA_1.png");
        m_gpu_resources->blue_noise_view_2      = dw::vk::ImageView::create(m_vk_backend, m_gpu_resources->blue_noise_image_2, VK_IMAGE_VIEW_TYPE_2D, VK_IMAGE_ASPECT_COLOR_BIT);

        create_output_images();
        create_render_passes();
        create_framebuffers();
        create_descriptor_set_layouts();
        create_descriptor_sets();
        write_descriptor_sets();
        create_deferred_pipeline();
        create_gbuffer_pipeline();
        create_shadow_mask_ray_tracing_pipeline();
        create_ambient_occlusion_ray_tracing_pipeline();
        create_reflection_ray_tracing_pipeline();
        create_gi_ray_tracing_pipeline();
        //create_reflection_resolve_pipeline();
        //create_reflection_temporal_pipeline();
        create_reflection_blur_pipeline();
        create_skybox_pipeline();
        create_tone_map_pipeline();
        create_taa_pipeline();
        create_cube();

        m_gpu_resources->svgf_shadow_denoiser = std::unique_ptr<SVGFDenoiser>(new SVGFDenoiser(this, "SVGF Shadow Denoiser", m_gpu_resources->visibility_image->width(), m_gpu_resources->visibility_image->height(), 4));
        m_gpu_resources->svgf_gi_denoiser     = std::unique_ptr<SVGFDenoiser>(new SVGFDenoiser(this, "SVGF Shadow Denoiser", m_gpu_resources->rtgi_image->width(), m_gpu_resources->rtgi_image->height(), 4));

        // Create camera.
        create_camera();

        for (int i = 1; i <= HALTON_SAMPLES; i++)
            m_jitter_samples.push_back(glm::vec2((2.0f * halton_sequence(2, i) - 1.0f), (2.0f * halton_sequence(3, i) - 1.0f)));

        return true;
    }

    // -----------------------------------------------------------------------------------------------------------------------------------

    void update(double delta) override
    {
        dw::vk::CommandBuffer::Ptr cmd_buf = m_vk_backend->allocate_graphics_command_buffer();

        VkCommandBufferBeginInfo begin_info;
        DW_ZERO_MEMORY(begin_info);

        begin_info.sType = VK_STRUCTURE_TYPE_COMMAND_BUFFER_BEGIN_INFO;

        vkBeginCommandBuffer(cmd_buf->handle(), &begin_info);

        {
            DW_SCOPED_SAMPLE("Update", cmd_buf);

            if (m_debug_gui)
            {
                if (ImGui::CollapsingHeader("Settings", ImGuiTreeNodeFlags_DefaultOpen))
                {
                    if (ImGui::BeginCombo("Scene", scene_types[m_current_scene].c_str()))
                    {
                        for (uint32_t i = 0; i < scene_types.size(); i++)
                        {
                            const bool is_selected = (i == m_current_scene);

                            if (ImGui::Selectable(scene_types[i].c_str(), is_selected))
                            {
                                m_current_scene = (SceneType)i;
                                set_active_scene();
                            }

                            if (is_selected)
                                ImGui::SetItemDefaultFocus();
                        }
                        ImGui::EndCombo();
                    }

                    if (ImGui::BeginCombo("Visualization", visualization_types[m_current_visualization].c_str()))
                    {
                        for (uint32_t i = 0; i < visualization_types.size(); i++)
                        {
                            const bool is_selected = (i == m_current_visualization);

                            if (ImGui::Selectable(visualization_types[i].c_str(), is_selected))
                                m_current_visualization = (VisualizationType)i;

                            if (is_selected)
                                ImGui::SetItemDefaultFocus();
                        }
                        ImGui::EndCombo();
                    }

                    ImGui::InputFloat("Exposure", &m_exposure);
                }
                if (ImGui::CollapsingHeader("Light", ImGuiTreeNodeFlags_DefaultOpen))
                {
                    ImGui::ColorEdit3("Color", &m_light_color.x);
                    ImGui::InputFloat("Intensity", &m_light_intensity);
                    ImGui::SliderFloat("Radius", &m_light_radius, 0.0f, 0.1f);
                    ImGui::InputFloat3("Direction", &m_light_direction.x);
                    ImGui::Checkbox("Animation", &m_light_animation);
                }
                if (ImGui::CollapsingHeader("Ray Traced Shadows", ImGuiTreeNodeFlags_DefaultOpen))
                {
                    ImGui::PushID("RTSS");
                    ImGui::Checkbox("Enabled", &m_rt_shadows_enabled);
                    ImGui::SliderInt("A-Trous Filter Radius", &m_a_trous_radius, 1, 2);
                    ImGui::SliderInt("A-Trous Filter Iterations", &m_a_trous_filter_iterations, 1, 5);
                    ImGui::SliderInt("A-Trous Filter Feedback Tap", &m_a_trous_feedback_iteration, 0, 4);
                    ImGui::SliderFloat("Alpha", &m_svgf_alpha, 0.0f, 1.0f);
                    ImGui::SliderFloat("Moments Alpha", &m_svgf_moments_alpha, 0.0f, 1.0f);
                    ImGui::Checkbox("Denoise", &m_svgf_shadow_denoise);
                    ImGui::Checkbox("Use filter output for reprojection", &m_svgf_shadow_use_spatial_for_feedback);
                    ImGui::InputFloat("Bias", &m_ray_traced_shadows_bias);
                    ImGui::PopID();
                }
                if (ImGui::CollapsingHeader("Ray Traced Reflections", ImGuiTreeNodeFlags_DefaultOpen))
                {
                    ImGui::PushID("RTR");
                    ImGui::Checkbox("Enabled", &m_rt_reflections_enabled);
                    ImGui::InputFloat("Bias", &m_ray_traced_reflections_bias);
                    ImGui::Checkbox("Spatial Resolve", &m_ray_traced_reflections_spatial_resolve);
                    ImGui::Checkbox("Neighborhood Clamping", &m_rt_reflections_neighborhood_clamping);
                    ImGui::Checkbox("Bilateral Blur", &m_rt_reflections_blur);
                    ImGui::SliderFloat("Alpha", &m_ray_traced_reflections_alpha, 0.0f, 1.0f);
                    ImGui::SliderFloat("Standard Deviation Scale", &m_ray_traced_reflections_std_scale, 0.0f, 20.0f);
                    ImGui::SliderFloat("Termporal Variance Threshold", &m_ray_traced_reflections_temporal_variance_threshold, 0.0f, 0.5f);
                    ImGui::PopID();
                }
                if (ImGui::CollapsingHeader("Ray Traced Global Illumination", ImGuiTreeNodeFlags_DefaultOpen))
                {
                    ImGui::PushID("RTR");
                    ImGui::InputFloat("Bias", &m_ray_traced_gi_bias);
                    ImGui::Checkbox("Sample Sky", &m_ray_traced_gi_sample_sky);
                    ImGui::SliderInt("Max Bounces", &m_ray_traced_gi_max_ray_bounces, 1, 4);
                    ImGui::PopID();
                }
                if (ImGui::CollapsingHeader("Ray Traced Ambient Occlusion", ImGuiTreeNodeFlags_DefaultOpen))
                {
                    ImGui::PushID("RTAO");
                    ImGui::Checkbox("Enabled", &m_rtao_enabled);
                    ImGui::SliderInt("Num Rays", &m_rtao_num_rays, 1, 8);
                    ImGui::SliderFloat("Ray Length", &m_rtao_ray_length, 1.0f, 100.0f);
                    ImGui::SliderFloat("Power", &m_rtao_power, 1.0f, 5.0f);
                    ImGui::InputFloat("Bias", &m_rtao_bias);
                    ImGui::PopID();
                }
                if (ImGui::CollapsingHeader("TAA", ImGuiTreeNodeFlags_DefaultOpen))
                {
                    ImGui::PushID("TAA");
                    ImGui::Checkbox("Enabled", &m_taa_enabled);
                    ImGui::Checkbox("Sharpen", &m_taa_sharpen);
                    ImGui::SliderFloat("Feedback Min", &m_taa_feedback_min, 0.0f, 1.0f);
                    ImGui::SliderFloat("Feedback Max", &m_taa_feedback_max, 0.0f, 1.0f);
                    ImGui::PopID();
                }
                if (ImGui::CollapsingHeader("Profiler", ImGuiTreeNodeFlags_DefaultOpen))
                    dw::profiler::ui();
            }

            // Update camera.
            update_camera();

            // Update light.
            update_light_animation();

            // Update uniforms.
            update_uniforms(cmd_buf);

            m_gpu_resources->current_scene->build_tlas(cmd_buf);

            update_ibl(cmd_buf);

            // Render.
            clear_images(cmd_buf);
            render_gbuffer(cmd_buf);
            downsample_gbuffer(cmd_buf);
            ray_trace_shadows(cmd_buf);
            ray_trace_ambient_occlusion(cmd_buf);
            m_gpu_resources->svgf_shadow_denoiser->denoise(cmd_buf, m_gpu_resources->visibility_read_ds);
            ray_trace_reflection(cmd_buf);
            ray_trace_gi(cmd_buf);
            m_gpu_resources->svgf_gi_denoiser->denoise(cmd_buf, m_gpu_resources->rtgi_read_ds);
            //reflection_spatial_resolve(cmd_buf);
            //reflection_temporal(cmd_buf);
            reflection_blur(cmd_buf);
            deferred_shading(cmd_buf);
            render_skybox(cmd_buf);
            temporal_aa(cmd_buf);
            tone_map(cmd_buf);
        }

        vkEndCommandBuffer(cmd_buf->handle());

        submit_and_present({ cmd_buf });

        m_num_frames++;

        if (m_first_frame)
            m_first_frame = false;

        m_ping_pong = !m_ping_pong;
    }

    // -----------------------------------------------------------------------------------------------------------------------------------

    void shutdown() override
    {
        m_gpu_resources.reset();
    }

    // -----------------------------------------------------------------------------------------------------------------------------------

    void key_pressed(int code) override
    {
        // Handle forward movement.
        if (code == GLFW_KEY_W)
            m_heading_speed = m_camera_speed;
        else if (code == GLFW_KEY_S)
            m_heading_speed = -m_camera_speed;

        // Handle sideways movement.
        if (code == GLFW_KEY_A)
            m_sideways_speed = -m_camera_speed;
        else if (code == GLFW_KEY_D)
            m_sideways_speed = m_camera_speed;

        if (code == GLFW_KEY_SPACE)
            m_mouse_look = true;

        if (code == GLFW_KEY_G)
            m_debug_gui = !m_debug_gui;
    }

    // -----------------------------------------------------------------------------------------------------------------------------------

    void key_released(int code) override
    {
        // Handle forward movement.
        if (code == GLFW_KEY_W || code == GLFW_KEY_S)
            m_heading_speed = 0.0f;

        // Handle sideways movement.
        if (code == GLFW_KEY_A || code == GLFW_KEY_D)
            m_sideways_speed = 0.0f;

        if (code == GLFW_KEY_SPACE)
            m_mouse_look = false;
    }

    // -----------------------------------------------------------------------------------------------------------------------------------

    void mouse_pressed(int code) override
    {
        // Enable mouse look.
        if (code == GLFW_MOUSE_BUTTON_RIGHT)
            m_mouse_look = true;
    }

    // -----------------------------------------------------------------------------------------------------------------------------------

    void mouse_released(int code) override
    {
        // Disable mouse look.
        if (code == GLFW_MOUSE_BUTTON_RIGHT)
            m_mouse_look = false;
    }

    // -----------------------------------------------------------------------------------------------------------------------------------

    dw::AppSettings intial_app_settings() override
    {
        // Set custom settings here...
        dw::AppSettings settings;

        settings.width       = 2560;
        settings.height      = 1440;
        settings.title       = "Hybrid Rendering (c) Dihara Wijetunga";
        settings.ray_tracing = true;
        settings.resizable   = false;

        return settings;
    }

    // -----------------------------------------------------------------------------------------------------------------------------------

    void window_resized(int width, int height) override
    {
        // Override window resized method to update camera projection.
        m_main_camera->update_projection(60.0f, m_near_plane, m_far_plane, float(m_width) / float(m_height));

        m_vk_backend->wait_idle();

        create_output_images();
        create_framebuffers();
        write_descriptor_sets();
    }

    // -----------------------------------------------------------------------------------------------------------------------------------

private:
    // -----------------------------------------------------------------------------------------------------------------------------------

    void create_output_images()
    {
        m_gpu_resources->g_buffer_1_view.reset();
        m_gpu_resources->g_buffer_2_view.reset();
        m_gpu_resources->g_buffer_3_view.reset();
        m_gpu_resources->g_buffer_linear_z_view.clear();
        m_gpu_resources->g_buffer_depth_view.reset();
        m_gpu_resources->visibility_view.reset();
        m_gpu_resources->taa_view.clear();
        m_gpu_resources->g_buffer_1.reset();
        m_gpu_resources->g_buffer_2.reset();
        m_gpu_resources->g_buffer_3.reset();
        m_gpu_resources->g_buffer_linear_z.clear();
        m_gpu_resources->g_buffer_depth.reset();
        m_gpu_resources->visibility_image.reset();

        m_gpu_resources->downsampled_g_buffer_1_view.reset();
        m_gpu_resources->downsampled_g_buffer_2_view.reset();
        m_gpu_resources->downsampled_g_buffer_3_view.reset();
        m_gpu_resources->downsampled_g_buffer_linear_z_view.clear();
        m_gpu_resources->downsampled_g_buffer_1.reset();
        m_gpu_resources->downsampled_g_buffer_2.reset();
        m_gpu_resources->downsampled_g_buffer_3.reset();
        m_gpu_resources->downsampled_g_buffer_linear_z.clear();
        m_gpu_resources->taa_image.clear();

        uint32_t rt_image_width  = m_quarter_resolution ? m_width / 2 : m_width;
        uint32_t rt_image_height = m_quarter_resolution ? m_height / 2 : m_height;

        uint32_t rtgi_image_width  = m_downscaled_rt ? m_width / 2 : m_width;
        uint32_t rtgi_image_height = m_downscaled_rt ? m_height / 2 : m_height;

        for (int i = 0; i < 2; i++)
        {
            {
                auto image = dw::vk::Image::create(m_vk_backend, VK_IMAGE_TYPE_2D, rt_image_width, rt_image_height, 1, 1, 1, VK_FORMAT_R16G16B16A16_SFLOAT, VMA_MEMORY_USAGE_GPU_ONLY, VK_IMAGE_USAGE_STORAGE_BIT | VK_IMAGE_USAGE_SAMPLED_BIT | VK_IMAGE_USAGE_TRANSFER_DST_BIT, VK_SAMPLE_COUNT_1_BIT);
                image->set_name("Reflection Blur Image " + std::to_string(i));

                m_gpu_resources->reflection_blur_image.push_back(image);

                auto image_view = dw::vk::ImageView::create(m_vk_backend, image, VK_IMAGE_VIEW_TYPE_2D, VK_IMAGE_ASPECT_COLOR_BIT);
                image_view->set_name("Reflection Blur Image View " + std::to_string(i));

                m_gpu_resources->reflection_blur_view.push_back(image_view);
            }

            {
                auto image = dw::vk::Image::create(m_vk_backend, VK_IMAGE_TYPE_2D, rt_image_width, rt_image_height, 1, 1, 1, VK_FORMAT_R16_SFLOAT, VMA_MEMORY_USAGE_GPU_ONLY, VK_IMAGE_USAGE_STORAGE_BIT | VK_IMAGE_USAGE_SAMPLED_BIT | VK_IMAGE_USAGE_TRANSFER_DST_BIT, VK_SAMPLE_COUNT_1_BIT);
                image->set_name("Reflection History Image " + std::to_string(i));

                m_gpu_resources->reflection_history_image.push_back(image);

                auto image_view = dw::vk::ImageView::create(m_vk_backend, image, VK_IMAGE_VIEW_TYPE_2D, VK_IMAGE_ASPECT_COLOR_BIT);
                image_view->set_name("Reflection History Image View " + std::to_string(i));

                m_gpu_resources->reflection_history_view.push_back(image_view);
            }
        }

        m_gpu_resources->visibility_image = dw::vk::Image::create(m_vk_backend, VK_IMAGE_TYPE_2D, rt_image_width, rt_image_height, 1, 1, 1, VK_FORMAT_R16G16B16A16_SFLOAT, VMA_MEMORY_USAGE_GPU_ONLY, VK_IMAGE_USAGE_STORAGE_BIT | VK_IMAGE_USAGE_SAMPLED_BIT | VK_IMAGE_USAGE_TRANSFER_SRC_BIT, VK_SAMPLE_COUNT_1_BIT);
        m_gpu_resources->visibility_image->set_name("Visibility Image");

        m_gpu_resources->visibility_view = dw::vk::ImageView::create(m_vk_backend, m_gpu_resources->visibility_image, VK_IMAGE_VIEW_TYPE_2D, VK_IMAGE_ASPECT_COLOR_BIT);
        m_gpu_resources->visibility_view->set_name("Visibility Image View");

        m_gpu_resources->deferred_image = dw::vk::Image::create(m_vk_backend, VK_IMAGE_TYPE_2D, m_width, m_height, 1, 1, 1, VK_FORMAT_R16G16B16A16_SFLOAT, VMA_MEMORY_USAGE_GPU_ONLY, VK_IMAGE_USAGE_SAMPLED_BIT | VK_IMAGE_USAGE_COLOR_ATTACHMENT_BIT | VK_IMAGE_USAGE_TRANSFER_SRC_BIT, VK_SAMPLE_COUNT_1_BIT);
        m_gpu_resources->deferred_image->set_name("Deferred Image");

        m_gpu_resources->deferred_view = dw::vk::ImageView::create(m_vk_backend, m_gpu_resources->deferred_image, VK_IMAGE_VIEW_TYPE_2D, VK_IMAGE_ASPECT_COLOR_BIT);
        m_gpu_resources->deferred_view->set_name("Deferred Image View");

        m_gpu_resources->reflection_rt_color_image = dw::vk::Image::create(m_vk_backend, VK_IMAGE_TYPE_2D, rt_image_width, rt_image_height, 1, 1, 1, VK_FORMAT_R16G16B16A16_SFLOAT, VMA_MEMORY_USAGE_GPU_ONLY, VK_IMAGE_USAGE_STORAGE_BIT | VK_IMAGE_USAGE_SAMPLED_BIT, VK_SAMPLE_COUNT_1_BIT);
        m_gpu_resources->reflection_rt_color_image->set_name("Reflection RT Color Image");

        m_gpu_resources->reflection_rt_color_view = dw::vk::ImageView::create(m_vk_backend, m_gpu_resources->reflection_rt_color_image, VK_IMAGE_VIEW_TYPE_2D, VK_IMAGE_ASPECT_COLOR_BIT);
        m_gpu_resources->reflection_rt_color_view->set_name("Reflection RT Color Image View");

        m_gpu_resources->rtgi_image = dw::vk::Image::create(m_vk_backend, VK_IMAGE_TYPE_2D, rtgi_image_width, rtgi_image_height, 1, 1, 1, VK_FORMAT_R16G16B16A16_SFLOAT, VMA_MEMORY_USAGE_GPU_ONLY, VK_IMAGE_USAGE_STORAGE_BIT | VK_IMAGE_USAGE_SAMPLED_BIT, VK_SAMPLE_COUNT_1_BIT);
        m_gpu_resources->rtgi_image->set_name("RTGI Image");

        m_gpu_resources->rtgi_view = dw::vk::ImageView::create(m_vk_backend, m_gpu_resources->rtgi_image, VK_IMAGE_VIEW_TYPE_2D, VK_IMAGE_ASPECT_COLOR_BIT);
        m_gpu_resources->rtgi_view->set_name("RTGI Image View");

        m_gpu_resources->reflection_resolve_image = dw::vk::Image::create(m_vk_backend, VK_IMAGE_TYPE_2D, m_width, m_height, 1, 1, 1, VK_FORMAT_R16G16B16A16_SFLOAT, VMA_MEMORY_USAGE_GPU_ONLY, VK_IMAGE_USAGE_STORAGE_BIT | VK_IMAGE_USAGE_SAMPLED_BIT | VK_IMAGE_USAGE_TRANSFER_SRC_BIT, VK_SAMPLE_COUNT_1_BIT);
        m_gpu_resources->reflection_resolve_image->set_name("Reflection Resolve Image");

        m_gpu_resources->reflection_resolve_view = dw::vk::ImageView::create(m_vk_backend, m_gpu_resources->reflection_resolve_image, VK_IMAGE_VIEW_TYPE_2D, VK_IMAGE_ASPECT_COLOR_BIT);
        m_gpu_resources->reflection_resolve_view->set_name("Reflection Resolve Image View");

        m_gpu_resources->reflection_temporal_image = dw::vk::Image::create(m_vk_backend, VK_IMAGE_TYPE_2D, m_width, m_height, 1, 1, 1, VK_FORMAT_R16G16B16A16_SFLOAT, VMA_MEMORY_USAGE_GPU_ONLY, VK_IMAGE_USAGE_STORAGE_BIT | VK_IMAGE_USAGE_SAMPLED_BIT, VK_SAMPLE_COUNT_1_BIT);
        m_gpu_resources->reflection_temporal_image->set_name("Reflection Temporal Image");

        m_gpu_resources->reflection_temporal_view = dw::vk::ImageView::create(m_vk_backend, m_gpu_resources->reflection_temporal_image, VK_IMAGE_VIEW_TYPE_2D, VK_IMAGE_ASPECT_COLOR_BIT);
        m_gpu_resources->reflection_temporal_view->set_name("Reflection Temporal Image View");

        m_gpu_resources->g_buffer_1 = dw::vk::Image::create(m_vk_backend, VK_IMAGE_TYPE_2D, m_width, m_height, 1, 1, 1, VK_FORMAT_R8G8B8A8_UNORM, VMA_MEMORY_USAGE_GPU_ONLY, VK_IMAGE_USAGE_STORAGE_BIT | VK_IMAGE_USAGE_SAMPLED_BIT | VK_IMAGE_USAGE_COLOR_ATTACHMENT_BIT | VK_IMAGE_USAGE_TRANSFER_SRC_BIT, VK_SAMPLE_COUNT_1_BIT);
        m_gpu_resources->g_buffer_1->set_name("G-Buffer 1 Image");

        m_gpu_resources->g_buffer_2 = dw::vk::Image::create(m_vk_backend, VK_IMAGE_TYPE_2D, m_width, m_height, 1, 1, 1, VK_FORMAT_R16G16B16A16_SFLOAT, VMA_MEMORY_USAGE_GPU_ONLY, VK_IMAGE_USAGE_STORAGE_BIT | VK_IMAGE_USAGE_SAMPLED_BIT | VK_IMAGE_USAGE_COLOR_ATTACHMENT_BIT | VK_IMAGE_USAGE_TRANSFER_SRC_BIT, VK_SAMPLE_COUNT_1_BIT);
        m_gpu_resources->g_buffer_2->set_name("G-Buffer 2 Image");

        m_gpu_resources->g_buffer_3 = dw::vk::Image::create(m_vk_backend, VK_IMAGE_TYPE_2D, m_width, m_height, 1, 1, 1, VK_FORMAT_R32G32B32A32_SFLOAT, VMA_MEMORY_USAGE_GPU_ONLY, VK_IMAGE_USAGE_STORAGE_BIT | VK_IMAGE_USAGE_SAMPLED_BIT | VK_IMAGE_USAGE_COLOR_ATTACHMENT_BIT | VK_IMAGE_USAGE_TRANSFER_SRC_BIT, VK_SAMPLE_COUNT_1_BIT);
        m_gpu_resources->g_buffer_3->set_name("G-Buffer 3 Image");

        for (int i = 0; i < 2; i++)
        {
            auto image = dw::vk::Image::create(m_vk_backend, VK_IMAGE_TYPE_2D, m_width, m_height, 1, 1, 1, VK_FORMAT_R32G32B32A32_SFLOAT, VMA_MEMORY_USAGE_GPU_ONLY, VK_IMAGE_USAGE_STORAGE_BIT | VK_IMAGE_USAGE_SAMPLED_BIT | VK_IMAGE_USAGE_COLOR_ATTACHMENT_BIT | VK_IMAGE_USAGE_TRANSFER_DST_BIT | VK_IMAGE_USAGE_TRANSFER_SRC_BIT, VK_SAMPLE_COUNT_1_BIT);
            image->set_name("G-Buffer Linear-Z Image " + std::to_string(i));

            auto image_view = dw::vk::ImageView::create(m_vk_backend, image, VK_IMAGE_VIEW_TYPE_2D, VK_IMAGE_ASPECT_COLOR_BIT);
            image_view->set_name("G-Buffer Linear-Z Image View " + std::to_string(i));

            m_gpu_resources->g_buffer_linear_z.push_back(image);
            m_gpu_resources->g_buffer_linear_z_view.push_back(image_view);
        }

        m_gpu_resources->g_buffer_depth = dw::vk::Image::create(m_vk_backend, VK_IMAGE_TYPE_2D, m_width, m_height, 1, 1, 1, m_vk_backend->swap_chain_depth_format(), VMA_MEMORY_USAGE_GPU_ONLY, VK_IMAGE_USAGE_SAMPLED_BIT | VK_IMAGE_USAGE_DEPTH_STENCIL_ATTACHMENT_BIT | VK_IMAGE_USAGE_TRANSFER_SRC_BIT, VK_SAMPLE_COUNT_1_BIT);
        m_gpu_resources->g_buffer_depth->set_name("G-Buffer Depth Image");

        m_gpu_resources->g_buffer_1_view = dw::vk::ImageView::create(m_vk_backend, m_gpu_resources->g_buffer_1, VK_IMAGE_VIEW_TYPE_2D, VK_IMAGE_ASPECT_COLOR_BIT);
        m_gpu_resources->g_buffer_1_view->set_name("G-Buffer 1 Image View");

        m_gpu_resources->g_buffer_2_view = dw::vk::ImageView::create(m_vk_backend, m_gpu_resources->g_buffer_2, VK_IMAGE_VIEW_TYPE_2D, VK_IMAGE_ASPECT_COLOR_BIT);
        m_gpu_resources->g_buffer_2_view->set_name("G-Buffer 2 Image View");

        m_gpu_resources->g_buffer_3_view = dw::vk::ImageView::create(m_vk_backend, m_gpu_resources->g_buffer_3, VK_IMAGE_VIEW_TYPE_2D, VK_IMAGE_ASPECT_COLOR_BIT);
        m_gpu_resources->g_buffer_3_view->set_name("G-Buffer 3 Image View");

        m_gpu_resources->g_buffer_depth_view = dw::vk::ImageView::create(m_vk_backend, m_gpu_resources->g_buffer_depth, VK_IMAGE_VIEW_TYPE_2D, VK_IMAGE_ASPECT_DEPTH_BIT);
        m_gpu_resources->g_buffer_depth_view->set_name("G-Buffer Depth Image View");

        // TAA
        for (int i = 0; i < 2; i++)
        {
            auto image = dw::vk::Image::create(m_vk_backend, VK_IMAGE_TYPE_2D, m_width, m_height, 1, 1, 1, VK_FORMAT_R16G16B16A16_SFLOAT, VMA_MEMORY_USAGE_GPU_ONLY, VK_IMAGE_USAGE_STORAGE_BIT | VK_IMAGE_USAGE_SAMPLED_BIT | VK_IMAGE_USAGE_TRANSFER_DST_BIT, VK_SAMPLE_COUNT_1_BIT);
            image->set_name("TAA Image " + std::to_string(i));

            m_gpu_resources->taa_image.push_back(image);

            auto image_view = dw::vk::ImageView::create(m_vk_backend, image, VK_IMAGE_VIEW_TYPE_2D, VK_IMAGE_ASPECT_COLOR_BIT);
            image_view->set_name("TAA Image View " + std::to_string(i));

            m_gpu_resources->taa_view.push_back(image_view);
        }

        // Downsampled G-Buffer
        for (int i = 0; i < 2; i++)
        {
            auto image = dw::vk::Image::create(m_vk_backend, VK_IMAGE_TYPE_2D, rt_image_width, rt_image_height, 1, 1, 1, VK_FORMAT_R32G32B32A32_SFLOAT, VMA_MEMORY_USAGE_GPU_ONLY, VK_IMAGE_USAGE_STORAGE_BIT | VK_IMAGE_USAGE_SAMPLED_BIT | VK_IMAGE_USAGE_COLOR_ATTACHMENT_BIT | VK_IMAGE_USAGE_TRANSFER_DST_BIT, VK_SAMPLE_COUNT_1_BIT);
            image->set_name("Downsampled G-Buffer Linear-Z Image " + std::to_string(i));

            auto image_view = dw::vk::ImageView::create(m_vk_backend, image, VK_IMAGE_VIEW_TYPE_2D, VK_IMAGE_ASPECT_COLOR_BIT);
            image_view->set_name("Downsampled G-Buffer Linear-Z Image View " + std::to_string(i));

            m_gpu_resources->downsampled_g_buffer_linear_z.push_back(image);
            m_gpu_resources->downsampled_g_buffer_linear_z_view.push_back(image_view);
        }

        m_gpu_resources->downsampled_g_buffer_depth = dw::vk::Image::create(m_vk_backend, VK_IMAGE_TYPE_2D, rt_image_width, rt_image_height, 1, 1, 1, m_vk_backend->swap_chain_depth_format(), VMA_MEMORY_USAGE_GPU_ONLY, VK_IMAGE_USAGE_SAMPLED_BIT | VK_IMAGE_USAGE_DEPTH_STENCIL_ATTACHMENT_BIT | VK_IMAGE_USAGE_TRANSFER_DST_BIT, VK_SAMPLE_COUNT_1_BIT);
        m_gpu_resources->downsampled_g_buffer_depth->set_name("Downsampled G-Buffer Depth Image");

        m_gpu_resources->downsampled_g_buffer_1 = dw::vk::Image::create(m_vk_backend, VK_IMAGE_TYPE_2D, rt_image_width, rt_image_height, 1, 1, 1, VK_FORMAT_R8G8B8A8_UNORM, VMA_MEMORY_USAGE_GPU_ONLY, VK_IMAGE_USAGE_STORAGE_BIT | VK_IMAGE_USAGE_SAMPLED_BIT | VK_IMAGE_USAGE_COLOR_ATTACHMENT_BIT | VK_IMAGE_USAGE_TRANSFER_DST_BIT, VK_SAMPLE_COUNT_1_BIT);
        m_gpu_resources->downsampled_g_buffer_1->set_name("Downsampled G-Buffer 1 Image");

        m_gpu_resources->downsampled_g_buffer_2 = dw::vk::Image::create(m_vk_backend, VK_IMAGE_TYPE_2D, rt_image_width, rt_image_height, 1, 1, 1, VK_FORMAT_R16G16B16A16_SFLOAT, VMA_MEMORY_USAGE_GPU_ONLY, VK_IMAGE_USAGE_STORAGE_BIT | VK_IMAGE_USAGE_SAMPLED_BIT | VK_IMAGE_USAGE_COLOR_ATTACHMENT_BIT | VK_IMAGE_USAGE_TRANSFER_DST_BIT, VK_SAMPLE_COUNT_1_BIT);
        m_gpu_resources->downsampled_g_buffer_2->set_name("Downsampled G-Buffer 2 Image");

        m_gpu_resources->downsampled_g_buffer_3 = dw::vk::Image::create(m_vk_backend, VK_IMAGE_TYPE_2D, rt_image_width, rt_image_height, 1, 1, 1, VK_FORMAT_R32G32B32A32_SFLOAT, VMA_MEMORY_USAGE_GPU_ONLY, VK_IMAGE_USAGE_STORAGE_BIT | VK_IMAGE_USAGE_SAMPLED_BIT | VK_IMAGE_USAGE_COLOR_ATTACHMENT_BIT | VK_IMAGE_USAGE_TRANSFER_DST_BIT, VK_SAMPLE_COUNT_1_BIT);
        m_gpu_resources->downsampled_g_buffer_3->set_name("Downsampled G-Buffer 3 Image");

        m_gpu_resources->downsampled_g_buffer_1_view = dw::vk::ImageView::create(m_vk_backend, m_gpu_resources->downsampled_g_buffer_1, VK_IMAGE_VIEW_TYPE_2D, VK_IMAGE_ASPECT_COLOR_BIT);
        m_gpu_resources->downsampled_g_buffer_1_view->set_name("Downsampled G-Buffer 1 Image View");

        m_gpu_resources->downsampled_g_buffer_2_view = dw::vk::ImageView::create(m_vk_backend, m_gpu_resources->downsampled_g_buffer_2, VK_IMAGE_VIEW_TYPE_2D, VK_IMAGE_ASPECT_COLOR_BIT);
        m_gpu_resources->downsampled_g_buffer_2_view->set_name("Downsampled G-Buffer 2 Image View");

        m_gpu_resources->downsampled_g_buffer_3_view = dw::vk::ImageView::create(m_vk_backend, m_gpu_resources->downsampled_g_buffer_3, VK_IMAGE_VIEW_TYPE_2D, VK_IMAGE_ASPECT_COLOR_BIT);
        m_gpu_resources->downsampled_g_buffer_3_view->set_name("Downsampled G-Buffer 3 Image View");

        m_gpu_resources->downsampled_g_buffer_depth_view = dw::vk::ImageView::create(m_vk_backend, m_gpu_resources->downsampled_g_buffer_depth, VK_IMAGE_VIEW_TYPE_2D, VK_IMAGE_ASPECT_DEPTH_BIT);
        m_gpu_resources->downsampled_g_buffer_depth_view->set_name("Downsampled G-Buffer Depth Image View");
    }

    // -----------------------------------------------------------------------------------------------------------------------------------

    void create_render_passes()
    {
        {
            std::vector<VkAttachmentDescription> attachments(5);

            // GBuffer1 attachment
            attachments[0].format         = VK_FORMAT_R8G8B8A8_UNORM;
            attachments[0].samples        = VK_SAMPLE_COUNT_1_BIT;
            attachments[0].loadOp         = VK_ATTACHMENT_LOAD_OP_CLEAR;
            attachments[0].storeOp        = VK_ATTACHMENT_STORE_OP_STORE;
            attachments[0].stencilLoadOp  = VK_ATTACHMENT_LOAD_OP_DONT_CARE;
            attachments[0].stencilStoreOp = VK_ATTACHMENT_STORE_OP_DONT_CARE;
            attachments[0].initialLayout  = VK_IMAGE_LAYOUT_UNDEFINED;
            attachments[0].finalLayout    = VK_IMAGE_LAYOUT_TRANSFER_SRC_OPTIMAL;

            // GBuffer2 attachment
            attachments[1].format         = VK_FORMAT_R16G16B16A16_SFLOAT;
            attachments[1].samples        = VK_SAMPLE_COUNT_1_BIT;
            attachments[1].loadOp         = VK_ATTACHMENT_LOAD_OP_CLEAR;
            attachments[1].storeOp        = VK_ATTACHMENT_STORE_OP_STORE;
            attachments[1].stencilLoadOp  = VK_ATTACHMENT_LOAD_OP_DONT_CARE;
            attachments[1].stencilStoreOp = VK_ATTACHMENT_STORE_OP_DONT_CARE;
            attachments[1].initialLayout  = VK_IMAGE_LAYOUT_UNDEFINED;
            attachments[1].finalLayout    = VK_IMAGE_LAYOUT_TRANSFER_SRC_OPTIMAL;

            // GBuffer3 attachment
            attachments[2].format         = VK_FORMAT_R32G32B32A32_SFLOAT;
            attachments[2].samples        = VK_SAMPLE_COUNT_1_BIT;
            attachments[2].loadOp         = VK_ATTACHMENT_LOAD_OP_CLEAR;
            attachments[2].storeOp        = VK_ATTACHMENT_STORE_OP_STORE;
            attachments[2].stencilLoadOp  = VK_ATTACHMENT_LOAD_OP_DONT_CARE;
            attachments[2].stencilStoreOp = VK_ATTACHMENT_STORE_OP_DONT_CARE;
            attachments[2].initialLayout  = VK_IMAGE_LAYOUT_UNDEFINED;
            attachments[2].finalLayout    = VK_IMAGE_LAYOUT_TRANSFER_SRC_OPTIMAL;

            // GBuffer Linear Z attachment
            attachments[3].format         = VK_FORMAT_R32G32B32A32_SFLOAT;
            attachments[3].samples        = VK_SAMPLE_COUNT_1_BIT;
            attachments[3].loadOp         = VK_ATTACHMENT_LOAD_OP_CLEAR;
            attachments[3].storeOp        = VK_ATTACHMENT_STORE_OP_STORE;
            attachments[3].stencilLoadOp  = VK_ATTACHMENT_LOAD_OP_DONT_CARE;
            attachments[3].stencilStoreOp = VK_ATTACHMENT_STORE_OP_DONT_CARE;
            attachments[3].initialLayout  = VK_IMAGE_LAYOUT_UNDEFINED;
            attachments[3].finalLayout    = VK_IMAGE_LAYOUT_TRANSFER_SRC_OPTIMAL;

            // Depth attachment
            attachments[4].format         = m_vk_backend->swap_chain_depth_format();
            attachments[4].samples        = VK_SAMPLE_COUNT_1_BIT;
            attachments[4].loadOp         = VK_ATTACHMENT_LOAD_OP_CLEAR;
            attachments[4].storeOp        = VK_ATTACHMENT_STORE_OP_STORE;
            attachments[4].stencilLoadOp  = VK_ATTACHMENT_LOAD_OP_CLEAR;
            attachments[4].stencilStoreOp = VK_ATTACHMENT_STORE_OP_DONT_CARE;
            attachments[4].initialLayout  = VK_IMAGE_LAYOUT_UNDEFINED;
            attachments[4].finalLayout    = VK_IMAGE_LAYOUT_TRANSFER_SRC_OPTIMAL;

            VkAttachmentReference gbuffer_references[4];

            gbuffer_references[0].attachment = 0;
            gbuffer_references[0].layout     = VK_IMAGE_LAYOUT_COLOR_ATTACHMENT_OPTIMAL;

            gbuffer_references[1].attachment = 1;
            gbuffer_references[1].layout     = VK_IMAGE_LAYOUT_COLOR_ATTACHMENT_OPTIMAL;

            gbuffer_references[2].attachment = 2;
            gbuffer_references[2].layout     = VK_IMAGE_LAYOUT_COLOR_ATTACHMENT_OPTIMAL;

            gbuffer_references[3].attachment = 3;
            gbuffer_references[3].layout     = VK_IMAGE_LAYOUT_COLOR_ATTACHMENT_OPTIMAL;

            VkAttachmentReference depth_reference;
            depth_reference.attachment = 4;
            depth_reference.layout     = VK_IMAGE_LAYOUT_DEPTH_STENCIL_ATTACHMENT_OPTIMAL;

            std::vector<VkSubpassDescription> subpass_description(1);

            subpass_description[0].pipelineBindPoint       = VK_PIPELINE_BIND_POINT_GRAPHICS;
            subpass_description[0].colorAttachmentCount    = 4;
            subpass_description[0].pColorAttachments       = gbuffer_references;
            subpass_description[0].pDepthStencilAttachment = &depth_reference;
            subpass_description[0].inputAttachmentCount    = 0;
            subpass_description[0].pInputAttachments       = nullptr;
            subpass_description[0].preserveAttachmentCount = 0;
            subpass_description[0].pPreserveAttachments    = nullptr;
            subpass_description[0].pResolveAttachments     = nullptr;

            // Subpass dependencies for layout transitions
            std::vector<VkSubpassDependency> dependencies(2);

            dependencies[0].srcSubpass      = VK_SUBPASS_EXTERNAL;
            dependencies[0].dstSubpass      = 0;
            dependencies[0].srcStageMask    = VK_PIPELINE_STAGE_BOTTOM_OF_PIPE_BIT;
            dependencies[0].dstStageMask    = VK_PIPELINE_STAGE_COLOR_ATTACHMENT_OUTPUT_BIT;
            dependencies[0].srcAccessMask   = VK_ACCESS_MEMORY_READ_BIT;
            dependencies[0].dstAccessMask   = VK_ACCESS_COLOR_ATTACHMENT_READ_BIT | VK_ACCESS_COLOR_ATTACHMENT_WRITE_BIT;
            dependencies[0].dependencyFlags = VK_DEPENDENCY_BY_REGION_BIT;

            dependencies[1].srcSubpass      = 0;
            dependencies[1].dstSubpass      = VK_SUBPASS_EXTERNAL;
            dependencies[1].srcStageMask    = VK_PIPELINE_STAGE_COLOR_ATTACHMENT_OUTPUT_BIT;
            dependencies[1].dstStageMask    = VK_PIPELINE_STAGE_BOTTOM_OF_PIPE_BIT;
            dependencies[1].srcAccessMask   = VK_ACCESS_COLOR_ATTACHMENT_READ_BIT | VK_ACCESS_COLOR_ATTACHMENT_WRITE_BIT;
            dependencies[1].dstAccessMask   = VK_ACCESS_MEMORY_READ_BIT;
            dependencies[1].dependencyFlags = VK_DEPENDENCY_BY_REGION_BIT;

            m_gpu_resources->g_buffer_rp = dw::vk::RenderPass::create(m_vk_backend, attachments, subpass_description, dependencies);
        }

        {
            std::vector<VkAttachmentDescription> attachments(1);

            // Deferred attachment
            attachments[0].format         = VK_FORMAT_R16G16B16A16_SFLOAT;
            attachments[0].samples        = VK_SAMPLE_COUNT_1_BIT;
            attachments[0].loadOp         = VK_ATTACHMENT_LOAD_OP_CLEAR;
            attachments[0].storeOp        = VK_ATTACHMENT_STORE_OP_STORE;
            attachments[0].stencilLoadOp  = VK_ATTACHMENT_LOAD_OP_DONT_CARE;
            attachments[0].stencilStoreOp = VK_ATTACHMENT_STORE_OP_DONT_CARE;
            attachments[0].initialLayout  = VK_IMAGE_LAYOUT_UNDEFINED;
            attachments[0].finalLayout    = VK_IMAGE_LAYOUT_COLOR_ATTACHMENT_OPTIMAL;

            VkAttachmentReference deferred_reference;

            deferred_reference.attachment = 0;
            deferred_reference.layout     = VK_IMAGE_LAYOUT_COLOR_ATTACHMENT_OPTIMAL;

            std::vector<VkSubpassDescription> subpass_description(1);

            subpass_description[0].pipelineBindPoint       = VK_PIPELINE_BIND_POINT_GRAPHICS;
            subpass_description[0].colorAttachmentCount    = 1;
            subpass_description[0].pColorAttachments       = &deferred_reference;
            subpass_description[0].pDepthStencilAttachment = nullptr;
            subpass_description[0].inputAttachmentCount    = 0;
            subpass_description[0].pInputAttachments       = nullptr;
            subpass_description[0].preserveAttachmentCount = 0;
            subpass_description[0].pPreserveAttachments    = nullptr;
            subpass_description[0].pResolveAttachments     = nullptr;

            // Subpass dependencies for layout transitions
            std::vector<VkSubpassDependency> dependencies(2);

            dependencies[0].srcSubpass      = VK_SUBPASS_EXTERNAL;
            dependencies[0].dstSubpass      = 0;
            dependencies[0].srcStageMask    = VK_PIPELINE_STAGE_BOTTOM_OF_PIPE_BIT;
            dependencies[0].dstStageMask    = VK_PIPELINE_STAGE_COLOR_ATTACHMENT_OUTPUT_BIT;
            dependencies[0].srcAccessMask   = VK_ACCESS_MEMORY_READ_BIT;
            dependencies[0].dstAccessMask   = VK_ACCESS_COLOR_ATTACHMENT_READ_BIT | VK_ACCESS_COLOR_ATTACHMENT_WRITE_BIT;
            dependencies[0].dependencyFlags = VK_DEPENDENCY_BY_REGION_BIT;

            dependencies[1].srcSubpass      = 0;
            dependencies[1].dstSubpass      = VK_SUBPASS_EXTERNAL;
            dependencies[1].srcStageMask    = VK_PIPELINE_STAGE_COLOR_ATTACHMENT_OUTPUT_BIT;
            dependencies[1].dstStageMask    = VK_PIPELINE_STAGE_BOTTOM_OF_PIPE_BIT;
            dependencies[1].srcAccessMask   = VK_ACCESS_COLOR_ATTACHMENT_READ_BIT | VK_ACCESS_COLOR_ATTACHMENT_WRITE_BIT;
            dependencies[1].dstAccessMask   = VK_ACCESS_MEMORY_READ_BIT;
            dependencies[1].dependencyFlags = VK_DEPENDENCY_BY_REGION_BIT;

            m_gpu_resources->deferred_rp = dw::vk::RenderPass::create(m_vk_backend, attachments, subpass_description, dependencies);
        }

        {
            std::vector<VkAttachmentDescription> attachments(2);

            // Deferred attachment
            attachments[0].format         = VK_FORMAT_R16G16B16A16_SFLOAT;
            attachments[0].samples        = VK_SAMPLE_COUNT_1_BIT;
            attachments[0].loadOp         = VK_ATTACHMENT_LOAD_OP_LOAD;
            attachments[0].storeOp        = VK_ATTACHMENT_STORE_OP_STORE;
            attachments[0].stencilLoadOp  = VK_ATTACHMENT_LOAD_OP_DONT_CARE;
            attachments[0].stencilStoreOp = VK_ATTACHMENT_STORE_OP_DONT_CARE;
            attachments[0].initialLayout  = VK_IMAGE_LAYOUT_COLOR_ATTACHMENT_OPTIMAL;
            attachments[0].finalLayout    = VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL;

            // Depth attachment
            attachments[1].format         = m_vk_backend->swap_chain_depth_format();
            attachments[1].samples        = VK_SAMPLE_COUNT_1_BIT;
            attachments[1].loadOp         = VK_ATTACHMENT_LOAD_OP_LOAD;
            attachments[1].storeOp        = VK_ATTACHMENT_STORE_OP_DONT_CARE;
            attachments[1].stencilLoadOp  = VK_ATTACHMENT_LOAD_OP_DONT_CARE;
            attachments[1].stencilStoreOp = VK_ATTACHMENT_STORE_OP_DONT_CARE;
            attachments[1].initialLayout  = VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL;
            attachments[1].finalLayout    = VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL;

            VkAttachmentReference deferred_reference;

            deferred_reference.attachment = 0;
            deferred_reference.layout     = VK_IMAGE_LAYOUT_COLOR_ATTACHMENT_OPTIMAL;

            VkAttachmentReference depth_reference;
            depth_reference.attachment = 1;
            depth_reference.layout     = VK_IMAGE_LAYOUT_DEPTH_STENCIL_ATTACHMENT_OPTIMAL;

            std::vector<VkSubpassDescription> subpass_description(1);

            subpass_description[0].pipelineBindPoint       = VK_PIPELINE_BIND_POINT_GRAPHICS;
            subpass_description[0].colorAttachmentCount    = 1;
            subpass_description[0].pColorAttachments       = &deferred_reference;
            subpass_description[0].pDepthStencilAttachment = &depth_reference;
            subpass_description[0].inputAttachmentCount    = 0;
            subpass_description[0].pInputAttachments       = nullptr;
            subpass_description[0].preserveAttachmentCount = 0;
            subpass_description[0].pPreserveAttachments    = nullptr;
            subpass_description[0].pResolveAttachments     = nullptr;

            // Subpass dependencies for layout transitions
            std::vector<VkSubpassDependency> dependencies(2);

            dependencies[0].srcSubpass      = VK_SUBPASS_EXTERNAL;
            dependencies[0].dstSubpass      = 0;
            dependencies[0].srcStageMask    = VK_PIPELINE_STAGE_BOTTOM_OF_PIPE_BIT;
            dependencies[0].dstStageMask    = VK_PIPELINE_STAGE_COLOR_ATTACHMENT_OUTPUT_BIT;
            dependencies[0].srcAccessMask   = VK_ACCESS_MEMORY_READ_BIT;
            dependencies[0].dstAccessMask   = VK_ACCESS_COLOR_ATTACHMENT_READ_BIT | VK_ACCESS_COLOR_ATTACHMENT_WRITE_BIT;
            dependencies[0].dependencyFlags = VK_DEPENDENCY_BY_REGION_BIT;

            dependencies[1].srcSubpass      = 0;
            dependencies[1].dstSubpass      = VK_SUBPASS_EXTERNAL;
            dependencies[1].srcStageMask    = VK_PIPELINE_STAGE_COLOR_ATTACHMENT_OUTPUT_BIT;
            dependencies[1].dstStageMask    = VK_PIPELINE_STAGE_BOTTOM_OF_PIPE_BIT;
            dependencies[1].srcAccessMask   = VK_ACCESS_COLOR_ATTACHMENT_READ_BIT | VK_ACCESS_COLOR_ATTACHMENT_WRITE_BIT;
            dependencies[1].dstAccessMask   = VK_ACCESS_MEMORY_READ_BIT;
            dependencies[1].dependencyFlags = VK_DEPENDENCY_BY_REGION_BIT;

            m_gpu_resources->skybox_rp = dw::vk::RenderPass::create(m_vk_backend, attachments, subpass_description, dependencies);
        }
    }

    // -----------------------------------------------------------------------------------------------------------------------------------

    void create_framebuffers()
    {
        m_gpu_resources->g_buffer_fbo.clear();

        for (int i = 0; i < 2; i++)
            m_gpu_resources->g_buffer_fbo.push_back(dw::vk::Framebuffer::create(m_vk_backend, m_gpu_resources->g_buffer_rp, { m_gpu_resources->g_buffer_1_view, m_gpu_resources->g_buffer_2_view, m_gpu_resources->g_buffer_3_view, m_gpu_resources->g_buffer_linear_z_view[i], m_gpu_resources->g_buffer_depth_view }, m_width, m_height, 1));

        m_gpu_resources->deferred_fbo.reset();
        m_gpu_resources->deferred_fbo = dw::vk::Framebuffer::create(m_vk_backend, m_gpu_resources->deferred_rp, { m_gpu_resources->deferred_view }, m_width, m_height, 1);

        m_gpu_resources->skybox_fbo.reset();
        m_gpu_resources->skybox_fbo = dw::vk::Framebuffer::create(m_vk_backend, m_gpu_resources->skybox_rp, { m_gpu_resources->deferred_view, m_gpu_resources->g_buffer_depth_view }, m_width, m_height, 1);
    }

    // -----------------------------------------------------------------------------------------------------------------------------------

    bool create_uniform_buffer()
    {
        m_ubo_size           = m_vk_backend->aligned_dynamic_ubo_size(sizeof(UBO));
        m_gpu_resources->ubo = dw::vk::Buffer::create(m_vk_backend, VK_BUFFER_USAGE_UNIFORM_BUFFER_BIT, m_ubo_size * dw::vk::Backend::kMaxFramesInFlight, VMA_MEMORY_USAGE_CPU_TO_GPU, VMA_ALLOCATION_CREATE_MAPPED_BIT);

        return true;
    }

    // -----------------------------------------------------------------------------------------------------------------------------------

    void create_descriptor_set_layouts()
    {
        {
            dw::vk::DescriptorSetLayout::Desc desc;

            desc.add_binding(0, VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER_DYNAMIC, 1, VK_SHADER_STAGE_RAYGEN_BIT_KHR | VK_SHADER_STAGE_CLOSEST_HIT_BIT_KHR | VK_SHADER_STAGE_FRAGMENT_BIT | VK_SHADER_STAGE_VERTEX_BIT | VK_SHADER_STAGE_COMPUTE_BIT);
            desc.add_binding(1, VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER, 1, VK_SHADER_STAGE_RAYGEN_BIT_KHR | VK_SHADER_STAGE_CLOSEST_HIT_BIT_KHR | VK_SHADER_STAGE_FRAGMENT_BIT | VK_SHADER_STAGE_VERTEX_BIT | VK_SHADER_STAGE_COMPUTE_BIT);
            desc.add_binding(2, VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER, 1, VK_SHADER_STAGE_RAYGEN_BIT_KHR | VK_SHADER_STAGE_CLOSEST_HIT_BIT_KHR | VK_SHADER_STAGE_FRAGMENT_BIT | VK_SHADER_STAGE_VERTEX_BIT | VK_SHADER_STAGE_COMPUTE_BIT);

            m_gpu_resources->per_frame_ds_layout = dw::vk::DescriptorSetLayout::create(m_vk_backend, desc);
        }

        {
            dw::vk::DescriptorSetLayout::Desc desc;

            desc.add_binding(0, VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER, 1, VK_SHADER_STAGE_FRAGMENT_BIT | VK_SHADER_STAGE_CLOSEST_HIT_BIT_KHR | VK_SHADER_STAGE_MISS_BIT_KHR);
            desc.add_binding(1, VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER, 1, VK_SHADER_STAGE_FRAGMENT_BIT | VK_SHADER_STAGE_CLOSEST_HIT_BIT_KHR | VK_SHADER_STAGE_MISS_BIT_KHR);
            desc.add_binding(2, VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER, 1, VK_SHADER_STAGE_FRAGMENT_BIT | VK_SHADER_STAGE_CLOSEST_HIT_BIT_KHR | VK_SHADER_STAGE_MISS_BIT_KHR);

            m_gpu_resources->pbr_ds_layout = dw::vk::DescriptorSetLayout::create(m_vk_backend, desc);
        }

        {
            dw::vk::DescriptorSetLayout::Desc desc;

            desc.add_binding(0, VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER, 1, VK_SHADER_STAGE_RAYGEN_BIT_KHR | VK_SHADER_STAGE_CLOSEST_HIT_BIT_KHR | VK_SHADER_STAGE_FRAGMENT_BIT | VK_SHADER_STAGE_COMPUTE_BIT);
            desc.add_binding(1, VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER, 1, VK_SHADER_STAGE_RAYGEN_BIT_KHR | VK_SHADER_STAGE_CLOSEST_HIT_BIT_KHR | VK_SHADER_STAGE_FRAGMENT_BIT | VK_SHADER_STAGE_COMPUTE_BIT);
            desc.add_binding(2, VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER, 1, VK_SHADER_STAGE_RAYGEN_BIT_KHR | VK_SHADER_STAGE_CLOSEST_HIT_BIT_KHR | VK_SHADER_STAGE_FRAGMENT_BIT | VK_SHADER_STAGE_COMPUTE_BIT);
            desc.add_binding(3, VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER, 1, VK_SHADER_STAGE_RAYGEN_BIT_KHR | VK_SHADER_STAGE_CLOSEST_HIT_BIT_KHR | VK_SHADER_STAGE_FRAGMENT_BIT | VK_SHADER_STAGE_COMPUTE_BIT);
            desc.add_binding(4, VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER, 1, VK_SHADER_STAGE_RAYGEN_BIT_KHR | VK_SHADER_STAGE_CLOSEST_HIT_BIT_KHR | VK_SHADER_STAGE_FRAGMENT_BIT | VK_SHADER_STAGE_COMPUTE_BIT);

            m_gpu_resources->g_buffer_ds_layout = dw::vk::DescriptorSetLayout::create(m_vk_backend, desc);
        }

        {
            dw::vk::DescriptorSetLayout::Desc desc;

            desc.add_binding(0, VK_DESCRIPTOR_TYPE_STORAGE_IMAGE, 1, VK_SHADER_STAGE_RAYGEN_BIT_KHR | VK_SHADER_STAGE_COMPUTE_BIT);

            m_gpu_resources->storage_image_ds_layout = dw::vk::DescriptorSetLayout::create(m_vk_backend, desc);
        }

        {
            dw::vk::DescriptorSetLayout::Desc desc;

            desc.add_binding(0, VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER, 1, VK_SHADER_STAGE_RAYGEN_BIT_KHR | VK_SHADER_STAGE_CLOSEST_HIT_BIT_KHR | VK_SHADER_STAGE_MISS_BIT_KHR | VK_SHADER_STAGE_FRAGMENT_BIT | VK_SHADER_STAGE_COMPUTE_BIT);

            m_gpu_resources->combined_sampler_ds_layout = dw::vk::DescriptorSetLayout::create(m_vk_backend, desc);
        }
    }

    // -----------------------------------------------------------------------------------------------------------------------------------

    void create_descriptor_sets()
    {
        m_gpu_resources->per_frame_ds                 = m_vk_backend->allocate_descriptor_set(m_gpu_resources->per_frame_ds_layout);
        m_gpu_resources->skybox_ds                    = m_vk_backend->allocate_descriptor_set(m_gpu_resources->combined_sampler_ds_layout);
        m_gpu_resources->pbr_ds                       = m_vk_backend->allocate_descriptor_set(m_gpu_resources->pbr_ds_layout);
        m_gpu_resources->reflection_rt_write_ds       = m_vk_backend->allocate_descriptor_set(m_gpu_resources->storage_image_ds_layout);
        m_gpu_resources->reflection_rt_read_ds        = m_vk_backend->allocate_descriptor_set(m_gpu_resources->combined_sampler_ds_layout);
        m_gpu_resources->reflection_resolve_write_ds  = m_vk_backend->allocate_descriptor_set(m_gpu_resources->storage_image_ds_layout);
        m_gpu_resources->reflection_resolve_read_ds   = m_vk_backend->allocate_descriptor_set(m_gpu_resources->combined_sampler_ds_layout);
        m_gpu_resources->reflection_temporal_write_ds = m_vk_backend->allocate_descriptor_set(m_gpu_resources->storage_image_ds_layout);
        m_gpu_resources->reflection_temporal_read_ds  = m_vk_backend->allocate_descriptor_set(m_gpu_resources->combined_sampler_ds_layout);
        m_gpu_resources->rtgi_write_ds                = m_vk_backend->allocate_descriptor_set(m_gpu_resources->storage_image_ds_layout);
        m_gpu_resources->rtgi_read_ds                 = m_vk_backend->allocate_descriptor_set(m_gpu_resources->combined_sampler_ds_layout);
        m_gpu_resources->deferred_read_ds             = m_vk_backend->allocate_descriptor_set(m_gpu_resources->combined_sampler_ds_layout);
        m_gpu_resources->visibility_write_ds          = m_vk_backend->allocate_descriptor_set(m_gpu_resources->storage_image_ds_layout);
        m_gpu_resources->visibility_read_ds           = m_vk_backend->allocate_descriptor_set(m_gpu_resources->combined_sampler_ds_layout);

        for (int i = 0; i < 2; i++)
        {
            m_gpu_resources->taa_read_ds.push_back(m_vk_backend->allocate_descriptor_set(m_gpu_resources->combined_sampler_ds_layout));
            m_gpu_resources->taa_write_ds.push_back(m_vk_backend->allocate_descriptor_set(m_gpu_resources->storage_image_ds_layout));
            m_gpu_resources->g_buffer_ds.push_back(m_vk_backend->allocate_descriptor_set(m_gpu_resources->g_buffer_ds_layout));
            m_gpu_resources->downsampled_g_buffer_ds.push_back(m_vk_backend->allocate_descriptor_set(m_gpu_resources->g_buffer_ds_layout));
            m_gpu_resources->reflection_blur_write_ds.push_back(m_vk_backend->allocate_descriptor_set(m_gpu_resources->storage_image_ds_layout));
            m_gpu_resources->reflection_blur_read_ds.push_back(m_vk_backend->allocate_descriptor_set(m_gpu_resources->combined_sampler_ds_layout));
            m_gpu_resources->reflection_history_ds.push_back(m_vk_backend->allocate_descriptor_set(m_gpu_resources->storage_image_ds_layout));
        }
    }

    // -----------------------------------------------------------------------------------------------------------------------------------

    void write_descriptor_sets()
    {
        // Per-frame
        {
            std::vector<VkWriteDescriptorSet> write_datas;

            {
                VkDescriptorBufferInfo buffer_info;

                buffer_info.range  = VK_WHOLE_SIZE;
                buffer_info.offset = 0;
                buffer_info.buffer = m_gpu_resources->ubo->handle();

                VkWriteDescriptorSet write_data;
                DW_ZERO_MEMORY(write_data);

                write_data.sType           = VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET;
                write_data.descriptorCount = 1;
                write_data.descriptorType  = VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER_DYNAMIC;
                write_data.pBufferInfo     = &buffer_info;
                write_data.dstBinding      = 0;
                write_data.dstSet          = m_gpu_resources->per_frame_ds->handle();

                write_datas.push_back(write_data);
            }

            {
                VkDescriptorImageInfo image_info;

                image_info.sampler     = m_vk_backend->nearest_sampler()->handle();
                image_info.imageView   = m_gpu_resources->blue_noise_view_1->handle();
                image_info.imageLayout = VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL;

                VkWriteDescriptorSet write_data;
                DW_ZERO_MEMORY(write_data);

                write_data.sType           = VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET;
                write_data.descriptorCount = 1;
                write_data.descriptorType  = VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER;
                write_data.pImageInfo      = &image_info;
                write_data.dstBinding      = 1;
                write_data.dstSet          = m_gpu_resources->per_frame_ds->handle();

                write_datas.push_back(write_data);
            }

            {
                VkDescriptorImageInfo image_info;

                image_info.sampler     = m_vk_backend->nearest_sampler()->handle();
                image_info.imageView   = m_gpu_resources->blue_noise_view_2->handle();
                image_info.imageLayout = VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL;

                VkWriteDescriptorSet write_data;
                DW_ZERO_MEMORY(write_data);

                write_data.sType           = VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET;
                write_data.descriptorCount = 1;
                write_data.descriptorType  = VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER;
                write_data.pImageInfo      = &image_info;
                write_data.dstBinding      = 2;
                write_data.dstSet          = m_gpu_resources->per_frame_ds->handle();

                write_datas.push_back(write_data);
            }

            vkUpdateDescriptorSets(m_vk_backend->device(), write_datas.size(), write_datas.data(), 0, nullptr);
        }

        // G-Buffer
        {
            for (int i = 0; i < 2; i++)
            {
                VkDescriptorImageInfo image_info[5];

                image_info[0].sampler     = m_vk_backend->nearest_sampler()->handle();
                image_info[0].imageView   = m_gpu_resources->g_buffer_1_view->handle();
                image_info[0].imageLayout = VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL;

                image_info[1].sampler     = m_vk_backend->nearest_sampler()->handle();
                image_info[1].imageView   = m_gpu_resources->g_buffer_2_view->handle();
                image_info[1].imageLayout = VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL;

                image_info[2].sampler     = m_vk_backend->nearest_sampler()->handle();
                image_info[2].imageView   = m_gpu_resources->g_buffer_3_view->handle();
                image_info[2].imageLayout = VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL;

                image_info[3].sampler     = m_vk_backend->nearest_sampler()->handle();
                image_info[3].imageView   = m_gpu_resources->g_buffer_depth_view->handle();
                image_info[3].imageLayout = VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL;

                image_info[4].sampler     = m_vk_backend->nearest_sampler()->handle();
                image_info[4].imageView   = m_gpu_resources->g_buffer_linear_z_view[i]->handle();
                image_info[4].imageLayout = VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL;

                VkWriteDescriptorSet write_data[5];
                DW_ZERO_MEMORY(write_data[0]);
                DW_ZERO_MEMORY(write_data[1]);
                DW_ZERO_MEMORY(write_data[2]);
                DW_ZERO_MEMORY(write_data[3]);
                DW_ZERO_MEMORY(write_data[4]);

                write_data[0].sType           = VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET;
                write_data[0].descriptorCount = 1;
                write_data[0].descriptorType  = VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER;
                write_data[0].pImageInfo      = &image_info[0];
                write_data[0].dstBinding      = 0;
                write_data[0].dstSet          = m_gpu_resources->g_buffer_ds[i]->handle();

                write_data[1].sType           = VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET;
                write_data[1].descriptorCount = 1;
                write_data[1].descriptorType  = VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER;
                write_data[1].pImageInfo      = &image_info[1];
                write_data[1].dstBinding      = 1;
                write_data[1].dstSet          = m_gpu_resources->g_buffer_ds[i]->handle();

                write_data[2].sType           = VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET;
                write_data[2].descriptorCount = 1;
                write_data[2].descriptorType  = VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER;
                write_data[2].pImageInfo      = &image_info[2];
                write_data[2].dstBinding      = 2;
                write_data[2].dstSet          = m_gpu_resources->g_buffer_ds[i]->handle();

                write_data[3].sType           = VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET;
                write_data[3].descriptorCount = 1;
                write_data[3].descriptorType  = VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER;
                write_data[3].pImageInfo      = &image_info[3];
                write_data[3].dstBinding      = 3;
                write_data[3].dstSet          = m_gpu_resources->g_buffer_ds[i]->handle();

                write_data[4].sType           = VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET;
                write_data[4].descriptorCount = 1;
                write_data[4].descriptorType  = VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER;
                write_data[4].pImageInfo      = &image_info[4];
                write_data[4].dstBinding      = 4;
                write_data[4].dstSet          = m_gpu_resources->g_buffer_ds[i]->handle();

                vkUpdateDescriptorSets(m_vk_backend->device(), 5, &write_data[0], 0, nullptr);
            }
        }

        // Downsampled G-Buffer
        {
            for (int i = 0; i < 2; i++)
            {
                VkDescriptorImageInfo image_info[5];

                image_info[0].sampler     = m_vk_backend->nearest_sampler()->handle();
                image_info[0].imageView   = m_gpu_resources->downsampled_g_buffer_1_view->handle();
                image_info[0].imageLayout = VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL;

                image_info[1].sampler     = m_vk_backend->nearest_sampler()->handle();
                image_info[1].imageView   = m_gpu_resources->downsampled_g_buffer_2_view->handle();
                image_info[1].imageLayout = VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL;

                image_info[2].sampler     = m_vk_backend->nearest_sampler()->handle();
                image_info[2].imageView   = m_gpu_resources->downsampled_g_buffer_3_view->handle();
                image_info[2].imageLayout = VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL;

                image_info[3].sampler     = m_vk_backend->nearest_sampler()->handle();
                image_info[3].imageView   = m_gpu_resources->downsampled_g_buffer_depth_view->handle();
                image_info[3].imageLayout = VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL;

                image_info[4].sampler     = m_vk_backend->nearest_sampler()->handle();
                image_info[4].imageView   = m_gpu_resources->downsampled_g_buffer_linear_z_view[i]->handle();
                image_info[4].imageLayout = VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL;

                VkWriteDescriptorSet write_data[5];
                DW_ZERO_MEMORY(write_data[0]);
                DW_ZERO_MEMORY(write_data[1]);
                DW_ZERO_MEMORY(write_data[2]);
                DW_ZERO_MEMORY(write_data[3]);
                DW_ZERO_MEMORY(write_data[4]);

                write_data[0].sType           = VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET;
                write_data[0].descriptorCount = 1;
                write_data[0].descriptorType  = VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER;
                write_data[0].pImageInfo      = &image_info[0];
                write_data[0].dstBinding      = 0;
                write_data[0].dstSet          = m_gpu_resources->downsampled_g_buffer_ds[i]->handle();

                write_data[1].sType           = VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET;
                write_data[1].descriptorCount = 1;
                write_data[1].descriptorType  = VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER;
                write_data[1].pImageInfo      = &image_info[1];
                write_data[1].dstBinding      = 1;
                write_data[1].dstSet          = m_gpu_resources->downsampled_g_buffer_ds[i]->handle();

                write_data[2].sType           = VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET;
                write_data[2].descriptorCount = 1;
                write_data[2].descriptorType  = VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER;
                write_data[2].pImageInfo      = &image_info[2];
                write_data[2].dstBinding      = 2;
                write_data[2].dstSet          = m_gpu_resources->downsampled_g_buffer_ds[i]->handle();

                write_data[3].sType           = VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET;
                write_data[3].descriptorCount = 1;
                write_data[3].descriptorType  = VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER;
                write_data[3].pImageInfo      = &image_info[3];
                write_data[3].dstBinding      = 3;
                write_data[3].dstSet          = m_gpu_resources->downsampled_g_buffer_ds[i]->handle();

                write_data[4].sType           = VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET;
                write_data[4].descriptorCount = 1;
                write_data[4].descriptorType  = VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER;
                write_data[4].pImageInfo      = &image_info[4];
                write_data[4].dstBinding      = 4;
                write_data[4].dstSet          = m_gpu_resources->downsampled_g_buffer_ds[i]->handle();

                vkUpdateDescriptorSets(m_vk_backend->device(), 5, &write_data[0], 0, nullptr);
            }
        }

        // PBR resources
        {
            VkDescriptorImageInfo image_info[3];

            image_info[0].sampler     = m_vk_backend->nearest_sampler()->handle();
            image_info[0].imageView   = m_gpu_resources->cubemap_sh_projection->image_view()->handle();
            image_info[0].imageLayout = VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL;

            image_info[1].sampler     = m_vk_backend->trilinear_sampler()->handle();
            image_info[1].imageView   = m_gpu_resources->cubemap_prefilter->image_view()->handle();
            image_info[1].imageLayout = VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL;

            image_info[2].sampler     = m_vk_backend->bilinear_sampler()->handle();
            image_info[2].imageView   = m_gpu_resources->brdf_preintegrate_lut->image_view()->handle();
            image_info[2].imageLayout = VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL;

            VkWriteDescriptorSet write_data[3];
            DW_ZERO_MEMORY(write_data[0]);
            DW_ZERO_MEMORY(write_data[1]);
            DW_ZERO_MEMORY(write_data[2]);

            write_data[0].sType           = VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET;
            write_data[0].descriptorCount = 1;
            write_data[0].descriptorType  = VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER;
            write_data[0].pImageInfo      = &image_info[0];
            write_data[0].dstBinding      = 0;
            write_data[0].dstSet          = m_gpu_resources->pbr_ds->handle();

            write_data[1].sType           = VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET;
            write_data[1].descriptorCount = 1;
            write_data[1].descriptorType  = VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER;
            write_data[1].pImageInfo      = &image_info[1];
            write_data[1].dstBinding      = 1;
            write_data[1].dstSet          = m_gpu_resources->pbr_ds->handle();

            write_data[2].sType           = VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET;
            write_data[2].descriptorCount = 1;
            write_data[2].descriptorType  = VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER;
            write_data[2].pImageInfo      = &image_info[2];
            write_data[2].dstBinding      = 2;
            write_data[2].dstSet          = m_gpu_resources->pbr_ds->handle();

            vkUpdateDescriptorSets(m_vk_backend->device(), 3, &write_data[0], 0, nullptr);
        }

        // Visibility write
        {
            std::vector<VkDescriptorImageInfo> image_infos;
            std::vector<VkWriteDescriptorSet>  write_datas;
            VkWriteDescriptorSet               write_data;

            image_infos.reserve(1);
            write_datas.reserve(1);

            VkDescriptorImageInfo storage_image_info;

            storage_image_info.sampler     = VK_NULL_HANDLE;
            storage_image_info.imageView   = m_gpu_resources->visibility_view->handle();
            storage_image_info.imageLayout = VK_IMAGE_LAYOUT_GENERAL;

            image_infos.push_back(storage_image_info);

            DW_ZERO_MEMORY(write_data);

            write_data.sType           = VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET;
            write_data.descriptorCount = 1;
            write_data.descriptorType  = VK_DESCRIPTOR_TYPE_STORAGE_IMAGE;
            write_data.pImageInfo      = &image_infos.back();
            write_data.dstBinding      = 0;
            write_data.dstSet          = m_gpu_resources->visibility_write_ds->handle();

            write_datas.push_back(write_data);

            vkUpdateDescriptorSets(m_vk_backend->device(), write_datas.size(), write_datas.data(), 0, nullptr);
        }

        // Visibility read
        {
            std::vector<VkDescriptorImageInfo> image_infos;
            std::vector<VkWriteDescriptorSet>  write_datas;
            VkWriteDescriptorSet               write_data;

            image_infos.reserve(1);
            write_datas.reserve(1);

            VkDescriptorImageInfo sampler_image_info;

            sampler_image_info.sampler     = m_vk_backend->nearest_sampler()->handle();
            sampler_image_info.imageView   = m_gpu_resources->visibility_view->handle();
            sampler_image_info.imageLayout = VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL;

            image_infos.push_back(sampler_image_info);

            DW_ZERO_MEMORY(write_data);

            write_data.sType           = VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET;
            write_data.descriptorCount = 1;
            write_data.descriptorType  = VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER;
            write_data.pImageInfo      = &image_infos.back();
            write_data.dstBinding      = 0;
            write_data.dstSet          = m_gpu_resources->visibility_read_ds->handle();

            write_datas.push_back(write_data);

            vkUpdateDescriptorSets(m_vk_backend->device(), write_datas.size(), write_datas.data(), 0, nullptr);
        }

        // Skybox
        {
            VkDescriptorImageInfo cubemap_image;

            cubemap_image.sampler     = m_vk_backend->bilinear_sampler()->handle();
            cubemap_image.imageView   = m_gpu_resources->hosek_wilkie_sky_model->image_view()->handle();
            cubemap_image.imageLayout = VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL;

            VkWriteDescriptorSet write_data;
            DW_ZERO_MEMORY(write_data);

            write_data.sType           = VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET;
            write_data.descriptorCount = 1;
            write_data.descriptorType  = VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER;
            write_data.pImageInfo      = &cubemap_image;
            write_data.dstBinding      = 0;
            write_data.dstSet          = m_gpu_resources->skybox_ds->handle();

            vkUpdateDescriptorSets(m_vk_backend->device(), 1, &write_data, 0, nullptr);
        }

        // Deferred Read
        {
            VkDescriptorImageInfo image;

            image.sampler     = m_vk_backend->bilinear_sampler()->handle();
            image.imageView   = m_gpu_resources->deferred_view->handle();
            image.imageLayout = VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL;

            VkWriteDescriptorSet write_data;
            DW_ZERO_MEMORY(write_data);

            write_data.sType           = VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET;
            write_data.descriptorCount = 1;
            write_data.descriptorType  = VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER;
            write_data.pImageInfo      = &image;
            write_data.dstBinding      = 0;
            write_data.dstSet          = m_gpu_resources->deferred_read_ds->handle();

            vkUpdateDescriptorSets(m_vk_backend->device(), 1, &write_data, 0, nullptr);
        }

        // TAA read
        {
            std::vector<VkDescriptorImageInfo> image_infos;
            std::vector<VkWriteDescriptorSet>  write_datas;
            VkWriteDescriptorSet               write_data;

            image_infos.reserve(2);
            write_datas.reserve(2);

            for (int i = 0; i < 2; i++)
            {
                VkDescriptorImageInfo combined_sampler_image_info;

                combined_sampler_image_info.sampler     = m_vk_backend->bilinear_sampler()->handle();
                combined_sampler_image_info.imageView   = m_gpu_resources->taa_view[i]->handle();
                combined_sampler_image_info.imageLayout = VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL;

                image_infos.push_back(combined_sampler_image_info);

                DW_ZERO_MEMORY(write_data);

                write_data.sType           = VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET;
                write_data.descriptorCount = 1;
                write_data.descriptorType  = VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER;
                write_data.pImageInfo      = &image_infos.back();
                write_data.dstBinding      = 0;
                write_data.dstSet          = m_gpu_resources->taa_read_ds[i]->handle();

                write_datas.push_back(write_data);
            }

            vkUpdateDescriptorSets(m_vk_backend->device(), write_datas.size(), write_datas.data(), 0, nullptr);
        }

        // TAA write
        {
            std::vector<VkDescriptorImageInfo> image_infos;
            std::vector<VkWriteDescriptorSet>  write_datas;
            VkWriteDescriptorSet               write_data;

            image_infos.reserve(2);
            write_datas.reserve(2);

            for (int i = 0; i < 2; i++)
            {
                VkDescriptorImageInfo storage_image_info;

                storage_image_info.sampler     = VK_NULL_HANDLE;
                storage_image_info.imageView   = m_gpu_resources->taa_view[i]->handle();
                storage_image_info.imageLayout = VK_IMAGE_LAYOUT_GENERAL;

                image_infos.push_back(storage_image_info);

                DW_ZERO_MEMORY(write_data);

                write_data.sType           = VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET;
                write_data.descriptorCount = 1;
                write_data.descriptorType  = VK_DESCRIPTOR_TYPE_STORAGE_IMAGE;
                write_data.pImageInfo      = &image_infos.back();
                write_data.dstBinding      = 0;
                write_data.dstSet          = m_gpu_resources->taa_write_ds[i]->handle();

                write_datas.push_back(write_data);
            }

            vkUpdateDescriptorSets(m_vk_backend->device(), write_datas.size(), write_datas.data(), 0, nullptr);
        }

        // Reflection RT write
        {
            VkDescriptorImageInfo storage_image_info;

            storage_image_info.sampler     = VK_NULL_HANDLE;
            storage_image_info.imageView   = m_gpu_resources->reflection_rt_color_view->handle();
            storage_image_info.imageLayout = VK_IMAGE_LAYOUT_GENERAL;

            VkWriteDescriptorSet write_data;

            DW_ZERO_MEMORY(write_data);

            write_data.sType           = VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET;
            write_data.descriptorCount = 1;
            write_data.descriptorType  = VK_DESCRIPTOR_TYPE_STORAGE_IMAGE;
            write_data.pImageInfo      = &storage_image_info;
            write_data.dstBinding      = 0;
            write_data.dstSet          = m_gpu_resources->reflection_rt_write_ds->handle();

            vkUpdateDescriptorSets(m_vk_backend->device(), 1, &write_data, 0, nullptr);
        }

        // Reflection RT read
        {
            VkDescriptorImageInfo sampler_image_info;

            sampler_image_info.sampler     = m_vk_backend->bilinear_sampler()->handle();
            sampler_image_info.imageView   = m_gpu_resources->reflection_rt_color_view->handle();
            sampler_image_info.imageLayout = VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL;

            VkWriteDescriptorSet write_data;

            DW_ZERO_MEMORY(write_data);

            write_data.sType           = VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET;
            write_data.descriptorCount = 1;
            write_data.descriptorType  = VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER;
            write_data.pImageInfo      = &sampler_image_info;
            write_data.dstBinding      = 0;
            write_data.dstSet          = m_gpu_resources->reflection_rt_read_ds->handle();

            vkUpdateDescriptorSets(m_vk_backend->device(), 1, &write_data, 0, nullptr);
        }

        // RTGI write
        {
            VkDescriptorImageInfo storage_image_info;

            storage_image_info.sampler     = VK_NULL_HANDLE;
            storage_image_info.imageView   = m_gpu_resources->rtgi_view->handle();
            storage_image_info.imageLayout = VK_IMAGE_LAYOUT_GENERAL;

            VkWriteDescriptorSet write_data;

            DW_ZERO_MEMORY(write_data);

            write_data.sType           = VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET;
            write_data.descriptorCount = 1;
            write_data.descriptorType  = VK_DESCRIPTOR_TYPE_STORAGE_IMAGE;
            write_data.pImageInfo      = &storage_image_info;
            write_data.dstBinding      = 0;
            write_data.dstSet          = m_gpu_resources->rtgi_write_ds->handle();

            vkUpdateDescriptorSets(m_vk_backend->device(), 1, &write_data, 0, nullptr);
        }

        // RTGI read
        {
            VkDescriptorImageInfo sampler_image_info;

            sampler_image_info.sampler     = m_vk_backend->bilinear_sampler()->handle();
            sampler_image_info.imageView   = m_gpu_resources->rtgi_view->handle();
            sampler_image_info.imageLayout = VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL;

            VkWriteDescriptorSet write_data;

            DW_ZERO_MEMORY(write_data);

            write_data.sType           = VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET;
            write_data.descriptorCount = 1;
            write_data.descriptorType  = VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER;
            write_data.pImageInfo      = &sampler_image_info;
            write_data.dstBinding      = 0;
            write_data.dstSet          = m_gpu_resources->rtgi_read_ds->handle();

            vkUpdateDescriptorSets(m_vk_backend->device(), 1, &write_data, 0, nullptr);
        }

        // Reflection Resolve write
        {
            VkDescriptorImageInfo storage_image_info;

            storage_image_info.sampler     = VK_NULL_HANDLE;
            storage_image_info.imageView   = m_gpu_resources->reflection_resolve_view->handle();
            storage_image_info.imageLayout = VK_IMAGE_LAYOUT_GENERAL;

            VkWriteDescriptorSet write_data;
            DW_ZERO_MEMORY(write_data);

            write_data.sType           = VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET;
            write_data.descriptorCount = 1;
            write_data.descriptorType  = VK_DESCRIPTOR_TYPE_STORAGE_IMAGE;
            write_data.pImageInfo      = &storage_image_info;
            write_data.dstBinding      = 0;
            write_data.dstSet          = m_gpu_resources->reflection_resolve_write_ds->handle();

            vkUpdateDescriptorSets(m_vk_backend->device(), 1, &write_data, 0, nullptr);
        }

        // Reflection Resolve read
        {
            VkDescriptorImageInfo sampler_image_info;

            sampler_image_info.sampler     = m_vk_backend->bilinear_sampler()->handle();
            sampler_image_info.imageView   = m_gpu_resources->reflection_resolve_view->handle();
            sampler_image_info.imageLayout = VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL;

            VkWriteDescriptorSet write_data;
            DW_ZERO_MEMORY(write_data);

            write_data.sType           = VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET;
            write_data.descriptorCount = 1;
            write_data.descriptorType  = VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER;
            write_data.pImageInfo      = &sampler_image_info;
            write_data.dstBinding      = 0;
            write_data.dstSet          = m_gpu_resources->reflection_resolve_read_ds->handle();

            vkUpdateDescriptorSets(m_vk_backend->device(), 1, &write_data, 0, nullptr);
        }

        // Reflection Temporal write
        {
            VkDescriptorImageInfo storage_image_info;

            storage_image_info.sampler     = VK_NULL_HANDLE;
            storage_image_info.imageView   = m_gpu_resources->reflection_temporal_view->handle();
            storage_image_info.imageLayout = VK_IMAGE_LAYOUT_GENERAL;

            VkWriteDescriptorSet write_data;
            DW_ZERO_MEMORY(write_data);

            write_data.sType           = VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET;
            write_data.descriptorCount = 1;
            write_data.descriptorType  = VK_DESCRIPTOR_TYPE_STORAGE_IMAGE;
            write_data.pImageInfo      = &storage_image_info;
            write_data.dstBinding      = 0;
            write_data.dstSet          = m_gpu_resources->reflection_temporal_write_ds->handle();

            vkUpdateDescriptorSets(m_vk_backend->device(), 1, &write_data, 0, nullptr);
        }

        // Reflection Temporal read
        {
            VkDescriptorImageInfo sampler_image_info;

            sampler_image_info.sampler     = m_vk_backend->bilinear_sampler()->handle();
            sampler_image_info.imageView   = m_gpu_resources->reflection_temporal_view->handle();
            sampler_image_info.imageLayout = VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL;

            VkWriteDescriptorSet write_data;
            DW_ZERO_MEMORY(write_data);

            write_data.sType           = VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET;
            write_data.descriptorCount = 1;
            write_data.descriptorType  = VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER;
            write_data.pImageInfo      = &sampler_image_info;
            write_data.dstBinding      = 0;
            write_data.dstSet          = m_gpu_resources->reflection_temporal_read_ds->handle();

            vkUpdateDescriptorSets(m_vk_backend->device(), 1, &write_data, 0, nullptr);
        }

        // Reflection History write
        {
            std::vector<VkDescriptorImageInfo> image_infos;
            std::vector<VkWriteDescriptorSet>  write_datas;
            VkWriteDescriptorSet               write_data;

            image_infos.reserve(2);
            write_datas.reserve(2);

            for (int i = 0; i < 2; i++)
            {
                VkDescriptorImageInfo storage_image_info;

                storage_image_info.sampler     = VK_NULL_HANDLE;
                storage_image_info.imageView   = m_gpu_resources->reflection_history_view[i]->handle();
                storage_image_info.imageLayout = VK_IMAGE_LAYOUT_GENERAL;

                image_infos.push_back(storage_image_info);

                DW_ZERO_MEMORY(write_data);

                write_data.sType           = VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET;
                write_data.descriptorCount = 1;
                write_data.descriptorType  = VK_DESCRIPTOR_TYPE_STORAGE_IMAGE;
                write_data.pImageInfo      = &image_infos.back();
                write_data.dstBinding      = 0;
                write_data.dstSet          = m_gpu_resources->reflection_history_ds[i]->handle();

                write_datas.push_back(write_data);
            }

            vkUpdateDescriptorSets(m_vk_backend->device(), write_datas.size(), write_datas.data(), 0, nullptr);
        }

        // Reflection Blur write
        {
            std::vector<VkDescriptorImageInfo> image_infos;
            std::vector<VkWriteDescriptorSet>  write_datas;
            VkWriteDescriptorSet               write_data;

            image_infos.reserve(2);
            write_datas.reserve(2);

            for (int i = 0; i < 2; i++)
            {
                VkDescriptorImageInfo storage_image_info;

                storage_image_info.sampler     = VK_NULL_HANDLE;
                storage_image_info.imageView   = m_gpu_resources->reflection_blur_view[i]->handle();
                storage_image_info.imageLayout = VK_IMAGE_LAYOUT_GENERAL;

                image_infos.push_back(storage_image_info);

                DW_ZERO_MEMORY(write_data);

                write_data.sType           = VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET;
                write_data.descriptorCount = 1;
                write_data.descriptorType  = VK_DESCRIPTOR_TYPE_STORAGE_IMAGE;
                write_data.pImageInfo      = &image_infos.back();
                write_data.dstBinding      = 0;
                write_data.dstSet          = m_gpu_resources->reflection_blur_write_ds[i]->handle();

                write_datas.push_back(write_data);
            }

            vkUpdateDescriptorSets(m_vk_backend->device(), write_datas.size(), write_datas.data(), 0, nullptr);
        }

        // Reflection Blur read
        {
            std::vector<VkDescriptorImageInfo> image_infos;
            std::vector<VkWriteDescriptorSet>  write_datas;
            VkWriteDescriptorSet               write_data;

            image_infos.reserve(2);
            write_datas.reserve(2);

            for (int i = 0; i < 2; i++)
            {
                VkDescriptorImageInfo sampler_image_info;

                sampler_image_info.sampler     = m_vk_backend->nearest_sampler()->handle();
                sampler_image_info.imageView   = m_gpu_resources->reflection_blur_view[i]->handle();
                sampler_image_info.imageLayout = VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL;

                image_infos.push_back(sampler_image_info);

                DW_ZERO_MEMORY(write_data);

                write_data.sType           = VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET;
                write_data.descriptorCount = 1;
                write_data.descriptorType  = VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER;
                write_data.pImageInfo      = &image_infos.back();
                write_data.dstBinding      = 0;
                write_data.dstSet          = m_gpu_resources->reflection_blur_read_ds[i]->handle();

                write_datas.push_back(write_data);
            }

            vkUpdateDescriptorSets(m_vk_backend->device(), write_datas.size(), write_datas.data(), 0, nullptr);
        }
    }

    // -----------------------------------------------------------------------------------------------------------------------------------

    void create_deferred_pipeline()
    {
        dw::vk::PipelineLayout::Desc desc;

        desc.add_descriptor_set_layout(m_gpu_resources->g_buffer_ds_layout);
        desc.add_descriptor_set_layout(m_gpu_resources->combined_sampler_ds_layout);
        desc.add_descriptor_set_layout(m_gpu_resources->combined_sampler_ds_layout);
        desc.add_descriptor_set_layout(m_gpu_resources->combined_sampler_ds_layout);
        desc.add_descriptor_set_layout(m_gpu_resources->per_frame_ds_layout);
        desc.add_descriptor_set_layout(m_gpu_resources->pbr_ds_layout);
        desc.add_push_constant_range(VK_SHADER_STAGE_FRAGMENT_BIT, 0, sizeof(DeferredShadingPushConstants));

        m_gpu_resources->deferred_pipeline_layout = dw::vk::PipelineLayout::create(m_vk_backend, desc);
        m_gpu_resources->deferred_pipeline        = dw::vk::GraphicsPipeline::create_for_post_process(m_vk_backend, "shaders/triangle.vert.spv", "shaders/deferred.frag.spv", m_gpu_resources->deferred_pipeline_layout, m_gpu_resources->deferred_rp);
    }

    // -----------------------------------------------------------------------------------------------------------------------------------

    void create_tone_map_pipeline()
    {
        dw::vk::PipelineLayout::Desc desc;

        desc.add_push_constant_range(VK_SHADER_STAGE_FRAGMENT_BIT, 0, sizeof(ToneMapPushConstants));
        desc.add_descriptor_set_layout(m_gpu_resources->combined_sampler_ds_layout);
        desc.add_descriptor_set_layout(m_gpu_resources->combined_sampler_ds_layout);
        desc.add_descriptor_set_layout(m_gpu_resources->combined_sampler_ds_layout);
        desc.add_descriptor_set_layout(m_gpu_resources->combined_sampler_ds_layout);

        m_gpu_resources->copy_pipeline_layout = dw::vk::PipelineLayout::create(m_vk_backend, desc);
        m_gpu_resources->copy_pipeline        = dw::vk::GraphicsPipeline::create_for_post_process(m_vk_backend, "shaders/triangle.vert.spv", "shaders/tone_map.frag.spv", m_gpu_resources->copy_pipeline_layout, m_vk_backend->swapchain_render_pass());
    }

    // -----------------------------------------------------------------------------------------------------------------------------------

    void create_shadow_mask_ray_tracing_pipeline()
    {
        // ---------------------------------------------------------------------------
        // Create shader modules
        // ---------------------------------------------------------------------------

        dw::vk::ShaderModule::Ptr rgen  = dw::vk::ShaderModule::create_from_file(m_vk_backend, "shaders/shadow.rgen.spv");
        dw::vk::ShaderModule::Ptr rchit = dw::vk::ShaderModule::create_from_file(m_vk_backend, "shaders/shadow.rchit.spv");
        dw::vk::ShaderModule::Ptr rmiss = dw::vk::ShaderModule::create_from_file(m_vk_backend, "shaders/shadow.rmiss.spv");

        dw::vk::ShaderBindingTable::Desc sbt_desc;

        sbt_desc.add_ray_gen_group(rgen, "main");
        sbt_desc.add_hit_group(rchit, "main");
        sbt_desc.add_miss_group(rmiss, "main");

        m_gpu_resources->shadow_mask_sbt = dw::vk::ShaderBindingTable::create(m_vk_backend, sbt_desc);

        dw::vk::RayTracingPipeline::Desc desc;

        desc.set_max_pipeline_ray_recursion_depth(1);
        desc.set_shader_binding_table(m_gpu_resources->shadow_mask_sbt);

        // ---------------------------------------------------------------------------
        // Create pipeline layout
        // ---------------------------------------------------------------------------

        dw::vk::PipelineLayout::Desc pl_desc;

        pl_desc.add_descriptor_set_layout(m_gpu_resources->pillars_scene->descriptor_set_layout());
        pl_desc.add_descriptor_set_layout(m_gpu_resources->storage_image_ds_layout);
        pl_desc.add_descriptor_set_layout(m_gpu_resources->per_frame_ds_layout);
        pl_desc.add_descriptor_set_layout(m_gpu_resources->g_buffer_ds_layout);

        pl_desc.add_push_constant_range(VK_SHADER_STAGE_RAYGEN_BIT_KHR, 0, sizeof(ShadowPushConstants));

        m_gpu_resources->shadow_mask_pipeline_layout = dw::vk::PipelineLayout::create(m_vk_backend, pl_desc);

        desc.set_pipeline_layout(m_gpu_resources->shadow_mask_pipeline_layout);

        m_gpu_resources->shadow_mask_pipeline = dw::vk::RayTracingPipeline::create(m_vk_backend, desc);
    }

    // -----------------------------------------------------------------------------------------------------------------------------------

    void create_ambient_occlusion_ray_tracing_pipeline()
    {
        // ---------------------------------------------------------------------------
        // Create shader modules
        // ---------------------------------------------------------------------------

        dw::vk::ShaderModule::Ptr rgen  = dw::vk::ShaderModule::create_from_file(m_vk_backend, "shaders/ambient_occlusion.rgen.spv");
        dw::vk::ShaderModule::Ptr rchit = dw::vk::ShaderModule::create_from_file(m_vk_backend, "shaders/shadow.rchit.spv");
        dw::vk::ShaderModule::Ptr rmiss = dw::vk::ShaderModule::create_from_file(m_vk_backend, "shaders/shadow.rmiss.spv");

        dw::vk::ShaderBindingTable::Desc sbt_desc;

        sbt_desc.add_ray_gen_group(rgen, "main");
        sbt_desc.add_hit_group(rchit, "main");
        sbt_desc.add_miss_group(rmiss, "main");

        m_gpu_resources->rtao_sbt = dw::vk::ShaderBindingTable::create(m_vk_backend, sbt_desc);

        dw::vk::RayTracingPipeline::Desc desc;

        desc.set_max_pipeline_ray_recursion_depth(1);
        desc.set_shader_binding_table(m_gpu_resources->rtao_sbt);

        // ---------------------------------------------------------------------------
        // Create pipeline layout
        // ---------------------------------------------------------------------------

        dw::vk::PipelineLayout::Desc pl_desc;

        pl_desc.add_descriptor_set_layout(m_gpu_resources->pillars_scene->descriptor_set_layout());
        pl_desc.add_descriptor_set_layout(m_gpu_resources->storage_image_ds_layout);
        pl_desc.add_descriptor_set_layout(m_gpu_resources->per_frame_ds_layout);
        pl_desc.add_descriptor_set_layout(m_gpu_resources->g_buffer_ds_layout);

        pl_desc.add_push_constant_range(VK_SHADER_STAGE_RAYGEN_BIT_KHR, 0, sizeof(AmbientOcclusionPushConstants));

        m_gpu_resources->rtao_pipeline_layout = dw::vk::PipelineLayout::create(m_vk_backend, pl_desc);

        desc.set_pipeline_layout(m_gpu_resources->rtao_pipeline_layout);

        m_gpu_resources->rtao_pipeline = dw::vk::RayTracingPipeline::create(m_vk_backend, desc);
    }

    // -----------------------------------------------------------------------------------------------------------------------------------

    void create_reflection_ray_tracing_pipeline()
    {
        // ---------------------------------------------------------------------------
        // Create shader modules
        // ---------------------------------------------------------------------------

        dw::vk::ShaderModule::Ptr rgen             = dw::vk::ShaderModule::create_from_file(m_vk_backend, "shaders/reflection.rgen.spv");
        dw::vk::ShaderModule::Ptr rchit            = dw::vk::ShaderModule::create_from_file(m_vk_backend, "shaders/reflection.rchit.spv");
        dw::vk::ShaderModule::Ptr rmiss            = dw::vk::ShaderModule::create_from_file(m_vk_backend, "shaders/reflection.rmiss.spv");
        dw::vk::ShaderModule::Ptr rchit_visibility = dw::vk::ShaderModule::create_from_file(m_vk_backend, "shaders/shadow.rchit.spv");
        dw::vk::ShaderModule::Ptr rmiss_visibility = dw::vk::ShaderModule::create_from_file(m_vk_backend, "shaders/shadow.rmiss.spv");

        dw::vk::ShaderBindingTable::Desc sbt_desc;

        sbt_desc.add_ray_gen_group(rgen, "main");
        sbt_desc.add_hit_group(rchit, "main");
        sbt_desc.add_hit_group(rchit_visibility, "main");
        sbt_desc.add_miss_group(rmiss, "main");
        sbt_desc.add_miss_group(rmiss_visibility, "main");

        m_gpu_resources->reflection_rt_sbt = dw::vk::ShaderBindingTable::create(m_vk_backend, sbt_desc);

        dw::vk::RayTracingPipeline::Desc desc;

        desc.set_max_pipeline_ray_recursion_depth(1);
        desc.set_shader_binding_table(m_gpu_resources->reflection_rt_sbt);

        // ---------------------------------------------------------------------------
        // Create pipeline layout
        // ---------------------------------------------------------------------------

        dw::vk::PipelineLayout::Desc pl_desc;

        pl_desc.add_descriptor_set_layout(m_gpu_resources->pillars_scene->descriptor_set_layout());
        pl_desc.add_descriptor_set_layout(m_gpu_resources->storage_image_ds_layout);
        pl_desc.add_descriptor_set_layout(m_gpu_resources->per_frame_ds_layout);
        pl_desc.add_descriptor_set_layout(m_gpu_resources->g_buffer_ds_layout);
        pl_desc.add_descriptor_set_layout(m_gpu_resources->pbr_ds_layout);
        pl_desc.add_descriptor_set_layout(m_gpu_resources->combined_sampler_ds_layout);
        pl_desc.add_push_constant_range(VK_SHADER_STAGE_RAYGEN_BIT_KHR, 0, sizeof(ReflectionsPushConstants));

        m_gpu_resources->reflection_rt_pipeline_layout = dw::vk::PipelineLayout::create(m_vk_backend, pl_desc);

        desc.set_pipeline_layout(m_gpu_resources->reflection_rt_pipeline_layout);

        m_gpu_resources->reflection_rt_pipeline = dw::vk::RayTracingPipeline::create(m_vk_backend, desc);
    }

    // -----------------------------------------------------------------------------------------------------------------------------------

    void create_gi_ray_tracing_pipeline()
    {
        // ---------------------------------------------------------------------------
        // Create shader modules
        // ---------------------------------------------------------------------------

        dw::vk::ShaderModule::Ptr rgen             = dw::vk::ShaderModule::create_from_file(m_vk_backend, "shaders/indirect_diffuse.rgen.spv");
        dw::vk::ShaderModule::Ptr rchit            = dw::vk::ShaderModule::create_from_file(m_vk_backend, "shaders/indirect_diffuse.rchit.spv");
        dw::vk::ShaderModule::Ptr rmiss            = dw::vk::ShaderModule::create_from_file(m_vk_backend, "shaders/indirect_diffuse.rmiss.spv");
        dw::vk::ShaderModule::Ptr rchit_visibility = dw::vk::ShaderModule::create_from_file(m_vk_backend, "shaders/shadow.rchit.spv");
        dw::vk::ShaderModule::Ptr rmiss_visibility = dw::vk::ShaderModule::create_from_file(m_vk_backend, "shaders/shadow.rmiss.spv");

        dw::vk::ShaderBindingTable::Desc sbt_desc;

        sbt_desc.add_ray_gen_group(rgen, "main");
        sbt_desc.add_hit_group(rchit, "main");
        sbt_desc.add_hit_group(rchit_visibility, "main");
        sbt_desc.add_miss_group(rmiss, "main");
        sbt_desc.add_miss_group(rmiss_visibility, "main");

        m_gpu_resources->rtgi_sbt = dw::vk::ShaderBindingTable::create(m_vk_backend, sbt_desc);

        dw::vk::RayTracingPipeline::Desc desc;

        desc.set_max_pipeline_ray_recursion_depth(2);
        desc.set_shader_binding_table(m_gpu_resources->rtgi_sbt);

        // ---------------------------------------------------------------------------
        // Create pipeline layout
        // ---------------------------------------------------------------------------

        dw::vk::PipelineLayout::Desc pl_desc;

        pl_desc.add_descriptor_set_layout(m_gpu_resources->pillars_scene->descriptor_set_layout());
        pl_desc.add_descriptor_set_layout(m_gpu_resources->storage_image_ds_layout);
        pl_desc.add_descriptor_set_layout(m_gpu_resources->per_frame_ds_layout);
        pl_desc.add_descriptor_set_layout(m_gpu_resources->g_buffer_ds_layout);
        pl_desc.add_descriptor_set_layout(m_gpu_resources->pbr_ds_layout);
        pl_desc.add_descriptor_set_layout(m_gpu_resources->combined_sampler_ds_layout);
        pl_desc.add_push_constant_range(VK_SHADER_STAGE_RAYGEN_BIT_KHR | VK_SHADER_STAGE_CLOSEST_HIT_BIT_KHR, 0, sizeof(GIPushConstants));

        m_gpu_resources->rtgi_pipeline_layout = dw::vk::PipelineLayout::create(m_vk_backend, pl_desc);

        desc.set_pipeline_layout(m_gpu_resources->rtgi_pipeline_layout);

        m_gpu_resources->rtgi_pipeline = dw::vk::RayTracingPipeline::create(m_vk_backend, desc);
    }

    // -----------------------------------------------------------------------------------------------------------------------------------

    void create_reflection_resolve_pipeline()
    {
        dw::vk::PipelineLayout::Desc desc;

        desc.add_descriptor_set_layout(m_gpu_resources->storage_image_ds_layout);
        desc.add_descriptor_set_layout(m_gpu_resources->combined_sampler_ds_layout);
        desc.add_descriptor_set_layout(m_gpu_resources->g_buffer_ds_layout);
        desc.add_descriptor_set_layout(m_gpu_resources->per_frame_ds_layout);
        desc.add_push_constant_range(VK_SHADER_STAGE_COMPUTE_BIT, 0, sizeof(ReflectionsSpatialResolvePushConstants));

        m_gpu_resources->reflection_resolve_pipeline_layout = dw::vk::PipelineLayout::create(m_vk_backend, desc);

        dw::vk::ShaderModule::Ptr module = dw::vk::ShaderModule::create_from_file(m_vk_backend, "shaders/reflection_spatial_resolve.comp.spv");

        dw::vk::ComputePipeline::Desc comp_desc;

        comp_desc.set_pipeline_layout(m_gpu_resources->reflection_resolve_pipeline_layout);
        comp_desc.set_shader_stage(module, "main");

        m_gpu_resources->reflection_resolve_pipeline = dw::vk::ComputePipeline::create(m_vk_backend, comp_desc);
    }

    // -----------------------------------------------------------------------------------------------------------------------------------

    void create_reflection_temporal_pipeline()
    {
        dw::vk::PipelineLayout::Desc desc;

        desc.add_descriptor_set_layout(m_gpu_resources->storage_image_ds_layout);
        desc.add_descriptor_set_layout(m_gpu_resources->combined_sampler_ds_layout);
        desc.add_descriptor_set_layout(m_gpu_resources->combined_sampler_ds_layout);
        desc.add_descriptor_set_layout(m_gpu_resources->g_buffer_ds_layout);
        desc.add_descriptor_set_layout(m_gpu_resources->per_frame_ds_layout);
        desc.add_push_constant_range(VK_SHADER_STAGE_COMPUTE_BIT, 0, sizeof(ReflectionsTemporalPushConstants));

        m_gpu_resources->reflection_temporal_pipeline_layout = dw::vk::PipelineLayout::create(m_vk_backend, desc);

        dw::vk::ShaderModule::Ptr module = dw::vk::ShaderModule::create_from_file(m_vk_backend, "shaders/reflection_reprojection.comp.spv");

        dw::vk::ComputePipeline::Desc comp_desc;

        comp_desc.set_pipeline_layout(m_gpu_resources->reflection_temporal_pipeline_layout);
        comp_desc.set_shader_stage(module, "main");

        m_gpu_resources->reflection_temporal_pipeline = dw::vk::ComputePipeline::create(m_vk_backend, comp_desc);
    }

    // -----------------------------------------------------------------------------------------------------------------------------------

    void create_reflection_blur_pipeline()
    {
        dw::vk::PipelineLayout::Desc desc;

        desc.add_descriptor_set_layout(m_gpu_resources->storage_image_ds_layout);
        desc.add_descriptor_set_layout(m_gpu_resources->storage_image_ds_layout);
        desc.add_descriptor_set_layout(m_gpu_resources->g_buffer_ds_layout);
        desc.add_descriptor_set_layout(m_gpu_resources->g_buffer_ds_layout);
        desc.add_descriptor_set_layout(m_gpu_resources->combined_sampler_ds_layout);
        desc.add_descriptor_set_layout(m_gpu_resources->combined_sampler_ds_layout);
        desc.add_descriptor_set_layout(m_gpu_resources->per_frame_ds_layout);
        desc.add_push_constant_range(VK_SHADER_STAGE_COMPUTE_BIT, 0, sizeof(ReflectionsBlurPushConstants));

        m_gpu_resources->reflection_blur_pipeline_layout = dw::vk::PipelineLayout::create(m_vk_backend, desc);

        dw::vk::ShaderModule::Ptr module = dw::vk::ShaderModule::create_from_file(m_vk_backend, "shaders/reflection_blur.comp.spv");

        dw::vk::ComputePipeline::Desc comp_desc;

        comp_desc.set_pipeline_layout(m_gpu_resources->reflection_blur_pipeline_layout);
        comp_desc.set_shader_stage(module, "main");

        m_gpu_resources->reflection_blur_pipeline = dw::vk::ComputePipeline::create(m_vk_backend, comp_desc);
    }

    // -----------------------------------------------------------------------------------------------------------------------------------

    void create_gbuffer_pipeline()
    {
        // ---------------------------------------------------------------------------
        // Create shader modules
        // ---------------------------------------------------------------------------

        dw::vk::ShaderModule::Ptr vs = dw::vk::ShaderModule::create_from_file(m_vk_backend, "shaders/g_buffer.vert.spv");
        dw::vk::ShaderModule::Ptr fs = dw::vk::ShaderModule::create_from_file(m_vk_backend, "shaders/g_buffer.frag.spv");

        dw::vk::GraphicsPipeline::Desc pso_desc;

        pso_desc.add_shader_stage(VK_SHADER_STAGE_VERTEX_BIT, vs, "main")
            .add_shader_stage(VK_SHADER_STAGE_FRAGMENT_BIT, fs, "main");

        // ---------------------------------------------------------------------------
        // Create vertex input state
        // ---------------------------------------------------------------------------

        pso_desc.set_vertex_input_state(m_gpu_resources->meshes[0]->vertex_input_state_desc());

        // ---------------------------------------------------------------------------
        // Create pipeline input assembly state
        // ---------------------------------------------------------------------------

        dw::vk::InputAssemblyStateDesc input_assembly_state_desc;

        input_assembly_state_desc.set_primitive_restart_enable(false)
            .set_topology(VK_PRIMITIVE_TOPOLOGY_TRIANGLE_LIST);

        pso_desc.set_input_assembly_state(input_assembly_state_desc);

        // ---------------------------------------------------------------------------
        // Create viewport state
        // ---------------------------------------------------------------------------

        dw::vk::ViewportStateDesc vp_desc;

        vp_desc.add_viewport(0.0f, 0.0f, m_width, m_height, 0.0f, 1.0f)
            .add_scissor(0, 0, m_width, m_height);

        pso_desc.set_viewport_state(vp_desc);

        // ---------------------------------------------------------------------------
        // Create rasterization state
        // ---------------------------------------------------------------------------

        dw::vk::RasterizationStateDesc rs_state;

        rs_state.set_depth_clamp(VK_FALSE)
            .set_rasterizer_discard_enable(VK_FALSE)
            .set_polygon_mode(VK_POLYGON_MODE_FILL)
            .set_line_width(1.0f)
            .set_cull_mode(VK_CULL_MODE_BACK_BIT)
            .set_front_face(VK_FRONT_FACE_CLOCKWISE)
            .set_depth_bias(VK_FALSE);

        pso_desc.set_rasterization_state(rs_state);

        // ---------------------------------------------------------------------------
        // Create multisample state
        // ---------------------------------------------------------------------------

        dw::vk::MultisampleStateDesc ms_state;

        ms_state.set_sample_shading_enable(VK_FALSE)
            .set_rasterization_samples(VK_SAMPLE_COUNT_1_BIT);

        pso_desc.set_multisample_state(ms_state);

        // ---------------------------------------------------------------------------
        // Create depth stencil state
        // ---------------------------------------------------------------------------

        dw::vk::DepthStencilStateDesc ds_state;

        ds_state.set_depth_test_enable(VK_TRUE)
            .set_depth_write_enable(VK_TRUE)
            .set_depth_compare_op(VK_COMPARE_OP_LESS)
            .set_depth_bounds_test_enable(VK_FALSE)
            .set_stencil_test_enable(VK_FALSE);

        pso_desc.set_depth_stencil_state(ds_state);

        // ---------------------------------------------------------------------------
        // Create color blend state
        // ---------------------------------------------------------------------------

        dw::vk::ColorBlendAttachmentStateDesc blend_att_desc;

        blend_att_desc.set_color_write_mask(VK_COLOR_COMPONENT_R_BIT | VK_COLOR_COMPONENT_G_BIT | VK_COLOR_COMPONENT_B_BIT | VK_COLOR_COMPONENT_A_BIT)
            .set_blend_enable(VK_FALSE);

        dw::vk::ColorBlendStateDesc blend_state;

        blend_state.set_logic_op_enable(VK_FALSE)
            .set_logic_op(VK_LOGIC_OP_COPY)
            .set_blend_constants(0.0f, 0.0f, 0.0f, 0.0f)
            .add_attachment(blend_att_desc)
            .add_attachment(blend_att_desc)
            .add_attachment(blend_att_desc)
            .add_attachment(blend_att_desc);

        pso_desc.set_color_blend_state(blend_state);

        // ---------------------------------------------------------------------------
        // Create pipeline layout
        // ---------------------------------------------------------------------------

        dw::vk::PipelineLayout::Desc pl_desc;

        pl_desc.add_descriptor_set_layout(m_gpu_resources->pillars_scene->descriptor_set_layout())
            .add_descriptor_set_layout(m_gpu_resources->per_frame_ds_layout)
            .add_push_constant_range(VK_SHADER_STAGE_VERTEX_BIT | VK_SHADER_STAGE_FRAGMENT_BIT, 0, sizeof(GBufferPushConstants));

        m_gpu_resources->g_buffer_pipeline_layout = dw::vk::PipelineLayout::create(m_vk_backend, pl_desc);

        pso_desc.set_pipeline_layout(m_gpu_resources->g_buffer_pipeline_layout);

        // ---------------------------------------------------------------------------
        // Create dynamic state
        // ---------------------------------------------------------------------------

        pso_desc.add_dynamic_state(VK_DYNAMIC_STATE_VIEWPORT)
            .add_dynamic_state(VK_DYNAMIC_STATE_SCISSOR);

        // ---------------------------------------------------------------------------
        // Create pipeline
        // ---------------------------------------------------------------------------

        pso_desc.set_render_pass(m_gpu_resources->g_buffer_rp);

        m_gpu_resources->g_buffer_pipeline = dw::vk::GraphicsPipeline::create(m_vk_backend, pso_desc);
    }

    // -----------------------------------------------------------------------------------------------------------------------------------

    void create_skybox_pipeline()
    {
        // ---------------------------------------------------------------------------
        // Create shader modules
        // ---------------------------------------------------------------------------

        dw::vk::ShaderModule::Ptr vs = dw::vk::ShaderModule::create_from_file(m_vk_backend, "shaders/skybox.vert.spv");
        dw::vk::ShaderModule::Ptr fs = dw::vk::ShaderModule::create_from_file(m_vk_backend, "shaders/skybox.frag.spv");

        dw::vk::GraphicsPipeline::Desc pso_desc;

        pso_desc.add_shader_stage(VK_SHADER_STAGE_VERTEX_BIT, vs, "main")
            .add_shader_stage(VK_SHADER_STAGE_FRAGMENT_BIT, fs, "main");

        // ---------------------------------------------------------------------------
        // Create vertex input state
        // ---------------------------------------------------------------------------

        dw::vk::VertexInputStateDesc vertex_input_state_desc;

        struct SkyboxVertex
        {
            glm::vec3 position;
            glm::vec3 normal;
            glm::vec2 texcoord;
        };

        vertex_input_state_desc.add_binding_desc(0, sizeof(SkyboxVertex), VK_VERTEX_INPUT_RATE_VERTEX);

        vertex_input_state_desc.add_attribute_desc(0, 0, VK_FORMAT_R32G32B32_SFLOAT, 0);
        vertex_input_state_desc.add_attribute_desc(1, 0, VK_FORMAT_R32G32B32_SFLOAT, offsetof(SkyboxVertex, normal));
        vertex_input_state_desc.add_attribute_desc(2, 0, VK_FORMAT_R32G32_SFLOAT, offsetof(SkyboxVertex, texcoord));

        pso_desc.set_vertex_input_state(vertex_input_state_desc);

        // ---------------------------------------------------------------------------
        // Create pipeline input assembly state
        // ---------------------------------------------------------------------------

        dw::vk::InputAssemblyStateDesc input_assembly_state_desc;

        input_assembly_state_desc.set_primitive_restart_enable(false)
            .set_topology(VK_PRIMITIVE_TOPOLOGY_TRIANGLE_LIST);

        pso_desc.set_input_assembly_state(input_assembly_state_desc);

        // ---------------------------------------------------------------------------
        // Create viewport state
        // ---------------------------------------------------------------------------

        dw::vk::ViewportStateDesc vp_desc;

        vp_desc.add_viewport(0.0f, 0.0f, m_width, m_height, 0.0f, 1.0f)
            .add_scissor(0, 0, m_width, m_height);

        pso_desc.set_viewport_state(vp_desc);

        // ---------------------------------------------------------------------------
        // Create rasterization state
        // ---------------------------------------------------------------------------

        dw::vk::RasterizationStateDesc rs_state;

        rs_state.set_depth_clamp(VK_FALSE)
            .set_rasterizer_discard_enable(VK_FALSE)
            .set_polygon_mode(VK_POLYGON_MODE_FILL)
            .set_line_width(1.0f)
            .set_cull_mode(VK_CULL_MODE_NONE)
            .set_front_face(VK_FRONT_FACE_COUNTER_CLOCKWISE)
            .set_depth_bias(VK_FALSE);

        pso_desc.set_rasterization_state(rs_state);

        // ---------------------------------------------------------------------------
        // Create multisample state
        // ---------------------------------------------------------------------------

        dw::vk::MultisampleStateDesc ms_state;

        ms_state.set_sample_shading_enable(VK_FALSE)
            .set_rasterization_samples(VK_SAMPLE_COUNT_1_BIT);

        pso_desc.set_multisample_state(ms_state);

        // ---------------------------------------------------------------------------
        // Create depth stencil state
        // ---------------------------------------------------------------------------

        dw::vk::DepthStencilStateDesc ds_state;

        ds_state.set_depth_test_enable(VK_TRUE)
            .set_depth_write_enable(VK_FALSE)
            .set_depth_compare_op(VK_COMPARE_OP_LESS_OR_EQUAL)
            .set_depth_bounds_test_enable(VK_FALSE)
            .set_stencil_test_enable(VK_FALSE);

        pso_desc.set_depth_stencil_state(ds_state);

        // ---------------------------------------------------------------------------
        // Create color blend state
        // ---------------------------------------------------------------------------

        dw::vk::ColorBlendAttachmentStateDesc blend_att_desc;

        blend_att_desc.set_color_write_mask(VK_COLOR_COMPONENT_R_BIT | VK_COLOR_COMPONENT_G_BIT | VK_COLOR_COMPONENT_B_BIT | VK_COLOR_COMPONENT_A_BIT)
            .set_blend_enable(VK_FALSE);

        dw::vk::ColorBlendStateDesc blend_state;

        blend_state.set_logic_op_enable(VK_FALSE)
            .set_logic_op(VK_LOGIC_OP_COPY)
            .set_blend_constants(0.0f, 0.0f, 0.0f, 0.0f)
            .add_attachment(blend_att_desc);

        pso_desc.set_color_blend_state(blend_state);

        // ---------------------------------------------------------------------------
        // Create pipeline layout
        // ---------------------------------------------------------------------------

        dw::vk::PipelineLayout::Desc pl_desc;

        pl_desc.add_descriptor_set_layout(m_gpu_resources->combined_sampler_ds_layout)
            .add_push_constant_range(VK_SHADER_STAGE_VERTEX_BIT, 0, sizeof(SkyboxPushConstants));

        m_gpu_resources->skybox_pipeline_layout = dw::vk::PipelineLayout::create(m_vk_backend, pl_desc);

        pso_desc.set_pipeline_layout(m_gpu_resources->skybox_pipeline_layout);

        // ---------------------------------------------------------------------------
        // Create dynamic state
        // ---------------------------------------------------------------------------

        pso_desc.add_dynamic_state(VK_DYNAMIC_STATE_VIEWPORT)
            .add_dynamic_state(VK_DYNAMIC_STATE_SCISSOR);

        // ---------------------------------------------------------------------------
        // Create pipeline
        // ---------------------------------------------------------------------------

        pso_desc.set_render_pass(m_gpu_resources->skybox_rp);

        m_gpu_resources->skybox_pipeline = dw::vk::GraphicsPipeline::create(m_vk_backend, pso_desc);
    }

    // -----------------------------------------------------------------------------------------------------------------------------------

    void create_taa_pipeline()
    {
        dw::vk::PipelineLayout::Desc desc;

        desc.add_descriptor_set_layout(m_gpu_resources->storage_image_ds_layout);
        desc.add_descriptor_set_layout(m_gpu_resources->combined_sampler_ds_layout);
        desc.add_descriptor_set_layout(m_gpu_resources->combined_sampler_ds_layout);
        desc.add_descriptor_set_layout(m_gpu_resources->g_buffer_ds_layout);
        desc.add_push_constant_range(VK_SHADER_STAGE_COMPUTE_BIT, 0, sizeof(TAAPushConstants));

        m_gpu_resources->taa_pipeline_layout = dw::vk::PipelineLayout::create(m_vk_backend, desc);

        dw::vk::ShaderModule::Ptr module = dw::vk::ShaderModule::create_from_file(m_vk_backend, "shaders/taa.comp.spv");

        dw::vk::ComputePipeline::Desc comp_desc;

        comp_desc.set_pipeline_layout(m_gpu_resources->taa_pipeline_layout);
        comp_desc.set_shader_stage(module, "main");

        m_gpu_resources->taa_pipeline = dw::vk::ComputePipeline::create(m_vk_backend, comp_desc);
    }

    // -----------------------------------------------------------------------------------------------------------------------------------

    void create_cube()
    {
        float cube_vertices[] = {
            // back face
            -1.0f,
            -1.0f,
            -1.0f,
            0.0f,
            0.0f,
            -1.0f,
            0.0f,
            0.0f, // bottom-left
            1.0f,
            1.0f,
            -1.0f,
            0.0f,
            0.0f,
            -1.0f,
            1.0f,
            1.0f, // top-right
            1.0f,
            -1.0f,
            -1.0f,
            0.0f,
            0.0f,
            -1.0f,
            1.0f,
            0.0f, // bottom-right
            1.0f,
            1.0f,
            -1.0f,
            0.0f,
            0.0f,
            -1.0f,
            1.0f,
            1.0f, // top-right
            -1.0f,
            -1.0f,
            -1.0f,
            0.0f,
            0.0f,
            -1.0f,
            0.0f,
            0.0f, // bottom-left
            -1.0f,
            1.0f,
            -1.0f,
            0.0f,
            0.0f,
            -1.0f,
            0.0f,
            1.0f, // top-left
            // front face
            -1.0f,
            -1.0f,
            1.0f,
            0.0f,
            0.0f,
            1.0f,
            0.0f,
            0.0f, // bottom-left
            1.0f,
            -1.0f,
            1.0f,
            0.0f,
            0.0f,
            1.0f,
            1.0f,
            0.0f, // bottom-right
            1.0f,
            1.0f,
            1.0f,
            0.0f,
            0.0f,
            1.0f,
            1.0f,
            1.0f, // top-right
            1.0f,
            1.0f,
            1.0f,
            0.0f,
            0.0f,
            1.0f,
            1.0f,
            1.0f, // top-right
            -1.0f,
            1.0f,
            1.0f,
            0.0f,
            0.0f,
            1.0f,
            0.0f,
            1.0f, // top-left
            -1.0f,
            -1.0f,
            1.0f,
            0.0f,
            0.0f,
            1.0f,
            0.0f,
            0.0f, // bottom-left
            // left face
            -1.0f,
            1.0f,
            1.0f,
            -1.0f,
            0.0f,
            0.0f,
            1.0f,
            0.0f, // top-right
            -1.0f,
            1.0f,
            -1.0f,
            -1.0f,
            0.0f,
            0.0f,
            1.0f,
            1.0f, // top-left
            -1.0f,
            -1.0f,
            -1.0f,
            -1.0f,
            0.0f,
            0.0f,
            0.0f,
            1.0f, // bottom-left
            -1.0f,
            -1.0f,
            -1.0f,
            -1.0f,
            0.0f,
            0.0f,
            0.0f,
            1.0f, // bottom-left
            -1.0f,
            -1.0f,
            1.0f,
            -1.0f,
            0.0f,
            0.0f,
            0.0f,
            0.0f, // bottom-right
            -1.0f,
            1.0f,
            1.0f,
            -1.0f,
            0.0f,
            0.0f,
            1.0f,
            0.0f, // top-right
            // right face
            1.0f,
            1.0f,
            1.0f,
            1.0f,
            0.0f,
            0.0f,
            1.0f,
            0.0f, // top-left
            1.0f,
            -1.0f,
            -1.0f,
            1.0f,
            0.0f,
            0.0f,
            0.0f,
            1.0f, // bottom-right
            1.0f,
            1.0f,
            -1.0f,
            1.0f,
            0.0f,
            0.0f,
            1.0f,
            1.0f, // top-right
            1.0f,
            -1.0f,
            -1.0f,
            1.0f,
            0.0f,
            0.0f,
            0.0f,
            1.0f, // bottom-right
            1.0f,
            1.0f,
            1.0f,
            1.0f,
            0.0f,
            0.0f,
            1.0f,
            0.0f, // top-left
            1.0f,
            -1.0f,
            1.0f,
            1.0f,
            0.0f,
            0.0f,
            0.0f,
            0.0f, // bottom-left
            // bottom face
            -1.0f,
            -1.0f,
            -1.0f,
            0.0f,
            -1.0f,
            0.0f,
            0.0f,
            1.0f, // top-right
            1.0f,
            -1.0f,
            -1.0f,
            0.0f,
            -1.0f,
            0.0f,
            1.0f,
            1.0f, // top-left
            1.0f,
            -1.0f,
            1.0f,
            0.0f,
            -1.0f,
            0.0f,
            1.0f,
            0.0f, // bottom-left
            1.0f,
            -1.0f,
            1.0f,
            0.0f,
            -1.0f,
            0.0f,
            1.0f,
            0.0f, // bottom-left
            -1.0f,
            -1.0f,
            1.0f,
            0.0f,
            -1.0f,
            0.0f,
            0.0f,
            0.0f, // bottom-right
            -1.0f,
            -1.0f,
            -1.0f,
            0.0f,
            -1.0f,
            0.0f,
            0.0f,
            1.0f, // top-right
            // top face
            -1.0f,
            1.0f,
            -1.0f,
            0.0f,
            1.0f,
            0.0f,
            0.0f,
            1.0f, // top-left
            1.0f,
            1.0f,
            1.0f,
            0.0f,
            1.0f,
            0.0f,
            1.0f,
            0.0f, // bottom-right
            1.0f,
            1.0f,
            -1.0f,
            0.0f,
            1.0f,
            0.0f,
            1.0f,
            1.0f, // top-right
            1.0f,
            1.0f,
            1.0f,
            0.0f,
            1.0f,
            0.0f,
            1.0f,
            0.0f, // bottom-right
            -1.0f,
            1.0f,
            -1.0f,
            0.0f,
            1.0f,
            0.0f,
            0.0f,
            1.0f, // top-left
            -1.0f,
            1.0f,
            1.0f,
            0.0f,
            1.0f,
            0.0f,
            0.0f,
            0.0f // bottom-left
        };
        m_gpu_resources->cube_vbo = dw::vk::Buffer::create(m_vk_backend, VK_BUFFER_USAGE_VERTEX_BUFFER_BIT | VK_BUFFER_USAGE_TRANSFER_DST_BIT, sizeof(cube_vertices), VMA_MEMORY_USAGE_GPU_ONLY, 0, cube_vertices);
    }

    // -----------------------------------------------------------------------------------------------------------------------------------

    bool load_mesh()
    {
        {
            std::vector<dw::RayTracedScene::Instance> instances;

            dw::Mesh::Ptr pillar = dw::Mesh::load(m_vk_backend, "mesh/pillar.gltf");

            if (!pillar)
            {
                DW_LOG_ERROR("Failed to load mesh");
                return false;
            }

            pillar->initialize_for_ray_tracing(m_vk_backend);

            m_gpu_resources->meshes.push_back(pillar);

            dw::Mesh::Ptr bunny = dw::Mesh::load(m_vk_backend, "mesh/bunny.gltf");

            if (!bunny)
            {
                DW_LOG_ERROR("Failed to load mesh");
                return false;
            }

            bunny->initialize_for_ray_tracing(m_vk_backend);

            m_gpu_resources->meshes.push_back(bunny);

            dw::Mesh::Ptr ground = dw::Mesh::load(m_vk_backend, "mesh/ground.gltf");

            if (!ground)
            {
                DW_LOG_ERROR("Failed to load mesh");
                return false;
            }

            ground->initialize_for_ray_tracing(m_vk_backend);

            m_gpu_resources->meshes.push_back(ground);

            float segment_length = (ground->max_extents().z - ground->min_extents().z) / (NUM_PILLARS + 1);

            for (uint32_t i = 0; i < NUM_PILLARS; i++)
            {
                dw::RayTracedScene::Instance pillar_instance;

                pillar_instance.mesh      = pillar;
                pillar_instance.transform = glm::mat4(1.0f);

                glm::vec3 pos = glm::vec3(15.0f, 0.0f, ground->min_extents().z + segment_length * (i + 1));

                pillar_instance.transform = glm::translate(pillar_instance.transform, pos);

                instances.push_back(pillar_instance);
            }

            for (uint32_t i = 0; i < NUM_PILLARS; i++)
            {
                dw::RayTracedScene::Instance pillar_instance;

                pillar_instance.mesh      = pillar;
                pillar_instance.transform = glm::mat4(1.0f);

                glm::vec3 pos = glm::vec3(-15.0f, 0.0f, ground->min_extents().z + segment_length * (i + 1));

                pillar_instance.transform = glm::translate(pillar_instance.transform, pos);

                instances.push_back(pillar_instance);
            }

            dw::RayTracedScene::Instance ground_instance;

            ground_instance.mesh      = ground;
            ground_instance.transform = glm::mat4(1.0f);

            instances.push_back(ground_instance);

            dw::RayTracedScene::Instance bunny_instance;

            bunny_instance.mesh = bunny;

            glm::mat4 S = glm::scale(glm::mat4(1.0f), glm::vec3(5.0f));
            glm::mat4 R = glm::rotate(glm::mat4(1.0f), glm::radians(135.0f), glm::vec3(0.0f, 1.0f, 0.0f));
            glm::mat4 T = glm::translate(glm::mat4(1.0f), glm::vec3(0.0f, -0.5f, 0.0f));

            bunny_instance.transform = T * R * S;

            instances.push_back(bunny_instance);

            m_gpu_resources->pillars_scene = dw::RayTracedScene::create(m_vk_backend, instances);
        }

        {
            std::vector<dw::RayTracedScene::Instance> instances;

            dw::Mesh::Ptr sponza = dw::Mesh::load(m_vk_backend, "mesh/sponza.obj");

            if (!sponza)
            {
                DW_LOG_ERROR("Failed to load mesh");
                return false;
            }

            sponza->initialize_for_ray_tracing(m_vk_backend);

            m_gpu_resources->meshes.push_back(sponza);

            dw::RayTracedScene::Instance sponza_instance;

            sponza_instance.mesh      = sponza;
            sponza_instance.transform = glm::scale(glm::mat4(1.0f), glm::vec3(0.3f));

            instances.push_back(sponza_instance);

            m_gpu_resources->sponza_scene = dw::RayTracedScene::create(m_vk_backend, instances);
        }

        {
            std::vector<dw::RayTracedScene::Instance> instances;

            dw::Mesh::Ptr pica_pica = dw::Mesh::load(m_vk_backend, "scene.gltf");

            if (!pica_pica)
            {
                DW_LOG_ERROR("Failed to load mesh");
                return false;
            }

            pica_pica->initialize_for_ray_tracing(m_vk_backend);

            m_gpu_resources->meshes.push_back(pica_pica);

            dw::RayTracedScene::Instance pica_pica_instance;

            pica_pica_instance.mesh      = pica_pica;
            pica_pica_instance.transform = glm::scale(glm::mat4(1.0f), glm::vec3(1.0f));

            instances.push_back(pica_pica_instance);

            m_rtao_ray_length = 7.0f;
            m_rtao_power      = 1.2f;

            m_gpu_resources->pica_pica_scene = dw::RayTracedScene::create(m_vk_backend, instances);
        }

        set_active_scene();

        return true;
    }

    // -----------------------------------------------------------------------------------------------------------------------------------

    void create_camera()
    {
        m_main_camera = std::make_unique<dw::Camera>(60.0f, m_near_plane, m_far_plane, float(m_width) / float(m_height), glm::vec3(0.0f, 35.0f, 125.0f), glm::vec3(0.0f, 0.0, -1.0f));
    }

    // -----------------------------------------------------------------------------------------------------------------------------------

    void ray_trace_shadows(dw::vk::CommandBuffer::Ptr cmd_buf)
    {
        DW_SCOPED_SAMPLE("Ray Traced Shadows", cmd_buf);

        VkImageSubresourceRange subresource_range = { VK_IMAGE_ASPECT_COLOR_BIT, 0, 1, 0, 1 };

        // Transition ray tracing output image back to general layout

        {
            std::vector<VkImageMemoryBarrier> image_barriers = {
                image_memory_barrier(m_gpu_resources->visibility_image, VK_IMAGE_LAYOUT_UNDEFINED, VK_IMAGE_LAYOUT_GENERAL, subresource_range, VK_ACCESS_SHADER_WRITE_BIT, VK_ACCESS_SHADER_READ_BIT)
            };

            pipeline_barrier(cmd_buf, {}, image_barriers, VK_PIPELINE_STAGE_FRAGMENT_SHADER_BIT, VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT);
        }

        vkCmdBindPipeline(cmd_buf->handle(), VK_PIPELINE_BIND_POINT_RAY_TRACING_KHR, m_gpu_resources->shadow_mask_pipeline->handle());

        ShadowPushConstants push_constants;

        push_constants.bias       = m_ray_traced_shadows_bias;
        push_constants.num_frames = m_num_frames;

        vkCmdPushConstants(cmd_buf->handle(), m_gpu_resources->shadow_mask_pipeline_layout->handle(), VK_SHADER_STAGE_RAYGEN_BIT_KHR, 0, sizeof(push_constants), &push_constants);

        const uint32_t dynamic_offset = m_ubo_size * m_vk_backend->current_frame_idx();

        VkDescriptorSet descriptor_sets[] = {
            m_gpu_resources->current_scene->descriptor_set()->handle(),
            m_gpu_resources->visibility_write_ds->handle(),
            m_gpu_resources->per_frame_ds->handle(),
            m_quarter_resolution ? m_gpu_resources->downsampled_g_buffer_ds[m_ping_pong]->handle() : m_gpu_resources->g_buffer_ds[m_ping_pong]->handle()
        };

        vkCmdBindDescriptorSets(cmd_buf->handle(), VK_PIPELINE_BIND_POINT_RAY_TRACING_KHR, m_gpu_resources->shadow_mask_pipeline_layout->handle(), 0, 4, descriptor_sets, 1, &dynamic_offset);

        auto& rt_pipeline_props = m_vk_backend->ray_tracing_pipeline_properties();

        VkDeviceSize group_size   = dw::vk::utilities::aligned_size(rt_pipeline_props.shaderGroupHandleSize, rt_pipeline_props.shaderGroupBaseAlignment);
        VkDeviceSize group_stride = group_size;

        const VkStridedDeviceAddressRegionKHR raygen_sbt   = { m_gpu_resources->shadow_mask_pipeline->shader_binding_table_buffer()->device_address(), group_stride, group_size };
        const VkStridedDeviceAddressRegionKHR miss_sbt     = { m_gpu_resources->shadow_mask_pipeline->shader_binding_table_buffer()->device_address() + m_gpu_resources->shadow_mask_sbt->miss_group_offset(), group_stride, group_size * 2 };
        const VkStridedDeviceAddressRegionKHR hit_sbt      = { m_gpu_resources->shadow_mask_pipeline->shader_binding_table_buffer()->device_address() + m_gpu_resources->shadow_mask_sbt->hit_group_offset(), group_stride, group_size * 2 };
        const VkStridedDeviceAddressRegionKHR callable_sbt = { VK_NULL_HANDLE, 0, 0 };

        uint32_t rt_image_width  = m_quarter_resolution ? m_width / 2 : m_width;
        uint32_t rt_image_height = m_quarter_resolution ? m_height / 2 : m_height;

        vkCmdTraceRaysKHR(cmd_buf->handle(), &raygen_sbt, &miss_sbt, &hit_sbt, &callable_sbt, rt_image_width, rt_image_height, 1);
    }

    // -----------------------------------------------------------------------------------------------------------------------------------

    void ray_trace_ambient_occlusion(dw::vk::CommandBuffer::Ptr cmd_buf)
    {
        DW_SCOPED_SAMPLE("Ray Traced Ambient Occlusion", cmd_buf);

        VkImageSubresourceRange subresource_range = { VK_IMAGE_ASPECT_COLOR_BIT, 0, 1, 0, 1 };

        std::vector<VkMemoryBarrier> memory_barriers = {
            memory_barrier(VK_ACCESS_SHADER_WRITE_BIT, VK_ACCESS_SHADER_READ_BIT)
        };

        pipeline_barrier(cmd_buf, memory_barriers, {}, VK_PIPELINE_STAGE_RAY_TRACING_SHADER_BIT_KHR, VK_PIPELINE_STAGE_RAY_TRACING_SHADER_BIT_KHR);

        vkCmdBindPipeline(cmd_buf->handle(), VK_PIPELINE_BIND_POINT_RAY_TRACING_KHR, m_gpu_resources->rtao_pipeline->handle());

        AmbientOcclusionPushConstants push_constants;

        push_constants.num_frames = m_num_frames;
        push_constants.num_rays   = m_rtao_num_rays;
        push_constants.ray_length = m_rtao_ray_length;
        push_constants.power      = m_rtao_power;
        push_constants.bias       = m_rtao_bias;

        vkCmdPushConstants(cmd_buf->handle(), m_gpu_resources->rtao_pipeline_layout->handle(), VK_SHADER_STAGE_RAYGEN_BIT_KHR, 0, sizeof(push_constants), &push_constants);

        const uint32_t dynamic_offset = m_ubo_size * m_vk_backend->current_frame_idx();

        VkDescriptorSet descriptor_sets[] = {
            m_gpu_resources->current_scene->descriptor_set()->handle(),
            m_gpu_resources->visibility_write_ds->handle(),
            m_gpu_resources->per_frame_ds->handle(),
            m_quarter_resolution ? m_gpu_resources->downsampled_g_buffer_ds[m_ping_pong]->handle() : m_gpu_resources->g_buffer_ds[m_ping_pong]->handle()
        };

        vkCmdBindDescriptorSets(cmd_buf->handle(), VK_PIPELINE_BIND_POINT_RAY_TRACING_KHR, m_gpu_resources->rtao_pipeline_layout->handle(), 0, 4, descriptor_sets, 1, &dynamic_offset);

        auto& rt_pipeline_props = m_vk_backend->ray_tracing_pipeline_properties();

        VkDeviceSize group_size   = dw::vk::utilities::aligned_size(rt_pipeline_props.shaderGroupHandleSize, rt_pipeline_props.shaderGroupBaseAlignment);
        VkDeviceSize group_stride = group_size;

        const VkStridedDeviceAddressRegionKHR raygen_sbt   = { m_gpu_resources->rtao_pipeline->shader_binding_table_buffer()->device_address(), group_stride, group_size };
        const VkStridedDeviceAddressRegionKHR miss_sbt     = { m_gpu_resources->rtao_pipeline->shader_binding_table_buffer()->device_address() + m_gpu_resources->rtao_sbt->miss_group_offset(), group_stride, group_size * 2 };
        const VkStridedDeviceAddressRegionKHR hit_sbt      = { m_gpu_resources->rtao_pipeline->shader_binding_table_buffer()->device_address() + m_gpu_resources->rtao_sbt->hit_group_offset(), group_stride, group_size * 2 };
        const VkStridedDeviceAddressRegionKHR callable_sbt = { VK_NULL_HANDLE, 0, 0 };

        uint32_t rt_image_width  = m_quarter_resolution ? m_width / 2 : m_width;
        uint32_t rt_image_height = m_quarter_resolution ? m_height / 2 : m_height;

        vkCmdTraceRaysKHR(cmd_buf->handle(), &raygen_sbt, &miss_sbt, &hit_sbt, &callable_sbt, rt_image_width, rt_image_height, 1);

        dw::vk::utilities::set_image_layout(
            cmd_buf->handle(),
            m_gpu_resources->visibility_image->handle(),
            VK_IMAGE_LAYOUT_GENERAL,
            VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL,
            subresource_range);
    }

    // -----------------------------------------------------------------------------------------------------------------------------------

    void ray_trace_reflection(dw::vk::CommandBuffer::Ptr cmd_buf)
    {
        DW_SCOPED_SAMPLE("Ray Traced Reflections", cmd_buf);

        VkImageSubresourceRange subresource_range = { VK_IMAGE_ASPECT_COLOR_BIT, 0, 1, 0, 1 };

        // Transition ray tracing output image back to general layout
        dw::vk::utilities::set_image_layout(
            cmd_buf->handle(),
            m_gpu_resources->reflection_rt_color_image->handle(),
            VK_IMAGE_LAYOUT_UNDEFINED,
            VK_IMAGE_LAYOUT_GENERAL,
            subresource_range);

        vkCmdBindPipeline(cmd_buf->handle(), VK_PIPELINE_BIND_POINT_RAY_TRACING_KHR, m_gpu_resources->reflection_rt_pipeline->handle());

        ReflectionsPushConstants push_constants;

        push_constants.bias       = m_ray_traced_reflections_bias;
        push_constants.num_frames = m_num_frames;

        vkCmdPushConstants(cmd_buf->handle(), m_gpu_resources->reflection_rt_pipeline_layout->handle(), VK_SHADER_STAGE_RAYGEN_BIT_KHR, 0, sizeof(push_constants), &push_constants);

        const uint32_t dynamic_offset = m_ubo_size * m_vk_backend->current_frame_idx();

        VkDescriptorSet descriptor_sets[] = {
            m_gpu_resources->current_scene->descriptor_set()->handle(),
            m_gpu_resources->reflection_rt_write_ds->handle(),
            m_gpu_resources->per_frame_ds->handle(),
            m_gpu_resources->g_buffer_ds[m_ping_pong]->handle(),
            m_gpu_resources->pbr_ds->handle(),
            m_gpu_resources->skybox_ds->handle()
        };

        vkCmdBindDescriptorSets(cmd_buf->handle(), VK_PIPELINE_BIND_POINT_RAY_TRACING_KHR, m_gpu_resources->reflection_rt_pipeline_layout->handle(), 0, 6, descriptor_sets, 1, &dynamic_offset);

        auto& rt_pipeline_props = m_vk_backend->ray_tracing_pipeline_properties();

        VkDeviceSize group_size   = dw::vk::utilities::aligned_size(rt_pipeline_props.shaderGroupHandleSize, rt_pipeline_props.shaderGroupBaseAlignment);
        VkDeviceSize group_stride = group_size;

        const VkStridedDeviceAddressRegionKHR raygen_sbt   = { m_gpu_resources->reflection_rt_pipeline->shader_binding_table_buffer()->device_address(), group_stride, group_size };
        const VkStridedDeviceAddressRegionKHR miss_sbt     = { m_gpu_resources->reflection_rt_pipeline->shader_binding_table_buffer()->device_address() + m_gpu_resources->reflection_rt_sbt->miss_group_offset(), group_stride, group_size * 2 };
        const VkStridedDeviceAddressRegionKHR hit_sbt      = { m_gpu_resources->reflection_rt_pipeline->shader_binding_table_buffer()->device_address() + m_gpu_resources->reflection_rt_sbt->hit_group_offset(), group_stride, group_size * 2 };
        const VkStridedDeviceAddressRegionKHR callable_sbt = { VK_NULL_HANDLE, 0, 0 };

        uint32_t rt_image_width  = m_quarter_resolution ? m_width / 2 : m_width;
        uint32_t rt_image_height = m_quarter_resolution ? m_height / 2 : m_height;

        vkCmdTraceRaysKHR(cmd_buf->handle(), &raygen_sbt, &miss_sbt, &hit_sbt, &callable_sbt, rt_image_width, rt_image_height, 1);

        dw::vk::utilities::set_image_layout(
            cmd_buf->handle(),
            m_gpu_resources->reflection_rt_color_image->handle(),
            VK_IMAGE_LAYOUT_GENERAL,
            VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL,
            subresource_range);
    }

    // -----------------------------------------------------------------------------------------------------------------------------------

    void ray_trace_gi(dw::vk::CommandBuffer::Ptr cmd_buf)
    {
        DW_SCOPED_SAMPLE("Ray Traced Global Illumination", cmd_buf);

        VkImageSubresourceRange subresource_range = { VK_IMAGE_ASPECT_COLOR_BIT, 0, 1, 0, 1 };

        // Transition ray tracing output image back to general layout
        dw::vk::utilities::set_image_layout(
            cmd_buf->handle(),
            m_gpu_resources->rtgi_image->handle(),
            VK_IMAGE_LAYOUT_UNDEFINED,
            VK_IMAGE_LAYOUT_GENERAL,
            subresource_range);

        vkCmdBindPipeline(cmd_buf->handle(), VK_PIPELINE_BIND_POINT_RAY_TRACING_KHR, m_gpu_resources->rtgi_pipeline->handle());

        GIPushConstants push_constants;

        push_constants.bias          = m_ray_traced_reflections_bias;
        push_constants.num_frames    = m_num_frames;
        push_constants.max_ray_depth = m_ray_traced_gi_max_ray_bounces - 1;
        push_constants.sample_sky    = static_cast<uint32_t>(m_ray_traced_gi_sample_sky);

        vkCmdPushConstants(cmd_buf->handle(), m_gpu_resources->rtgi_pipeline_layout->handle(), VK_SHADER_STAGE_RAYGEN_BIT_KHR | VK_SHADER_STAGE_CLOSEST_HIT_BIT_KHR, 0, sizeof(push_constants), &push_constants);

        const uint32_t dynamic_offset = m_ubo_size * m_vk_backend->current_frame_idx();

        VkDescriptorSet descriptor_sets[] = {
            m_gpu_resources->current_scene->descriptor_set()->handle(),
            m_gpu_resources->rtgi_write_ds->handle(),
            m_gpu_resources->per_frame_ds->handle(),
            m_gpu_resources->g_buffer_ds[m_ping_pong]->handle(),
            m_gpu_resources->pbr_ds->handle(),
            m_gpu_resources->skybox_ds->handle()
        };

        vkCmdBindDescriptorSets(cmd_buf->handle(), VK_PIPELINE_BIND_POINT_RAY_TRACING_KHR, m_gpu_resources->rtgi_pipeline_layout->handle(), 0, 6, descriptor_sets, 1, &dynamic_offset);

        auto& rt_pipeline_props = m_vk_backend->ray_tracing_pipeline_properties();

        VkDeviceSize group_size   = dw::vk::utilities::aligned_size(rt_pipeline_props.shaderGroupHandleSize, rt_pipeline_props.shaderGroupBaseAlignment);
        VkDeviceSize group_stride = group_size;

        const VkStridedDeviceAddressRegionKHR raygen_sbt   = { m_gpu_resources->rtgi_pipeline->shader_binding_table_buffer()->device_address(), group_stride, group_size };
        const VkStridedDeviceAddressRegionKHR miss_sbt     = { m_gpu_resources->rtgi_pipeline->shader_binding_table_buffer()->device_address() + m_gpu_resources->rtgi_sbt->miss_group_offset(), group_stride, group_size * 2 };
        const VkStridedDeviceAddressRegionKHR hit_sbt      = { m_gpu_resources->rtgi_pipeline->shader_binding_table_buffer()->device_address() + m_gpu_resources->rtgi_sbt->hit_group_offset(), group_stride, group_size * 2 };
        const VkStridedDeviceAddressRegionKHR callable_sbt = { VK_NULL_HANDLE, 0, 0 };

        uint32_t rt_image_width  = m_downscaled_rt ? m_width / 2 : m_width;
        uint32_t rt_image_height = m_downscaled_rt ? m_height / 2 : m_height;

        vkCmdTraceRaysKHR(cmd_buf->handle(), &raygen_sbt, &miss_sbt, &hit_sbt, &callable_sbt, rt_image_width, rt_image_height, 1);

        dw::vk::utilities::set_image_layout(
            cmd_buf->handle(),
            m_gpu_resources->rtgi_image->handle(),
            VK_IMAGE_LAYOUT_GENERAL,
            VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL,
            subresource_range);
    }

    // -----------------------------------------------------------------------------------------------------------------------------------

    void reflection_spatial_resolve(dw::vk::CommandBuffer::Ptr cmd_buf)
    {
        DW_SCOPED_SAMPLE("Reflection Spatial Resolve", cmd_buf);

        VkImageSubresourceRange subresource_range = { VK_IMAGE_ASPECT_COLOR_BIT, 0, 1, 0, 1 };

        dw::vk::utilities::set_image_layout(
            cmd_buf->handle(),
            m_gpu_resources->reflection_resolve_image->handle(),
            VK_IMAGE_LAYOUT_UNDEFINED,
            VK_IMAGE_LAYOUT_GENERAL,
            subresource_range);

        const uint32_t NUM_THREADS = 32;

        vkCmdBindPipeline(cmd_buf->handle(), VK_PIPELINE_BIND_POINT_COMPUTE, m_gpu_resources->reflection_resolve_pipeline->handle());

        ReflectionsSpatialResolvePushConstants push_constants;

        float z_buffer_params_x = -1.0 + (m_near_plane / m_far_plane);

        push_constants.z_buffer_params = glm::vec4(z_buffer_params_x, 1.0f, z_buffer_params_x / m_near_plane, 1.0f / m_near_plane);
        push_constants.bypass          = (uint32_t)!m_ray_traced_reflections_spatial_resolve;

        vkCmdPushConstants(cmd_buf->handle(), m_gpu_resources->reflection_resolve_pipeline_layout->handle(), VK_SHADER_STAGE_COMPUTE_BIT, 0, sizeof(push_constants), &push_constants);

        const uint32_t dynamic_offset = m_ubo_size * m_vk_backend->current_frame_idx();

        VkDescriptorSet descriptor_sets[] = {
            m_gpu_resources->reflection_resolve_write_ds->handle(),
            m_gpu_resources->reflection_rt_read_ds->handle(),
            m_gpu_resources->g_buffer_ds[m_ping_pong]->handle(),
            m_gpu_resources->per_frame_ds->handle()
        };

        vkCmdBindDescriptorSets(cmd_buf->handle(), VK_PIPELINE_BIND_POINT_COMPUTE, m_gpu_resources->reflection_resolve_pipeline_layout->handle(), 0, 4, descriptor_sets, 1, &dynamic_offset);

        vkCmdDispatch(cmd_buf->handle(), static_cast<uint32_t>(ceil(float(m_width) / float(NUM_THREADS))), static_cast<uint32_t>(ceil(float(m_height) / float(NUM_THREADS))), 1);

        // Prepare ray tracing output image as transfer source
        dw::vk::utilities::set_image_layout(
            cmd_buf->handle(),
            m_gpu_resources->reflection_resolve_image->handle(),
            VK_IMAGE_LAYOUT_GENERAL,
            VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL,
            subresource_range);
    }

    // -----------------------------------------------------------------------------------------------------------------------------------

    void reflection_temporal(dw::vk::CommandBuffer::Ptr cmd_buf)
    {
        DW_SCOPED_SAMPLE("Reflection Temporal", cmd_buf);

        const uint32_t NUM_THREADS = 32;
        const uint32_t read_idx    = (uint32_t)!m_ping_pong;

        VkImageSubresourceRange subresource_range = { VK_IMAGE_ASPECT_COLOR_BIT, 0, 1, 0, 1 };

        dw::vk::utilities::set_image_layout(
            cmd_buf->handle(),
            m_gpu_resources->reflection_temporal_image->handle(),
            VK_IMAGE_LAYOUT_UNDEFINED,
            VK_IMAGE_LAYOUT_GENERAL,
            subresource_range);

        if (m_first_frame)
        {
            blitt_image(cmd_buf,
                        m_gpu_resources->reflection_resolve_image,
                        m_gpu_resources->reflection_blur_image[read_idx],
                        VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL,
                        VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL,
                        VK_IMAGE_LAYOUT_UNDEFINED,
                        VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL,
                        VK_IMAGE_ASPECT_COLOR_BIT,
                        VK_FILTER_NEAREST);
        }

        vkCmdBindPipeline(cmd_buf->handle(), VK_PIPELINE_BIND_POINT_COMPUTE, m_gpu_resources->reflection_temporal_pipeline->handle());

        ReflectionsTemporalPushConstants push_constants;

        push_constants.first_frame            = (uint32_t)m_first_frame;
        push_constants.neighborhood_clamping  = (uint32_t)m_rt_reflections_neighborhood_clamping;
        push_constants.neighborhood_std_scale = m_ray_traced_reflections_std_scale;
        push_constants.alpha                  = m_ray_traced_reflections_alpha;

        vkCmdPushConstants(cmd_buf->handle(), m_gpu_resources->reflection_temporal_pipeline_layout->handle(), VK_SHADER_STAGE_COMPUTE_BIT, 0, sizeof(push_constants), &push_constants);

        const uint32_t dynamic_offset = m_ubo_size * m_vk_backend->current_frame_idx();

        VkDescriptorSet descriptor_sets[] = {
            m_gpu_resources->reflection_temporal_write_ds->handle(),
            m_gpu_resources->reflection_resolve_read_ds->handle(),
            m_gpu_resources->reflection_blur_read_ds[read_idx]->handle(),
            m_gpu_resources->g_buffer_ds[m_ping_pong]->handle(),
            m_gpu_resources->per_frame_ds->handle()
        };

        vkCmdBindDescriptorSets(cmd_buf->handle(), VK_PIPELINE_BIND_POINT_COMPUTE, m_gpu_resources->reflection_temporal_pipeline_layout->handle(), 0, 5, descriptor_sets, 1, &dynamic_offset);

        vkCmdDispatch(cmd_buf->handle(), m_width / NUM_THREADS, m_height / NUM_THREADS, 1);

        // Prepare ray tracing output image as transfer source
        dw::vk::utilities::set_image_layout(
            cmd_buf->handle(),
            m_gpu_resources->reflection_temporal_image->handle(),
            VK_IMAGE_LAYOUT_GENERAL,
            VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL,
            subresource_range);
    }

    // -----------------------------------------------------------------------------------------------------------------------------------

    void reflection_blur(dw::vk::CommandBuffer::Ptr cmd_buf)
    {
        DW_SCOPED_SAMPLE("Reflection Blur", cmd_buf);

        const uint32_t NUM_THREADS = 32;
        const uint32_t write_idx   = (uint32_t)m_ping_pong;
        const uint32_t read_idx    = (uint32_t)!m_ping_pong;

        VkImageSubresourceRange subresource_range = { VK_IMAGE_ASPECT_COLOR_BIT, 0, 1, 0, 1 };

        dw::vk::utilities::set_image_layout(
            cmd_buf->handle(),
            m_gpu_resources->reflection_blur_image[write_idx]->handle(),
            VK_IMAGE_LAYOUT_UNDEFINED,
            VK_IMAGE_LAYOUT_GENERAL,
            subresource_range);

        dw::vk::utilities::set_image_layout(
            cmd_buf->handle(),
            m_gpu_resources->reflection_history_image[write_idx]->handle(),
            VK_IMAGE_LAYOUT_UNDEFINED,
            VK_IMAGE_LAYOUT_GENERAL,
            subresource_range);

        vkCmdBindPipeline(cmd_buf->handle(), VK_PIPELINE_BIND_POINT_COMPUTE, m_gpu_resources->reflection_blur_pipeline->handle());

        ReflectionsBlurPushConstants push_constants;

        push_constants.alpha = m_svgf_alpha;

        vkCmdPushConstants(cmd_buf->handle(), m_gpu_resources->reflection_blur_pipeline_layout->handle(), VK_SHADER_STAGE_COMPUTE_BIT, 0, sizeof(push_constants), &push_constants);

        VkDescriptorSet descriptor_sets[] = {
            m_gpu_resources->reflection_blur_write_ds[write_idx]->handle(),
            m_gpu_resources->reflection_history_ds[write_idx]->handle(),
            m_gpu_resources->downsampled_g_buffer_ds[m_ping_pong]->handle(),
            m_gpu_resources->downsampled_g_buffer_ds[!m_ping_pong]->handle(),
            m_gpu_resources->reflection_rt_read_ds->handle(),
            m_gpu_resources->reflection_blur_read_ds[read_idx]->handle(),
            m_gpu_resources->per_frame_ds->handle()
        };

        const uint32_t dynamic_offset = m_ubo_size * m_vk_backend->current_frame_idx();

        vkCmdBindDescriptorSets(cmd_buf->handle(), VK_PIPELINE_BIND_POINT_COMPUTE, m_gpu_resources->reflection_blur_pipeline_layout->handle(), 0, 7, descriptor_sets, 1, &dynamic_offset);

        uint32_t rt_image_width  = m_quarter_resolution ? m_width / 2 : m_width;
        uint32_t rt_image_height = m_quarter_resolution ? m_height / 2 : m_height;

        vkCmdDispatch(cmd_buf->handle(), static_cast<uint32_t>(ceil(float(rt_image_width) / float(NUM_THREADS))), static_cast<uint32_t>(ceil(float(rt_image_height) / float(NUM_THREADS))), 1);

        // Prepare ray tracing output image as transfer source
        dw::vk::utilities::set_image_layout(
            cmd_buf->handle(),
            m_gpu_resources->reflection_blur_image[write_idx]->handle(),
            VK_IMAGE_LAYOUT_GENERAL,
            VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL,
            subresource_range);

        dw::vk::utilities::set_image_layout(
            cmd_buf->handle(),
            m_gpu_resources->reflection_history_image[write_idx]->handle(),
            VK_IMAGE_LAYOUT_GENERAL,
            VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL,
            subresource_range);
    }

    // -----------------------------------------------------------------------------------------------------------------------------------

    void render_gbuffer(dw::vk::CommandBuffer::Ptr cmd_buf)
    {
        DW_SCOPED_SAMPLE("G-Buffer", cmd_buf);

        VkClearValue clear_values[5];

        clear_values[0].color.float32[0] = 0.0f;
        clear_values[0].color.float32[1] = 0.0f;
        clear_values[0].color.float32[2] = 0.0f;
        clear_values[0].color.float32[3] = 1.0f;

        clear_values[1].color.float32[0] = 0.0f;
        clear_values[1].color.float32[1] = 0.0f;
        clear_values[1].color.float32[2] = 0.0f;
        clear_values[1].color.float32[3] = 1.0f;

        clear_values[2].color.float32[0] = 0.0f;
        clear_values[2].color.float32[1] = 0.0f;
        clear_values[2].color.float32[2] = 0.0f;
        clear_values[2].color.float32[3] = 1.0f;

        clear_values[3].color.float32[0] = 0.0f;
        clear_values[3].color.float32[1] = 0.0f;
        clear_values[3].color.float32[2] = 0.0f;
        clear_values[3].color.float32[3] = 1.0f;

        clear_values[4].color.float32[0] = 1.0f;
        clear_values[4].color.float32[1] = 1.0f;
        clear_values[4].color.float32[2] = 1.0f;
        clear_values[4].color.float32[3] = 1.0f;

        VkRenderPassBeginInfo info    = {};
        info.sType                    = VK_STRUCTURE_TYPE_RENDER_PASS_BEGIN_INFO;
        info.renderPass               = m_gpu_resources->g_buffer_rp->handle();
        info.framebuffer              = m_gpu_resources->g_buffer_fbo[m_ping_pong]->handle();
        info.renderArea.extent.width  = m_width;
        info.renderArea.extent.height = m_height;
        info.clearValueCount          = 5;
        info.pClearValues             = &clear_values[0];

        vkCmdBeginRenderPass(cmd_buf->handle(), &info, VK_SUBPASS_CONTENTS_INLINE);

        VkViewport vp;

        vp.x        = 0.0f;
        vp.y        = 0.0f;
        vp.width    = (float)m_width;
        vp.height   = (float)m_height;
        vp.minDepth = 0.0f;
        vp.maxDepth = 1.0f;

        vkCmdSetViewport(cmd_buf->handle(), 0, 1, &vp);

        VkRect2D scissor_rect;

        scissor_rect.extent.width  = m_width;
        scissor_rect.extent.height = m_height;
        scissor_rect.offset.x      = 0;
        scissor_rect.offset.y      = 0;

        vkCmdSetScissor(cmd_buf->handle(), 0, 1, &scissor_rect);

        vkCmdBindPipeline(cmd_buf->handle(), VK_PIPELINE_BIND_POINT_GRAPHICS, m_gpu_resources->g_buffer_pipeline->handle());

        const uint32_t dynamic_offset = m_ubo_size * m_vk_backend->current_frame_idx();

        VkDescriptorSet descriptor_sets[] = {
            m_gpu_resources->current_scene->descriptor_set()->handle(),
            m_gpu_resources->per_frame_ds->handle()
        };

        vkCmdBindDescriptorSets(cmd_buf->handle(), VK_PIPELINE_BIND_POINT_GRAPHICS, m_gpu_resources->g_buffer_pipeline_layout->handle(), 0, 2, descriptor_sets, 1, &dynamic_offset);

        const auto& instances = m_gpu_resources->current_scene->instances();

        for (uint32_t instance_idx = 0; instance_idx < instances.size(); instance_idx++)
        {
            const auto& instance = instances[instance_idx];

            if (!instance.mesh.expired())
            {
                const auto& mesh      = instance.mesh.lock();
                const auto& submeshes = mesh->sub_meshes();

                VkDeviceSize offset = 0;
                vkCmdBindVertexBuffers(cmd_buf->handle(), 0, 1, &mesh->vertex_buffer()->handle(), &offset);
                vkCmdBindIndexBuffer(cmd_buf->handle(), mesh->index_buffer()->handle(), 0, VK_INDEX_TYPE_UINT32);

                for (uint32_t submesh_idx = 0; submesh_idx < submeshes.size(); submesh_idx++)
                {
                    auto& submesh = submeshes[submesh_idx];
                    auto& mat     = mesh->material(submesh.mat_idx);

                    GBufferPushConstants push_constants;

                    push_constants.model          = instance.transform;
                    push_constants.prev_model     = instance.transform;
                    push_constants.material_index = m_gpu_resources->current_scene->material_index(mat->id());

                    vkCmdPushConstants(cmd_buf->handle(), m_gpu_resources->g_buffer_pipeline_layout->handle(), VK_SHADER_STAGE_VERTEX_BIT | VK_SHADER_STAGE_FRAGMENT_BIT, 0, sizeof(GBufferPushConstants), &push_constants);

                    // Issue draw call.
                    vkCmdDrawIndexed(cmd_buf->handle(), submesh.index_count, 1, submesh.base_index, submesh.base_vertex, 0);
                }
            }
        }

        vkCmdEndRenderPass(cmd_buf->handle());
    }

    // -----------------------------------------------------------------------------------------------------------------------------------

    void downsample_gbuffer(dw::vk::CommandBuffer::Ptr cmd_buf)
    {
        DW_SCOPED_SAMPLE("Downsample G-Buffer", cmd_buf);

        blitt_image(cmd_buf,
                    m_gpu_resources->g_buffer_1,
                    m_gpu_resources->downsampled_g_buffer_1,
                    VK_IMAGE_LAYOUT_TRANSFER_SRC_OPTIMAL,
                    VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL,
                    VK_IMAGE_LAYOUT_UNDEFINED,
                    VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL,
                    VK_IMAGE_ASPECT_COLOR_BIT,
                    VK_FILTER_NEAREST);
        blitt_image(cmd_buf,
                    m_gpu_resources->g_buffer_2,
                    m_gpu_resources->downsampled_g_buffer_2,
                    VK_IMAGE_LAYOUT_TRANSFER_SRC_OPTIMAL,
                    VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL,
                    VK_IMAGE_LAYOUT_UNDEFINED,
                    VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL,
                    VK_IMAGE_ASPECT_COLOR_BIT,
                    VK_FILTER_NEAREST);
        blitt_image(cmd_buf,
                    m_gpu_resources->g_buffer_3,
                    m_gpu_resources->downsampled_g_buffer_3,
                    VK_IMAGE_LAYOUT_TRANSFER_SRC_OPTIMAL,
                    VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL,
                    VK_IMAGE_LAYOUT_UNDEFINED,
                    VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL,
                    VK_IMAGE_ASPECT_COLOR_BIT,
                    VK_FILTER_NEAREST);
        blitt_image(cmd_buf,
                    m_gpu_resources->g_buffer_depth,
                    m_gpu_resources->downsampled_g_buffer_depth,
                    VK_IMAGE_LAYOUT_TRANSFER_SRC_OPTIMAL,
                    VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL,
                    VK_IMAGE_LAYOUT_UNDEFINED,
                    VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL,
                    VK_IMAGE_ASPECT_DEPTH_BIT,
                    VK_FILTER_NEAREST);
        blitt_image(cmd_buf,
                    m_gpu_resources->g_buffer_linear_z[m_ping_pong],
                    m_gpu_resources->downsampled_g_buffer_linear_z[m_ping_pong],
                    VK_IMAGE_LAYOUT_TRANSFER_SRC_OPTIMAL,
                    VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL,
                    VK_IMAGE_LAYOUT_UNDEFINED,
                    VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL,
                    VK_IMAGE_ASPECT_COLOR_BIT,
                    VK_FILTER_NEAREST);
    }

    // -----------------------------------------------------------------------------------------------------------------------------------

    void blitt_image(dw::vk::CommandBuffer::Ptr cmd_buf,
                     dw::vk::Image::Ptr         src,
                     dw::vk::Image::Ptr         dst,
                     VkImageLayout              src_img_src_layout,
                     VkImageLayout              src_img_dst_layout,
                     VkImageLayout              dst_img_src_layout,
                     VkImageLayout              dst_img_dst_layout,
                     VkImageAspectFlags         aspect_flags,
                     VkFilter                   filter)
    {
        VkImageSubresourceRange initial_subresource_range;
        DW_ZERO_MEMORY(initial_subresource_range);

        initial_subresource_range.aspectMask     = aspect_flags;
        initial_subresource_range.levelCount     = 1;
        initial_subresource_range.layerCount     = 1;
        initial_subresource_range.baseArrayLayer = 0;
        initial_subresource_range.baseMipLevel   = 0;

        if (src_img_src_layout != VK_IMAGE_LAYOUT_TRANSFER_SRC_OPTIMAL)
        {
            dw::vk::utilities::set_image_layout(cmd_buf->handle(),
                                                src->handle(),
                                                src_img_src_layout,
                                                VK_IMAGE_LAYOUT_TRANSFER_SRC_OPTIMAL,
                                                initial_subresource_range);
        }

        dw::vk::utilities::set_image_layout(cmd_buf->handle(),
                                            dst->handle(),
                                            dst_img_src_layout,
                                            VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL,
                                            initial_subresource_range);

        VkImageBlit blit                   = {};
        blit.srcOffsets[0]                 = { 0, 0, 0 };
        blit.srcOffsets[1]                 = { static_cast<int32_t>(src->width()), static_cast<int32_t>(src->height()), 1 };
        blit.srcSubresource.aspectMask     = aspect_flags;
        blit.srcSubresource.mipLevel       = 0;
        blit.srcSubresource.baseArrayLayer = 0;
        blit.srcSubresource.layerCount     = 1;
        blit.dstOffsets[0]                 = { 0, 0, 0 };
        blit.dstOffsets[1]                 = { static_cast<int32_t>(dst->width()), static_cast<int32_t>(dst->height()), 1 };
        blit.dstSubresource.aspectMask     = aspect_flags;
        blit.dstSubresource.mipLevel       = 0;
        blit.dstSubresource.baseArrayLayer = 0;
        blit.dstSubresource.layerCount     = 1;

        vkCmdBlitImage(cmd_buf->handle(),
                       src->handle(),
                       VK_IMAGE_LAYOUT_TRANSFER_SRC_OPTIMAL,
                       dst->handle(),
                       VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL,
                       1,
                       &blit,
                       filter);

        dw::vk::utilities::set_image_layout(cmd_buf->handle(),
                                            src->handle(),
                                            VK_IMAGE_LAYOUT_TRANSFER_SRC_OPTIMAL,
                                            src_img_dst_layout,
                                            initial_subresource_range);

        dw::vk::utilities::set_image_layout(cmd_buf->handle(),
                                            dst->handle(),
                                            VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL,
                                            dst_img_dst_layout,
                                            initial_subresource_range);
    }

    // -----------------------------------------------------------------------------------------------------------------------------------

    void render_skybox(dw::vk::CommandBuffer::Ptr cmd_buf)
    {
        DW_SCOPED_SAMPLE("Deferred Shading", cmd_buf);

        VkRenderPassBeginInfo info    = {};
        info.sType                    = VK_STRUCTURE_TYPE_RENDER_PASS_BEGIN_INFO;
        info.renderPass               = m_gpu_resources->skybox_rp->handle();
        info.framebuffer              = m_gpu_resources->skybox_fbo->handle();
        info.renderArea.extent.width  = m_width;
        info.renderArea.extent.height = m_height;
        info.clearValueCount          = 0;
        info.pClearValues             = nullptr;

        vkCmdBeginRenderPass(cmd_buf->handle(), &info, VK_SUBPASS_CONTENTS_INLINE);

        VkViewport vp;

        vp.x        = 0.0f;
        vp.y        = 0.0f;
        vp.width    = (float)m_width;
        vp.height   = (float)m_height;
        vp.minDepth = 0.0f;
        vp.maxDepth = 1.0f;

        vkCmdSetViewport(cmd_buf->handle(), 0, 1, &vp);

        VkRect2D scissor_rect;

        scissor_rect.extent.width  = m_width;
        scissor_rect.extent.height = m_height;
        scissor_rect.offset.x      = 0;
        scissor_rect.offset.y      = 0;

        vkCmdSetScissor(cmd_buf->handle(), 0, 1, &scissor_rect);

        vkCmdBindPipeline(cmd_buf->handle(), VK_PIPELINE_BIND_POINT_GRAPHICS, m_gpu_resources->skybox_pipeline->handle());
        vkCmdBindDescriptorSets(cmd_buf->handle(), VK_PIPELINE_BIND_POINT_GRAPHICS, m_gpu_resources->skybox_pipeline_layout->handle(), 0, 1, &m_gpu_resources->skybox_ds->handle(), 0, nullptr);

        SkyboxPushConstants push_constants;

        push_constants.projection = m_projection;
        push_constants.view       = m_main_camera->m_view;

        vkCmdPushConstants(cmd_buf->handle(), m_gpu_resources->skybox_pipeline_layout->handle(), VK_SHADER_STAGE_VERTEX_BIT, 0, sizeof(push_constants), &push_constants);

        const VkBuffer     buffer = m_gpu_resources->cube_vbo->handle();
        const VkDeviceSize size   = 0;
        vkCmdBindVertexBuffers(cmd_buf->handle(), 0, 1, &buffer, &size);

        vkCmdDraw(cmd_buf->handle(), 36, 1, 0, 0);

        vkCmdEndRenderPass(cmd_buf->handle());
    }

    // -----------------------------------------------------------------------------------------------------------------------------------

    void deferred_shading(dw::vk::CommandBuffer::Ptr cmd_buf)
    {
        DW_SCOPED_SAMPLE("Deferred Shading", cmd_buf);

        VkClearValue clear_value;

        clear_value.color.float32[0] = 0.0f;
        clear_value.color.float32[1] = 0.0f;
        clear_value.color.float32[2] = 0.0f;
        clear_value.color.float32[3] = 1.0f;

        VkRenderPassBeginInfo info    = {};
        info.sType                    = VK_STRUCTURE_TYPE_RENDER_PASS_BEGIN_INFO;
        info.renderPass               = m_gpu_resources->deferred_rp->handle();
        info.framebuffer              = m_gpu_resources->deferred_fbo->handle();
        info.renderArea.extent.width  = m_width;
        info.renderArea.extent.height = m_height;
        info.clearValueCount          = 1;
        info.pClearValues             = &clear_value;

        vkCmdBeginRenderPass(cmd_buf->handle(), &info, VK_SUBPASS_CONTENTS_INLINE);

        VkViewport vp;

        vp.x        = 0.0f;
        vp.y        = 0.0f;
        vp.width    = (float)m_width;
        vp.height   = (float)m_height;
        vp.minDepth = 0.0f;
        vp.maxDepth = 1.0f;

        vkCmdSetViewport(cmd_buf->handle(), 0, 1, &vp);

        VkRect2D scissor_rect;

        scissor_rect.extent.width  = m_width;
        scissor_rect.extent.height = m_height;
        scissor_rect.offset.x      = 0;
        scissor_rect.offset.y      = 0;

        vkCmdSetScissor(cmd_buf->handle(), 0, 1, &scissor_rect);

        vkCmdBindPipeline(cmd_buf->handle(), VK_PIPELINE_BIND_POINT_GRAPHICS, m_gpu_resources->deferred_pipeline->handle());

        DeferredShadingPushConstants push_constants;

        push_constants.shadows     = (float)m_rt_shadows_enabled;
        push_constants.ao          = (float)m_rtao_enabled;
        push_constants.reflections = (float)m_rt_reflections_enabled;

        vkCmdPushConstants(cmd_buf->handle(), m_gpu_resources->deferred_pipeline_layout->handle(), VK_SHADER_STAGE_FRAGMENT_BIT, 0, sizeof(push_constants), &push_constants);

        const uint32_t dynamic_offset = m_ubo_size * m_vk_backend->current_frame_idx();

        VkDescriptorSet descriptor_sets[] = {
            m_gpu_resources->g_buffer_ds[m_ping_pong]->handle(),
            //m_gpu_resources->rtgi_temporal_read_ds[(uint32_t)m_ping_pong]->handle(),
            m_gpu_resources->svgf_shadow_denoiser->denoised_image_ds()->handle(),
            m_gpu_resources->reflection_blur_read_ds[(uint32_t)m_ping_pong]->handle(),
            m_gpu_resources->svgf_gi_denoiser->denoised_image_ds()->handle(),
            //m_gpu_resources->rtgi_temporal_read_ds[(uint32_t)m_ping_pong]->handle(),
            m_gpu_resources->per_frame_ds->handle(),
            m_gpu_resources->pbr_ds->handle()
        };

        vkCmdBindDescriptorSets(cmd_buf->handle(), VK_PIPELINE_BIND_POINT_GRAPHICS, m_gpu_resources->deferred_pipeline_layout->handle(), 0, 6, descriptor_sets, 1, &dynamic_offset);

        vkCmdDraw(cmd_buf->handle(), 3, 1, 0, 0);

        vkCmdEndRenderPass(cmd_buf->handle());
    }

    // -----------------------------------------------------------------------------------------------------------------------------------

    void temporal_aa(dw::vk::CommandBuffer::Ptr cmd_buf)
    {
        DW_SCOPED_SAMPLE("TAA", cmd_buf);

        const uint32_t NUM_THREADS = 32;
        const uint32_t write_idx   = (uint32_t)m_ping_pong;
        const uint32_t read_idx    = (uint32_t)!m_ping_pong;

        if (m_taa_enabled)
        {
            VkImageSubresourceRange subresource_range = { VK_IMAGE_ASPECT_COLOR_BIT, 0, 1, 0, 1 };

            dw::vk::utilities::set_image_layout(
                cmd_buf->handle(),
                m_gpu_resources->taa_image[write_idx]->handle(),
                VK_IMAGE_LAYOUT_UNDEFINED,
                VK_IMAGE_LAYOUT_GENERAL,
                subresource_range);

            if (m_first_frame)
            {
                blitt_image(cmd_buf,
                            m_gpu_resources->deferred_image,
                            m_gpu_resources->taa_image[read_idx],
                            VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL,
                            VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL,
                            VK_IMAGE_LAYOUT_UNDEFINED,
                            VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL,
                            VK_IMAGE_ASPECT_COLOR_BIT,
                            VK_FILTER_NEAREST);
            }

            vkCmdBindPipeline(cmd_buf->handle(), VK_PIPELINE_BIND_POINT_COMPUTE, m_gpu_resources->taa_pipeline->handle());

            TAAPushConstants push_constants;

            push_constants.texel_size          = glm::vec4(1.0f / float(m_width), 1.0f / float(m_height), float(m_width), float(m_height));
            push_constants.current_prev_jitter = glm::vec4(m_current_jitter, m_prev_jitter);
            push_constants.time_params         = glm::vec4(static_cast<float>(glfwGetTime()), sinf(static_cast<float>(glfwGetTime())), cosf(static_cast<float>(glfwGetTime())), static_cast<float>(m_delta_seconds));
            push_constants.feedback_min        = m_taa_feedback_min;
            push_constants.feedback_max        = m_taa_feedback_max;
            push_constants.sharpen             = static_cast<int>(m_taa_sharpen);

            vkCmdPushConstants(cmd_buf->handle(), m_gpu_resources->taa_pipeline_layout->handle(), VK_SHADER_STAGE_COMPUTE_BIT, 0, sizeof(push_constants), &push_constants);

            VkDescriptorSet descriptor_sets[] = {
                m_gpu_resources->taa_write_ds[write_idx]->handle(),
                m_gpu_resources->deferred_read_ds->handle(),
                m_gpu_resources->taa_read_ds[read_idx]->handle(),
                m_gpu_resources->g_buffer_ds[m_ping_pong]->handle()
            };

            vkCmdBindDescriptorSets(cmd_buf->handle(), VK_PIPELINE_BIND_POINT_COMPUTE, m_gpu_resources->taa_pipeline_layout->handle(), 0, 4, descriptor_sets, 0, nullptr);

            vkCmdDispatch(cmd_buf->handle(), m_width / NUM_THREADS, m_height / NUM_THREADS, 1);

            // Prepare ray tracing output image as transfer source
            dw::vk::utilities::set_image_layout(
                cmd_buf->handle(),
                m_gpu_resources->taa_image[write_idx]->handle(),
                VK_IMAGE_LAYOUT_GENERAL,
                VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL,
                subresource_range);
        }
        else
        {
            blitt_image(cmd_buf,
                        m_gpu_resources->deferred_image,
                        m_gpu_resources->taa_image[write_idx],
                        VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL,
                        VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL,
                        VK_IMAGE_LAYOUT_UNDEFINED,
                        VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL,
                        VK_IMAGE_ASPECT_COLOR_BIT,
                        VK_FILTER_NEAREST);
        }
    }

    // -----------------------------------------------------------------------------------------------------------------------------------

    void tone_map(dw::vk::CommandBuffer::Ptr cmd_buf)
    {
        DW_SCOPED_SAMPLE("Tone Map", cmd_buf);

        VkClearValue clear_values[2];

        clear_values[0].color.float32[0] = 0.0f;
        clear_values[0].color.float32[1] = 0.0f;
        clear_values[0].color.float32[2] = 0.0f;
        clear_values[0].color.float32[3] = 1.0f;

        clear_values[1].color.float32[0] = 1.0f;
        clear_values[1].color.float32[1] = 1.0f;
        clear_values[1].color.float32[2] = 1.0f;
        clear_values[1].color.float32[3] = 1.0f;

        VkRenderPassBeginInfo info    = {};
        info.sType                    = VK_STRUCTURE_TYPE_RENDER_PASS_BEGIN_INFO;
        info.renderPass               = m_vk_backend->swapchain_render_pass()->handle();
        info.framebuffer              = m_vk_backend->swapchain_framebuffer()->handle();
        info.renderArea.extent.width  = m_width;
        info.renderArea.extent.height = m_height;
        info.clearValueCount          = 2;
        info.pClearValues             = &clear_values[0];

        vkCmdBeginRenderPass(cmd_buf->handle(), &info, VK_SUBPASS_CONTENTS_INLINE);

        VkViewport vp;

        vp.x        = 0.0f;
        vp.y        = (float)m_height;
        vp.width    = (float)m_width;
        vp.height   = -(float)m_height;
        vp.minDepth = 0.0f;
        vp.maxDepth = 1.0f;

        vkCmdSetViewport(cmd_buf->handle(), 0, 1, &vp);

        VkRect2D scissor_rect;

        scissor_rect.extent.width  = m_width;
        scissor_rect.extent.height = m_height;
        scissor_rect.offset.x      = 0;
        scissor_rect.offset.y      = 0;

        vkCmdSetScissor(cmd_buf->handle(), 0, 1, &scissor_rect);

        vkCmdBindPipeline(cmd_buf->handle(), VK_PIPELINE_BIND_POINT_GRAPHICS, m_gpu_resources->copy_pipeline->handle());

        VkDescriptorSet descriptor_sets[] = {
            m_gpu_resources->taa_read_ds[m_ping_pong]->handle(),
            //m_gpu_resources->rtgi_temporal_read_ds[(uint32_t)m_ping_pong]->handle(),
            m_gpu_resources->svgf_shadow_denoiser->denoised_image_ds()->handle(),
            m_gpu_resources->reflection_blur_read_ds[(uint32_t)m_ping_pong]->handle(),
            m_gpu_resources->svgf_gi_denoiser->denoised_image_ds()->handle()
            //m_gpu_resources->rtgi_temporal_read_ds[(uint32_t)m_ping_pong]->handle()
        };

        ToneMapPushConstants push_constants;

        push_constants.visualization = m_current_visualization;
        push_constants.exposure      = m_exposure;

        vkCmdPushConstants(cmd_buf->handle(), m_gpu_resources->copy_pipeline_layout->handle(), VK_SHADER_STAGE_FRAGMENT_BIT, 0, sizeof(ToneMapPushConstants), &push_constants);

        vkCmdBindDescriptorSets(cmd_buf->handle(), VK_PIPELINE_BIND_POINT_GRAPHICS, m_gpu_resources->copy_pipeline_layout->handle(), 0, 4, descriptor_sets, 0, nullptr);

        vkCmdDraw(cmd_buf->handle(), 3, 1, 0, 0);

        render_gui(cmd_buf);

        vkCmdEndRenderPass(cmd_buf->handle());
    }

    // -----------------------------------------------------------------------------------------------------------------------------------

    void update_uniforms(dw::vk::CommandBuffer::Ptr cmd_buf)
    {
        DW_SCOPED_SAMPLE("Update Uniforms", cmd_buf);

        glm::mat4 current_jitter = glm::translate(glm::mat4(1.0f), glm::vec3(m_current_jitter, 0.0f));
        m_projection             = m_taa_enabled ? current_jitter * m_main_camera->m_projection : m_main_camera->m_projection;

        m_ubo_data.proj_inverse      = glm::inverse(m_projection);
        m_ubo_data.view_inverse      = glm::inverse(m_main_camera->m_view);
        m_ubo_data.view_proj         = m_projection * m_main_camera->m_view;
        m_ubo_data.view_proj_inverse = glm::inverse(m_ubo_data.view_proj);
        m_ubo_data.prev_view_proj    = m_first_frame ? m_main_camera->m_prev_view_projection : current_jitter * m_main_camera->m_prev_view_projection;
        m_ubo_data.cam_pos           = glm::vec4(m_main_camera->m_position, float(m_rtao_enabled));

        set_light_radius(m_ubo_data.light, m_light_radius);
        set_light_direction(m_ubo_data.light, m_light_direction);
        set_light_color(m_ubo_data.light, m_light_color);
        set_light_intensity(m_ubo_data.light, m_light_intensity);

        m_prev_view_proj = m_ubo_data.view_proj;

        uint8_t* ptr = (uint8_t*)m_gpu_resources->ubo->mapped_ptr();
        memcpy(ptr + m_ubo_size * m_vk_backend->current_frame_idx(), &m_ubo_data, sizeof(UBO));
    }

    // -----------------------------------------------------------------------------------------------------------------------------------

    void update_ibl(dw::vk::CommandBuffer::Ptr cmd_buf)
    {
        m_gpu_resources->hosek_wilkie_sky_model->update(cmd_buf, m_light_direction);

        {
            DW_SCOPED_SAMPLE("Generate Skybox Mipmap", cmd_buf);
            m_gpu_resources->hosek_wilkie_sky_model->image()->generate_mipmaps(cmd_buf);
        }

        m_gpu_resources->cubemap_sh_projection->update(cmd_buf);
        m_gpu_resources->cubemap_prefilter->update(cmd_buf);
    }

    // -----------------------------------------------------------------------------------------------------------------------------------

    void update_light_animation()
    {
        if (m_light_animation)
        {
            double time = glfwGetTime() * 0.5f;

            m_light_direction.x = sinf(time);
            m_light_direction.z = cosf(time);
            m_light_direction.y = 1.0f;
            m_light_direction   = glm::normalize(m_light_direction);
        }
    }

    // -----------------------------------------------------------------------------------------------------------------------------------

    void update_camera()
    {
        if (m_taa_enabled)
        {
            m_prev_jitter        = m_current_jitter;
            uint32_t  sample_idx = m_num_frames % (m_jitter_samples.size());
            glm::vec2 halton     = m_jitter_samples[sample_idx];

            m_current_jitter = glm::vec2(halton.x / float(m_width), halton.y / float(m_height));
        }
        else
        {
            m_prev_jitter    = glm::vec2(0.0f);
            m_current_jitter = glm::vec2(0.0f);
        }

        dw::Camera* current = m_main_camera.get();

        float forward_delta = m_heading_speed * m_delta;
        float right_delta   = m_sideways_speed * m_delta;

        current->set_translation_delta(current->m_forward, forward_delta);
        current->set_translation_delta(current->m_right, right_delta);

        m_camera_x = m_mouse_delta_x * m_camera_sensitivity;
        m_camera_y = m_mouse_delta_y * m_camera_sensitivity;

        if (m_mouse_look)
        {
            // Activate Mouse Look
            current->set_rotatation_delta(glm::vec3((float)(m_camera_y),
                                                    (float)(m_camera_x),
                                                    (float)(0.0f)));
        }
        else
        {
            current->set_rotatation_delta(glm::vec3((float)(0),
                                                    (float)(0),
                                                    (float)(0)));
        }

        current->update();
    }

    // -----------------------------------------------------------------------------------------------------------------------------------

    void clear_images(dw::vk::CommandBuffer::Ptr cmd_buf)
    {
        if (m_first_frame)
        {
            VkImageSubresourceRange subresource_range = { VK_IMAGE_ASPECT_COLOR_BIT, 0, 1, 0, 1 };

            dw::vk::utilities::set_image_layout(
                cmd_buf->handle(),
                m_gpu_resources->reflection_history_image[!m_ping_pong]->handle(),
                VK_IMAGE_LAYOUT_UNDEFINED,
                VK_IMAGE_LAYOUT_GENERAL,
                subresource_range);

            dw::vk::utilities::set_image_layout(
                cmd_buf->handle(),
                m_gpu_resources->g_buffer_linear_z[!m_ping_pong]->handle(),
                VK_IMAGE_LAYOUT_UNDEFINED,
                VK_IMAGE_LAYOUT_GENERAL,
                subresource_range);

            VkClearColorValue color;

            color.float32[0] = 0.0f;
            color.float32[1] = 0.0f;
            color.float32[2] = 0.0f;
            color.float32[3] = 0.0f;

            vkCmdClearColorImage(cmd_buf->handle(), m_gpu_resources->g_buffer_linear_z[!m_ping_pong]->handle(), VK_IMAGE_LAYOUT_GENERAL, &color, 1, &subresource_range);
            vkCmdClearColorImage(cmd_buf->handle(), m_gpu_resources->reflection_history_image[!m_ping_pong]->handle(), VK_IMAGE_LAYOUT_GENERAL, &color, 1, &subresource_range);

            dw::vk::utilities::set_image_layout(
                cmd_buf->handle(),
                m_gpu_resources->g_buffer_linear_z[!m_ping_pong]->handle(),
                VK_IMAGE_LAYOUT_GENERAL,
                VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL,
                subresource_range);

            dw::vk::utilities::set_image_layout(
                cmd_buf->handle(),
                m_gpu_resources->reflection_history_image[!m_ping_pong]->handle(),
                VK_IMAGE_LAYOUT_GENERAL,
                VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL,
                subresource_range);

            dw::vk::utilities::set_image_layout(
                cmd_buf->handle(),
                m_gpu_resources->reflection_blur_image[!m_ping_pong]->handle(),
                VK_IMAGE_LAYOUT_UNDEFINED,
                VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL,
                subresource_range);

            dw::vk::utilities::set_image_layout(
                cmd_buf->handle(),
                m_gpu_resources->downsampled_g_buffer_linear_z[!m_ping_pong]->handle(),
                VK_IMAGE_LAYOUT_UNDEFINED,
                VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL,
                subresource_range);
        }
    }

    // -----------------------------------------------------------------------------------------------------------------------------------

    void set_active_scene()
    {
        if (m_current_scene == SCENE_PILLARS)
            m_gpu_resources->current_scene = m_gpu_resources->pillars_scene;
        else if (m_current_scene == SCENE_SPONZA)
            m_gpu_resources->current_scene = m_gpu_resources->sponza_scene;
        else if (m_current_scene == SCENE_PICA_PICA)
            m_gpu_resources->current_scene = m_gpu_resources->pica_pica_scene;
    }

    // -----------------------------------------------------------------------------------------------------------------------------------

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

SVGFDenoiser::SVGFDenoiser(HybridRendering* sample, std::string name, uint32_t input_width, uint32_t input_height, uint32_t filter_iterations) :
    m_name(name), m_sample(sample), m_input_width(input_width), m_input_height(input_height), m_a_trous_filter_iterations(filter_iterations)
{
    create_reprojection_resources();
    create_filter_moments_resources();
    create_a_trous_filter_resources();
}

SVGFDenoiser::~SVGFDenoiser()
{
}

dw::vk::DescriptorSet::Ptr SVGFDenoiser::denoised_image_ds()
{
    return m_a_trous_read_ds[m_read_idx];
}

void SVGFDenoiser::denoise(dw::vk::CommandBuffer::Ptr cmd_buf, dw::vk::DescriptorSet::Ptr input)
{
    clear_images(cmd_buf);
    reprojection(cmd_buf, input);
    filter_moments(cmd_buf);
    a_trous_filter(cmd_buf);
}

void SVGFDenoiser::create_reprojection_resources()
{
    {
        dw::vk::DescriptorSetLayout::Desc desc;

        desc.add_binding(0, VK_DESCRIPTOR_TYPE_STORAGE_IMAGE, 1, VK_SHADER_STAGE_COMPUTE_BIT);
        desc.add_binding(1, VK_DESCRIPTOR_TYPE_STORAGE_IMAGE, 1, VK_SHADER_STAGE_COMPUTE_BIT);
        desc.add_binding(2, VK_DESCRIPTOR_TYPE_STORAGE_IMAGE, 1, VK_SHADER_STAGE_COMPUTE_BIT);

        m_reprojection_write_ds_layout = dw::vk::DescriptorSetLayout::create(m_sample->m_vk_backend, desc);
    }

    {
        dw::vk::DescriptorSetLayout::Desc desc;

        desc.add_binding(0, VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER, 1, VK_SHADER_STAGE_COMPUTE_BIT);
        desc.add_binding(1, VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER, 1, VK_SHADER_STAGE_COMPUTE_BIT);
        desc.add_binding(2, VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER, 1, VK_SHADER_STAGE_COMPUTE_BIT);

        m_reprojection_read_ds_layout = dw::vk::DescriptorSetLayout::create(m_sample->m_vk_backend, desc);
    }

    for (int i = 0; i < 2; i++)
    {
        m_reprojection_write_ds[i] = m_sample->m_vk_backend->allocate_descriptor_set(m_reprojection_write_ds_layout);
        m_reprojection_read_ds[i]  = m_sample->m_vk_backend->allocate_descriptor_set(m_reprojection_read_ds_layout);
    }

    m_prev_reprojection_read_ds = m_sample->m_vk_backend->allocate_descriptor_set(m_sample->m_gpu_resources->combined_sampler_ds_layout);

    {
        dw::vk::PipelineLayout::Desc desc;

        desc.add_descriptor_set_layout(m_reprojection_write_ds_layout);
        desc.add_descriptor_set_layout(m_sample->m_gpu_resources->g_buffer_ds_layout);
        desc.add_descriptor_set_layout(m_sample->m_gpu_resources->g_buffer_ds_layout);
        desc.add_descriptor_set_layout(m_sample->m_gpu_resources->combined_sampler_ds_layout);
        desc.add_descriptor_set_layout(m_sample->m_gpu_resources->combined_sampler_ds_layout);
        desc.add_descriptor_set_layout(m_reprojection_read_ds_layout);

        desc.add_push_constant_range(VK_SHADER_STAGE_COMPUTE_BIT, 0, sizeof(ReprojectionPushConstants));

        m_reprojection_pipeline_layout = dw::vk::PipelineLayout::create(m_sample->m_vk_backend, desc);

        dw::vk::ShaderModule::Ptr module = dw::vk::ShaderModule::create_from_file(m_sample->m_vk_backend, "shaders/svgf_reprojection.comp.spv");

        dw::vk::ComputePipeline::Desc comp_desc;

        comp_desc.set_pipeline_layout(m_reprojection_pipeline_layout);
        comp_desc.set_shader_stage(module, "main");

        m_reprojection_pipeline = dw::vk::ComputePipeline::create(m_sample->m_vk_backend, comp_desc);
    }

    for (int i = 0; i < 2; i++)
    {
        m_reprojection_image[i] = dw::vk::Image::create(m_sample->m_vk_backend, VK_IMAGE_TYPE_2D, m_input_width, m_input_height, 1, 1, 1, VK_FORMAT_R16G16B16A16_SFLOAT, VMA_MEMORY_USAGE_GPU_ONLY, VK_IMAGE_USAGE_TRANSFER_DST_BIT | VK_IMAGE_USAGE_TRANSFER_SRC_BIT | VK_IMAGE_USAGE_SAMPLED_BIT | VK_IMAGE_USAGE_STORAGE_BIT, VK_SAMPLE_COUNT_1_BIT);
        m_reprojection_image[i]->set_name(m_name + " Reprojection Color");

        m_reprojection_view[i] = dw::vk::ImageView::create(m_sample->m_vk_backend, m_reprojection_image[i], VK_IMAGE_VIEW_TYPE_2D, VK_IMAGE_ASPECT_COLOR_BIT);
        m_reprojection_view[i]->set_name(m_name + " Reprojection Color");

        m_moments_image[i] = dw::vk::Image::create(m_sample->m_vk_backend, VK_IMAGE_TYPE_2D, m_input_width, m_input_height, 1, 1, 1, VK_FORMAT_R32G32B32A32_SFLOAT, VMA_MEMORY_USAGE_GPU_ONLY, VK_IMAGE_USAGE_TRANSFER_DST_BIT | VK_IMAGE_USAGE_SAMPLED_BIT | VK_IMAGE_USAGE_STORAGE_BIT, VK_SAMPLE_COUNT_1_BIT);
        m_moments_image[i]->set_name(m_name + " Reprojection Moments");

        m_moments_view[i] = dw::vk::ImageView::create(m_sample->m_vk_backend, m_moments_image[i], VK_IMAGE_VIEW_TYPE_2D, VK_IMAGE_ASPECT_COLOR_BIT);
        m_moments_view[i]->set_name(m_name + " Reprojection Moments");

        m_history_length_image[i] = dw::vk::Image::create(m_sample->m_vk_backend, VK_IMAGE_TYPE_2D, m_input_width, m_input_height, 1, 1, 1, VK_FORMAT_R16_SFLOAT, VMA_MEMORY_USAGE_GPU_ONLY, VK_IMAGE_USAGE_TRANSFER_DST_BIT | VK_IMAGE_USAGE_SAMPLED_BIT | VK_IMAGE_USAGE_STORAGE_BIT, VK_SAMPLE_COUNT_1_BIT);
        m_history_length_image[i]->set_name(m_name + " Reprojection History");

        m_history_length_view[i] = dw::vk::ImageView::create(m_sample->m_vk_backend, m_history_length_image[i], VK_IMAGE_VIEW_TYPE_2D, VK_IMAGE_ASPECT_COLOR_BIT);
        m_history_length_view[i]->set_name(m_name + " Reprojection History");
    }

    m_prev_reprojection_image = dw::vk::Image::create(m_sample->m_vk_backend, VK_IMAGE_TYPE_2D, m_input_width, m_input_height, 1, 1, 1, VK_FORMAT_R16G16B16A16_SFLOAT, VMA_MEMORY_USAGE_GPU_ONLY, VK_IMAGE_USAGE_TRANSFER_DST_BIT | VK_IMAGE_USAGE_SAMPLED_BIT | VK_IMAGE_USAGE_STORAGE_BIT, VK_SAMPLE_COUNT_1_BIT);
    m_prev_reprojection_image->set_name(m_name + " Previous Reprojection");

    m_prev_reprojection_view = dw::vk::ImageView::create(m_sample->m_vk_backend, m_prev_reprojection_image, VK_IMAGE_VIEW_TYPE_2D, VK_IMAGE_ASPECT_COLOR_BIT);
    m_prev_reprojection_view->set_name(m_name + " Previous Reprojection");

    // Reprojection write
    {
        std::vector<VkDescriptorImageInfo> image_infos;
        std::vector<VkWriteDescriptorSet>  write_datas;
        VkWriteDescriptorSet               write_data;

        image_infos.reserve(6);
        write_datas.reserve(6);

        for (int i = 0; i < 2; i++)
        {
            {
                VkDescriptorImageInfo storage_image_info;

                storage_image_info.sampler     = VK_NULL_HANDLE;
                storage_image_info.imageView   = m_reprojection_view[i]->handle();
                storage_image_info.imageLayout = VK_IMAGE_LAYOUT_GENERAL;

                image_infos.push_back(storage_image_info);

                DW_ZERO_MEMORY(write_data);

                write_data.sType           = VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET;
                write_data.descriptorCount = 1;
                write_data.descriptorType  = VK_DESCRIPTOR_TYPE_STORAGE_IMAGE;
                write_data.pImageInfo      = &image_infos.back();
                write_data.dstBinding      = 0;
                write_data.dstSet          = m_reprojection_write_ds[i]->handle();

                write_datas.push_back(write_data);
            }

            {
                VkDescriptorImageInfo storage_image_info;

                storage_image_info.sampler     = VK_NULL_HANDLE;
                storage_image_info.imageView   = m_moments_view[i]->handle();
                storage_image_info.imageLayout = VK_IMAGE_LAYOUT_GENERAL;

                image_infos.push_back(storage_image_info);

                DW_ZERO_MEMORY(write_data);

                write_data.sType           = VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET;
                write_data.descriptorCount = 1;
                write_data.descriptorType  = VK_DESCRIPTOR_TYPE_STORAGE_IMAGE;
                write_data.pImageInfo      = &image_infos.back();
                write_data.dstBinding      = 1;
                write_data.dstSet          = m_reprojection_write_ds[i]->handle();

                write_datas.push_back(write_data);
            }

            {
                VkDescriptorImageInfo storage_image_info;

                storage_image_info.sampler     = VK_NULL_HANDLE;
                storage_image_info.imageView   = m_history_length_view[i]->handle();
                storage_image_info.imageLayout = VK_IMAGE_LAYOUT_GENERAL;

                image_infos.push_back(storage_image_info);

                DW_ZERO_MEMORY(write_data);

                write_data.sType           = VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET;
                write_data.descriptorCount = 1;
                write_data.descriptorType  = VK_DESCRIPTOR_TYPE_STORAGE_IMAGE;
                write_data.pImageInfo      = &image_infos.back();
                write_data.dstBinding      = 2;
                write_data.dstSet          = m_reprojection_write_ds[i]->handle();

                write_datas.push_back(write_data);
            }
        }

        vkUpdateDescriptorSets(m_sample->m_vk_backend->device(), write_datas.size(), write_datas.data(), 0, nullptr);
    }

    // Reprojection read
    {
        std::vector<VkDescriptorImageInfo> image_infos;
        std::vector<VkWriteDescriptorSet>  write_datas;
        VkWriteDescriptorSet               write_data;

        image_infos.reserve(6);
        write_datas.reserve(6);

        for (int i = 0; i < 2; i++)
        {
            {
                VkDescriptorImageInfo sampler_image_info;

                sampler_image_info.sampler     = m_sample->m_vk_backend->nearest_sampler()->handle();
                sampler_image_info.imageView   = m_reprojection_view[i]->handle();
                sampler_image_info.imageLayout = VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL;

                image_infos.push_back(sampler_image_info);

                DW_ZERO_MEMORY(write_data);

                write_data.sType           = VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET;
                write_data.descriptorCount = 1;
                write_data.descriptorType  = VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER;
                write_data.pImageInfo      = &image_infos.back();
                write_data.dstBinding      = 0;
                write_data.dstSet          = m_reprojection_read_ds[i]->handle();

                write_datas.push_back(write_data);
            }

            {
                VkDescriptorImageInfo sampler_image_info;

                sampler_image_info.sampler     = m_sample->m_vk_backend->nearest_sampler()->handle();
                sampler_image_info.imageView   = m_moments_view[i]->handle();
                sampler_image_info.imageLayout = VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL;

                image_infos.push_back(sampler_image_info);

                DW_ZERO_MEMORY(write_data);

                write_data.sType           = VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET;
                write_data.descriptorCount = 1;
                write_data.descriptorType  = VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER;
                write_data.pImageInfo      = &image_infos.back();
                write_data.dstBinding      = 1;
                write_data.dstSet          = m_reprojection_read_ds[i]->handle();

                write_datas.push_back(write_data);
            }

            {
                VkDescriptorImageInfo sampler_image_info;

                sampler_image_info.sampler     = m_sample->m_vk_backend->nearest_sampler()->handle();
                sampler_image_info.imageView   = m_history_length_view[i]->handle();
                sampler_image_info.imageLayout = VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL;

                image_infos.push_back(sampler_image_info);

                DW_ZERO_MEMORY(write_data);

                write_data.sType           = VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET;
                write_data.descriptorCount = 1;
                write_data.descriptorType  = VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER;
                write_data.pImageInfo      = &image_infos.back();
                write_data.dstBinding      = 2;
                write_data.dstSet          = m_reprojection_read_ds[i]->handle();

                write_datas.push_back(write_data);
            }
        }

        vkUpdateDescriptorSets(m_sample->m_vk_backend->device(), write_datas.size(), write_datas.data(), 0, nullptr);
    }

    // Previous Reprojection read
    {
        std::vector<VkDescriptorImageInfo> image_infos;
        std::vector<VkWriteDescriptorSet>  write_datas;
        VkWriteDescriptorSet               write_data;

        image_infos.reserve(1);
        write_datas.reserve(1);

        VkDescriptorImageInfo sampler_image_info;

        sampler_image_info.sampler     = m_sample->m_vk_backend->nearest_sampler()->handle();
        sampler_image_info.imageView   = m_prev_reprojection_view->handle();
        sampler_image_info.imageLayout = VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL;

        image_infos.push_back(sampler_image_info);

        DW_ZERO_MEMORY(write_data);

        write_data.sType           = VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET;
        write_data.descriptorCount = 1;
        write_data.descriptorType  = VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER;
        write_data.pImageInfo      = &image_infos.back();
        write_data.dstBinding      = 0;
        write_data.dstSet          = m_prev_reprojection_read_ds->handle();

        write_datas.push_back(write_data);

        vkUpdateDescriptorSets(m_sample->m_vk_backend->device(), write_datas.size(), write_datas.data(), 0, nullptr);
    }
}

void SVGFDenoiser::create_filter_moments_resources()
{
    m_filter_moments_write_ds = m_sample->m_vk_backend->allocate_descriptor_set(m_sample->m_gpu_resources->storage_image_ds_layout);
    m_filter_moments_read_ds  = m_sample->m_vk_backend->allocate_descriptor_set(m_sample->m_gpu_resources->combined_sampler_ds_layout);

    dw::vk::PipelineLayout::Desc desc;

    desc.add_descriptor_set_layout(m_sample->m_gpu_resources->storage_image_ds_layout);
    desc.add_descriptor_set_layout(m_sample->m_gpu_resources->g_buffer_ds_layout);
    desc.add_descriptor_set_layout(m_reprojection_read_ds_layout);

    desc.add_push_constant_range(VK_SHADER_STAGE_COMPUTE_BIT, 0, sizeof(FilterMomentsPushConstants));

    m_filter_moments_pipeline_layout = dw::vk::PipelineLayout::create(m_sample->m_vk_backend, desc);

    dw::vk::ShaderModule::Ptr module = dw::vk::ShaderModule::create_from_file(m_sample->m_vk_backend, "shaders/svgf_filter_moments.comp.spv");

    dw::vk::ComputePipeline::Desc comp_desc;

    comp_desc.set_pipeline_layout(m_filter_moments_pipeline_layout);
    comp_desc.set_shader_stage(module, "main");

    m_filter_moments_pipeline = dw::vk::ComputePipeline::create(m_sample->m_vk_backend, comp_desc);

    m_filter_moments_image = dw::vk::Image::create(m_sample->m_vk_backend, VK_IMAGE_TYPE_2D, m_input_width, m_input_height, 1, 1, 1, VK_FORMAT_R16G16B16A16_SFLOAT, VMA_MEMORY_USAGE_GPU_ONLY, VK_IMAGE_USAGE_TRANSFER_DST_BIT | VK_IMAGE_USAGE_SAMPLED_BIT | VK_IMAGE_USAGE_STORAGE_BIT, VK_SAMPLE_COUNT_1_BIT);
    m_filter_moments_image->set_name(m_name + " Filter Moments");

    m_filter_moments_view = dw::vk::ImageView::create(m_sample->m_vk_backend, m_filter_moments_image, VK_IMAGE_VIEW_TYPE_2D, VK_IMAGE_ASPECT_COLOR_BIT);
    m_filter_moments_view->set_name(m_name + " Filter Moments");

    // Filter Moments write
    {
        std::vector<VkDescriptorImageInfo> image_infos;
        std::vector<VkWriteDescriptorSet>  write_datas;
        VkWriteDescriptorSet               write_data;

        image_infos.reserve(1);
        write_datas.reserve(1);

        VkDescriptorImageInfo storage_image_info;

        storage_image_info.sampler     = VK_NULL_HANDLE;
        storage_image_info.imageView   = m_filter_moments_view->handle();
        storage_image_info.imageLayout = VK_IMAGE_LAYOUT_GENERAL;

        image_infos.push_back(storage_image_info);

        DW_ZERO_MEMORY(write_data);

        write_data.sType           = VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET;
        write_data.descriptorCount = 1;
        write_data.descriptorType  = VK_DESCRIPTOR_TYPE_STORAGE_IMAGE;
        write_data.pImageInfo      = &image_infos.back();
        write_data.dstBinding      = 0;
        write_data.dstSet          = m_filter_moments_write_ds->handle();

        write_datas.push_back(write_data);

        vkUpdateDescriptorSets(m_sample->m_vk_backend->device(), write_datas.size(), write_datas.data(), 0, nullptr);
    }

    // Filter Moments read
    {
        std::vector<VkDescriptorImageInfo> image_infos;
        std::vector<VkWriteDescriptorSet>  write_datas;
        VkWriteDescriptorSet               write_data;

        image_infos.reserve(1);
        write_datas.reserve(1);

        VkDescriptorImageInfo sampler_image_info;

        sampler_image_info.sampler     = m_sample->m_vk_backend->nearest_sampler()->handle();
        sampler_image_info.imageView   = m_filter_moments_view->handle();
        sampler_image_info.imageLayout = VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL;

        image_infos.push_back(sampler_image_info);

        DW_ZERO_MEMORY(write_data);

        write_data.sType           = VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET;
        write_data.descriptorCount = 1;
        write_data.descriptorType  = VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER;
        write_data.pImageInfo      = &image_infos.back();
        write_data.dstBinding      = 0;
        write_data.dstSet          = m_filter_moments_read_ds->handle();

        write_datas.push_back(write_data);

        vkUpdateDescriptorSets(m_sample->m_vk_backend->device(), write_datas.size(), write_datas.data(), 0, nullptr);
    }
}

void SVGFDenoiser::create_a_trous_filter_resources()
{
    for (int i = 0; i < 2; i++)
    {
        m_a_trous_read_ds[i]  = m_sample->m_vk_backend->allocate_descriptor_set(m_sample->m_gpu_resources->combined_sampler_ds_layout);
        m_a_trous_write_ds[i] = m_sample->m_vk_backend->allocate_descriptor_set(m_sample->m_gpu_resources->storage_image_ds_layout);
    }

    dw::vk::PipelineLayout::Desc desc;

    desc.add_descriptor_set_layout(m_sample->m_gpu_resources->storage_image_ds_layout);
    desc.add_descriptor_set_layout(m_sample->m_gpu_resources->combined_sampler_ds_layout);
    desc.add_descriptor_set_layout(m_sample->m_gpu_resources->g_buffer_ds_layout);
    desc.add_descriptor_set_layout(m_reprojection_read_ds_layout);

    desc.add_push_constant_range(VK_SHADER_STAGE_COMPUTE_BIT, 0, sizeof(ATrousFilterPushConstants));

    m_a_trous_filter_pipeline_layout = dw::vk::PipelineLayout::create(m_sample->m_vk_backend, desc);

    dw::vk::ShaderModule::Ptr module = dw::vk::ShaderModule::create_from_file(m_sample->m_vk_backend, "shaders/svgf_a_trous_filter.comp.spv");

    dw::vk::ComputePipeline::Desc comp_desc;

    comp_desc.set_pipeline_layout(m_a_trous_filter_pipeline_layout);
    comp_desc.set_shader_stage(module, "main");

    m_a_trous_filter_pipeline = dw::vk::ComputePipeline::create(m_sample->m_vk_backend, comp_desc);

    for (int i = 0; i < 2; i++)
    {
        m_a_trous_image[i] = dw::vk::Image::create(m_sample->m_vk_backend, VK_IMAGE_TYPE_2D, m_input_width, m_input_height, 1, 1, 1, VK_FORMAT_R16G16B16A16_SFLOAT, VMA_MEMORY_USAGE_GPU_ONLY, VK_IMAGE_USAGE_TRANSFER_DST_BIT | VK_IMAGE_USAGE_SAMPLED_BIT | VK_IMAGE_USAGE_STORAGE_BIT, VK_SAMPLE_COUNT_1_BIT);
        m_a_trous_image[i]->set_name(m_name + " A-Trous Filter");

        m_a_trous_view[i] = dw::vk::ImageView::create(m_sample->m_vk_backend, m_a_trous_image[i], VK_IMAGE_VIEW_TYPE_2D, VK_IMAGE_ASPECT_COLOR_BIT);
        m_a_trous_view[i]->set_name(m_name + " A-Trous Filter");
    }

    // A-Trous write
    {
        std::vector<VkDescriptorImageInfo> image_infos;
        std::vector<VkWriteDescriptorSet>  write_datas;
        VkWriteDescriptorSet               write_data;

        image_infos.reserve(2);
        write_datas.reserve(2);

        for (int i = 0; i < 2; i++)
        {
            VkDescriptorImageInfo storage_image_info;

            storage_image_info.sampler     = VK_NULL_HANDLE;
            storage_image_info.imageView   = m_a_trous_view[i]->handle();
            storage_image_info.imageLayout = VK_IMAGE_LAYOUT_GENERAL;

            image_infos.push_back(storage_image_info);

            DW_ZERO_MEMORY(write_data);

            write_data.sType           = VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET;
            write_data.descriptorCount = 1;
            write_data.descriptorType  = VK_DESCRIPTOR_TYPE_STORAGE_IMAGE;
            write_data.pImageInfo      = &image_infos.back();
            write_data.dstBinding      = 0;
            write_data.dstSet          = m_a_trous_write_ds[i]->handle();

            write_datas.push_back(write_data);
        }

        vkUpdateDescriptorSets(m_sample->m_vk_backend->device(), write_datas.size(), write_datas.data(), 0, nullptr);
    }

    // A-Trous read
    {
        std::vector<VkDescriptorImageInfo> image_infos;
        std::vector<VkWriteDescriptorSet>  write_datas;
        VkWriteDescriptorSet               write_data;

        image_infos.reserve(2);
        write_datas.reserve(2);

        for (int i = 0; i < 2; i++)
        {
            VkDescriptorImageInfo sampler_image_info;

            sampler_image_info.sampler     = m_sample->m_vk_backend->nearest_sampler()->handle();
            sampler_image_info.imageView   = m_a_trous_view[i]->handle();
            sampler_image_info.imageLayout = VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL;

            image_infos.push_back(sampler_image_info);

            DW_ZERO_MEMORY(write_data);

            write_data.sType           = VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET;
            write_data.descriptorCount = 1;
            write_data.descriptorType  = VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER;
            write_data.pImageInfo      = &image_infos.back();
            write_data.dstBinding      = 0;
            write_data.dstSet          = m_a_trous_read_ds[i]->handle();

            write_datas.push_back(write_data);
        }

        vkUpdateDescriptorSets(m_sample->m_vk_backend->device(), write_datas.size(), write_datas.data(), 0, nullptr);
    }
}

void SVGFDenoiser::clear_images(dw::vk::CommandBuffer::Ptr cmd_buf)
{
    if (m_sample->m_first_frame)
    {
        VkImageSubresourceRange subresource_range = { VK_IMAGE_ASPECT_COLOR_BIT, 0, 1, 0, 1 };

        VkClearColorValue color;

        color.float32[0] = 0.0f;
        color.float32[1] = 0.0f;
        color.float32[2] = 0.0f;
        color.float32[3] = 0.0f;

        dw::vk::utilities::set_image_layout(
            cmd_buf->handle(),
            m_prev_reprojection_image->handle(),
            VK_IMAGE_LAYOUT_UNDEFINED,
            VK_IMAGE_LAYOUT_GENERAL,
            subresource_range);

        dw::vk::utilities::set_image_layout(
            cmd_buf->handle(),
            m_moments_image[!m_sample->m_ping_pong]->handle(),
            VK_IMAGE_LAYOUT_UNDEFINED,
            VK_IMAGE_LAYOUT_GENERAL,
            subresource_range);

        dw::vk::utilities::set_image_layout(
            cmd_buf->handle(),
            m_history_length_image[!m_sample->m_ping_pong]->handle(),
            VK_IMAGE_LAYOUT_UNDEFINED,
            VK_IMAGE_LAYOUT_GENERAL,
            subresource_range);

        dw::vk::utilities::set_image_layout(
            cmd_buf->handle(),
            m_reprojection_image[!m_sample->m_ping_pong]->handle(),
            VK_IMAGE_LAYOUT_UNDEFINED,
            VK_IMAGE_LAYOUT_GENERAL,
            subresource_range);

        vkCmdClearColorImage(cmd_buf->handle(), m_moments_image[!m_sample->m_ping_pong]->handle(), VK_IMAGE_LAYOUT_GENERAL, &color, 1, &subresource_range);
        vkCmdClearColorImage(cmd_buf->handle(), m_history_length_image[!m_sample->m_ping_pong]->handle(), VK_IMAGE_LAYOUT_GENERAL, &color, 1, &subresource_range);
        vkCmdClearColorImage(cmd_buf->handle(), m_reprojection_image[!m_sample->m_ping_pong]->handle(), VK_IMAGE_LAYOUT_GENERAL, &color, 1, &subresource_range);
        vkCmdClearColorImage(cmd_buf->handle(), m_prev_reprojection_image->handle(), VK_IMAGE_LAYOUT_GENERAL, &color, 1, &subresource_range);

        dw::vk::utilities::set_image_layout(
            cmd_buf->handle(),
            m_prev_reprojection_image->handle(),
            VK_IMAGE_LAYOUT_GENERAL,
            VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL,
            subresource_range);

        dw::vk::utilities::set_image_layout(
            cmd_buf->handle(),
            m_moments_image[!m_sample->m_ping_pong]->handle(),
            VK_IMAGE_LAYOUT_GENERAL,
            VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL,
            subresource_range);

        dw::vk::utilities::set_image_layout(
            cmd_buf->handle(),
            m_history_length_image[!m_sample->m_ping_pong]->handle(),
            VK_IMAGE_LAYOUT_GENERAL,
            VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL,
            subresource_range);

        dw::vk::utilities::set_image_layout(
            cmd_buf->handle(),
            m_reprojection_image[!m_sample->m_ping_pong]->handle(),
            VK_IMAGE_LAYOUT_GENERAL,
            VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL,
            subresource_range);
    }
}

void SVGFDenoiser::reprojection(dw::vk::CommandBuffer::Ptr cmd_buf, dw::vk::DescriptorSet::Ptr input)
{
    DW_SCOPED_SAMPLE(m_name + " SVGF Temporal Reprojection", cmd_buf);

    VkImageSubresourceRange subresource_range = { VK_IMAGE_ASPECT_COLOR_BIT, 0, 1, 0, 1 };

    {
        std::vector<VkMemoryBarrier> memory_barriers = {
            memory_barrier(VK_ACCESS_SHADER_READ_BIT, VK_ACCESS_SHADER_WRITE_BIT)
        };

        std::vector<VkImageMemoryBarrier> image_barriers = {
            image_memory_barrier(m_reprojection_image[m_sample->m_ping_pong], VK_IMAGE_LAYOUT_UNDEFINED, VK_IMAGE_LAYOUT_GENERAL, subresource_range, VK_ACCESS_SHADER_READ_BIT, VK_ACCESS_SHADER_WRITE_BIT),
            image_memory_barrier(m_moments_image[m_sample->m_ping_pong], VK_IMAGE_LAYOUT_UNDEFINED, VK_IMAGE_LAYOUT_GENERAL, subresource_range, VK_ACCESS_SHADER_READ_BIT, VK_ACCESS_SHADER_WRITE_BIT),
            image_memory_barrier(m_history_length_image[m_sample->m_ping_pong], VK_IMAGE_LAYOUT_UNDEFINED, VK_IMAGE_LAYOUT_GENERAL, subresource_range, VK_ACCESS_SHADER_READ_BIT, VK_ACCESS_SHADER_WRITE_BIT)
        };

        pipeline_barrier(cmd_buf, memory_barriers, image_barriers, VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT, VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT);
    }

    const uint32_t NUM_THREADS = 32;

    vkCmdBindPipeline(cmd_buf->handle(), VK_PIPELINE_BIND_POINT_COMPUTE, m_reprojection_pipeline->handle());

    ReprojectionPushConstants push_constants;

    push_constants.alpha         = m_alpha;
    push_constants.moments_alpha = m_moments_alpha;

    vkCmdPushConstants(cmd_buf->handle(), m_reprojection_pipeline_layout->handle(), VK_SHADER_STAGE_COMPUTE_BIT, 0, sizeof(push_constants), &push_constants);

    VkDescriptorSet descriptor_sets[] = {
        m_reprojection_write_ds[m_sample->m_ping_pong]->handle(),
        m_sample->m_gpu_resources->downsampled_g_buffer_ds[m_sample->m_ping_pong]->handle(),
        m_sample->m_gpu_resources->downsampled_g_buffer_ds[!m_sample->m_ping_pong]->handle(),
        input->handle(),
        m_prev_reprojection_read_ds->handle(),
        m_reprojection_read_ds[!m_sample->m_ping_pong]->handle()
    };

    vkCmdBindDescriptorSets(cmd_buf->handle(), VK_PIPELINE_BIND_POINT_COMPUTE, m_reprojection_pipeline_layout->handle(), 0, 6, descriptor_sets, 0, nullptr);

    vkCmdDispatch(cmd_buf->handle(), static_cast<uint32_t>(ceil(float(m_input_width) / float(NUM_THREADS))), static_cast<uint32_t>(ceil(float(m_input_height) / float(NUM_THREADS))), 1);

    if (!m_use_spatial_for_feedback)
    {
        dw::vk::utilities::set_image_layout(
            cmd_buf->handle(),
            m_reprojection_image[m_sample->m_ping_pong]->handle(),
            VK_IMAGE_LAYOUT_GENERAL,
            VK_IMAGE_LAYOUT_TRANSFER_SRC_OPTIMAL,
            subresource_range);

        dw::vk::utilities::set_image_layout(
            cmd_buf->handle(),
            m_prev_reprojection_image->handle(),
            VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL,
            VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL,
            subresource_range);

        VkImageCopy image_copy_region {};
        image_copy_region.srcSubresource.aspectMask = VK_IMAGE_ASPECT_COLOR_BIT;
        image_copy_region.srcSubresource.layerCount = 1;
        image_copy_region.dstSubresource.aspectMask = VK_IMAGE_ASPECT_COLOR_BIT;
        image_copy_region.dstSubresource.layerCount = 1;
        image_copy_region.extent.width              = m_input_width;
        image_copy_region.extent.height             = m_input_height;
        image_copy_region.extent.depth              = 1;

        // Issue the copy command
        vkCmdCopyImage(
            cmd_buf->handle(),
            m_reprojection_image[m_sample->m_ping_pong]->handle(),
            VK_IMAGE_LAYOUT_TRANSFER_SRC_OPTIMAL,
            m_prev_reprojection_image->handle(),
            VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL,
            1,
            &image_copy_region);

        dw::vk::utilities::set_image_layout(
            cmd_buf->handle(),
            m_reprojection_image[m_sample->m_ping_pong]->handle(),
            VK_IMAGE_LAYOUT_TRANSFER_SRC_OPTIMAL,
            VK_IMAGE_LAYOUT_GENERAL,
            subresource_range);

        dw::vk::utilities::set_image_layout(
            cmd_buf->handle(),
            m_prev_reprojection_image->handle(),
            VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL,
            VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL,
            subresource_range);
    }

    {
        std::vector<VkMemoryBarrier> memory_barriers = {
            memory_barrier(VK_ACCESS_SHADER_WRITE_BIT, VK_ACCESS_SHADER_READ_BIT)
        };

        std::vector<VkImageMemoryBarrier> image_barriers = {
            image_memory_barrier(m_reprojection_image[m_sample->m_ping_pong], VK_IMAGE_LAYOUT_GENERAL, VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL, subresource_range, VK_ACCESS_SHADER_WRITE_BIT, VK_ACCESS_SHADER_READ_BIT),
            image_memory_barrier(m_moments_image[m_sample->m_ping_pong], VK_IMAGE_LAYOUT_GENERAL, VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL, subresource_range, VK_ACCESS_SHADER_WRITE_BIT, VK_ACCESS_SHADER_READ_BIT),
            image_memory_barrier(m_history_length_image[m_sample->m_ping_pong], VK_IMAGE_LAYOUT_GENERAL, VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL, subresource_range, VK_ACCESS_SHADER_WRITE_BIT, VK_ACCESS_SHADER_READ_BIT)
        };

        pipeline_barrier(cmd_buf, memory_barriers, image_barriers, VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT, VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT);
    }
}

void SVGFDenoiser::filter_moments(dw::vk::CommandBuffer::Ptr cmd_buf)
{
    DW_SCOPED_SAMPLE(m_name + " SVGF Filter Moments", cmd_buf);

    VkImageSubresourceRange subresource_range = { VK_IMAGE_ASPECT_COLOR_BIT, 0, 1, 0, 1 };

    {
        std::vector<VkMemoryBarrier> memory_barriers = {
            memory_barrier(VK_ACCESS_SHADER_READ_BIT, VK_ACCESS_SHADER_WRITE_BIT)
        };

        std::vector<VkImageMemoryBarrier> image_barriers = {
            image_memory_barrier(m_filter_moments_image, VK_IMAGE_LAYOUT_UNDEFINED, VK_IMAGE_LAYOUT_GENERAL, subresource_range, VK_ACCESS_SHADER_READ_BIT, VK_ACCESS_SHADER_WRITE_BIT),
        };

        pipeline_barrier(cmd_buf, memory_barriers, image_barriers, VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT, VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT);
    }

    const uint32_t NUM_THREADS = 32;

    vkCmdBindPipeline(cmd_buf->handle(), VK_PIPELINE_BIND_POINT_COMPUTE, m_filter_moments_pipeline->handle());

    FilterMomentsPushConstants push_constants;

    push_constants.phi_color  = m_phi_color;
    push_constants.phi_normal = m_phi_normal;

    vkCmdPushConstants(cmd_buf->handle(), m_filter_moments_pipeline_layout->handle(), VK_SHADER_STAGE_COMPUTE_BIT, 0, sizeof(push_constants), &push_constants);

    VkDescriptorSet descriptor_sets[] = {
        m_filter_moments_write_ds->handle(),
        m_sample->m_gpu_resources->downsampled_g_buffer_ds[m_sample->m_ping_pong]->handle(),
        m_reprojection_read_ds[m_sample->m_ping_pong]->handle()
    };

    vkCmdBindDescriptorSets(cmd_buf->handle(), VK_PIPELINE_BIND_POINT_COMPUTE, m_filter_moments_pipeline_layout->handle(), 0, 3, descriptor_sets, 0, nullptr);

    vkCmdDispatch(cmd_buf->handle(), static_cast<uint32_t>(ceil(float(m_input_width) / float(NUM_THREADS))), static_cast<uint32_t>(ceil(float(m_input_height) / float(NUM_THREADS))), 1);

    {
        std::vector<VkMemoryBarrier> memory_barriers = {
            memory_barrier(VK_ACCESS_SHADER_WRITE_BIT, VK_ACCESS_SHADER_READ_BIT)
        };

        std::vector<VkImageMemoryBarrier> image_barriers = {
            image_memory_barrier(m_filter_moments_image, VK_IMAGE_LAYOUT_GENERAL, VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL, subresource_range, VK_ACCESS_SHADER_WRITE_BIT, VK_ACCESS_SHADER_READ_BIT),
        };

        pipeline_barrier(cmd_buf, memory_barriers, image_barriers, VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT, VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT);
    }
}

void SVGFDenoiser::a_trous_filter(dw::vk::CommandBuffer::Ptr cmd_buf)
{
    DW_SCOPED_SAMPLE(m_name + " SVGF A-Trous Filter", cmd_buf);

    const uint32_t NUM_THREADS = 32;

    VkImageSubresourceRange subresource_range = { VK_IMAGE_ASPECT_COLOR_BIT, 0, 1, 0, 1 };

    vkCmdBindPipeline(cmd_buf->handle(), VK_PIPELINE_BIND_POINT_COMPUTE, m_a_trous_filter_pipeline->handle());

    bool    ping_pong = false;
    int32_t read_idx  = 0;
    int32_t write_idx = 1;

    for (int i = 0; i < m_a_trous_filter_iterations; i++)
    {
        read_idx  = (int32_t)ping_pong;
        write_idx = (int32_t)!ping_pong;

        if (i == 0)
        {
            std::vector<VkMemoryBarrier> memory_barriers = {
                memory_barrier(VK_ACCESS_SHADER_WRITE_BIT, VK_ACCESS_SHADER_READ_BIT)
            };

            std::vector<VkImageMemoryBarrier> image_barriers = {
                image_memory_barrier(m_a_trous_image[write_idx], VK_IMAGE_LAYOUT_UNDEFINED, VK_IMAGE_LAYOUT_GENERAL, subresource_range, VK_ACCESS_SHADER_READ_BIT, VK_ACCESS_SHADER_WRITE_BIT)
            };

            pipeline_barrier(cmd_buf, memory_barriers, image_barriers, VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT, VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT);
        }
        else
        {
            std::vector<VkMemoryBarrier> memory_barriers = {
                memory_barrier(VK_ACCESS_SHADER_WRITE_BIT, VK_ACCESS_SHADER_READ_BIT)
            };

            std::vector<VkImageMemoryBarrier> image_barriers = {
                image_memory_barrier(m_a_trous_image[read_idx], VK_IMAGE_LAYOUT_GENERAL, VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL, subresource_range, VK_ACCESS_SHADER_WRITE_BIT, VK_ACCESS_SHADER_READ_BIT),
                image_memory_barrier(m_a_trous_image[write_idx], VK_IMAGE_LAYOUT_UNDEFINED, VK_IMAGE_LAYOUT_GENERAL, subresource_range, VK_ACCESS_SHADER_READ_BIT, VK_ACCESS_SHADER_WRITE_BIT)
            };

            pipeline_barrier(cmd_buf, memory_barriers, image_barriers, VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT, VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT);
        }

        ATrousFilterPushConstants push_constants;

        push_constants.radius     = m_a_trous_radius;
        push_constants.step_size  = 1 << i;
        push_constants.phi_color  = m_phi_color;
        push_constants.phi_normal = m_phi_normal;

        vkCmdPushConstants(cmd_buf->handle(), m_a_trous_filter_pipeline_layout->handle(), VK_SHADER_STAGE_COMPUTE_BIT, 0, sizeof(push_constants), &push_constants);

        VkDescriptorSet descriptor_sets[] = {
            m_a_trous_write_ds[write_idx]->handle(),
            i == 0 ? m_filter_moments_read_ds->handle() : m_a_trous_read_ds[read_idx]->handle(),
            m_sample->m_gpu_resources->downsampled_g_buffer_ds[m_sample->m_ping_pong]->handle(),
            m_reprojection_read_ds[!m_sample->m_ping_pong]->handle()
        };

        vkCmdBindDescriptorSets(cmd_buf->handle(), VK_PIPELINE_BIND_POINT_COMPUTE, m_a_trous_filter_pipeline_layout->handle(), 0, 4, descriptor_sets, 0, nullptr);

        vkCmdDispatch(cmd_buf->handle(), static_cast<uint32_t>(ceil(float(m_input_width) / float(NUM_THREADS))), static_cast<uint32_t>(ceil(float(m_input_height) / float(NUM_THREADS))), 1);

        ping_pong = !ping_pong;

        if (m_use_spatial_for_feedback && m_a_trous_feedback_iteration == i)
        {
            dw::vk::utilities::set_image_layout(
                cmd_buf->handle(),
                m_a_trous_image[write_idx]->handle(),
                VK_IMAGE_LAYOUT_GENERAL,
                VK_IMAGE_LAYOUT_TRANSFER_SRC_OPTIMAL,
                subresource_range);

            dw::vk::utilities::set_image_layout(
                cmd_buf->handle(),
                m_prev_reprojection_image->handle(),
                VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL,
                VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL,
                subresource_range);

            VkImageCopy image_copy_region {};
            image_copy_region.srcSubresource.aspectMask = VK_IMAGE_ASPECT_COLOR_BIT;
            image_copy_region.srcSubresource.layerCount = 1;
            image_copy_region.dstSubresource.aspectMask = VK_IMAGE_ASPECT_COLOR_BIT;
            image_copy_region.dstSubresource.layerCount = 1;
            image_copy_region.extent.width              = m_input_width;
            image_copy_region.extent.height             = m_input_height;
            image_copy_region.extent.depth              = 1;

            // Issue the copy command
            vkCmdCopyImage(
                cmd_buf->handle(),
                m_a_trous_image[write_idx]->handle(),
                VK_IMAGE_LAYOUT_TRANSFER_SRC_OPTIMAL,
                m_prev_reprojection_image->handle(),
                VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL,
                1,
                &image_copy_region);

            dw::vk::utilities::set_image_layout(
                cmd_buf->handle(),
                m_a_trous_image[write_idx]->handle(),
                VK_IMAGE_LAYOUT_TRANSFER_SRC_OPTIMAL,
                VK_IMAGE_LAYOUT_GENERAL,
                subresource_range);

            dw::vk::utilities::set_image_layout(
                cmd_buf->handle(),
                m_prev_reprojection_image->handle(),
                VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL,
                VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL,
                subresource_range);
        }
    }

    m_read_idx = write_idx;

    std::vector<VkMemoryBarrier> memory_barriers = {
        memory_barrier(VK_ACCESS_SHADER_WRITE_BIT, VK_ACCESS_SHADER_READ_BIT)
    };

    std::vector<VkImageMemoryBarrier> image_barriers = {
        image_memory_barrier(m_a_trous_image[write_idx], VK_IMAGE_LAYOUT_GENERAL, VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL, subresource_range, VK_ACCESS_SHADER_WRITE_BIT, VK_ACCESS_SHADER_READ_BIT)
    };

    pipeline_barrier(cmd_buf, memory_barriers, image_barriers, VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT, VK_PIPELINE_STAGE_FRAGMENT_SHADER_BIT);
}

TemporalDenoiser::TemporalDenoiser(HybridRendering* sample, std::string name, uint32_t input_width, uint32_t input_height) :
    m_name(name), m_sample(sample), m_input_width(input_width), m_input_height(input_height)
{
    {
        dw::vk::DescriptorSetLayout::Desc desc;

        desc.add_binding(0, VK_DESCRIPTOR_TYPE_STORAGE_IMAGE, 1, VK_SHADER_STAGE_COMPUTE_BIT);
        desc.add_binding(1, VK_DESCRIPTOR_TYPE_STORAGE_IMAGE, 1, VK_SHADER_STAGE_COMPUTE_BIT);

        m_write_ds_layout = dw::vk::DescriptorSetLayout::create(m_sample->m_vk_backend, desc);
    }

    {
        dw::vk::DescriptorSetLayout::Desc desc;

        desc.add_binding(0, VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER, 1, VK_SHADER_STAGE_COMPUTE_BIT);
        desc.add_binding(1, VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER, 1, VK_SHADER_STAGE_COMPUTE_BIT);

        m_read_ds_layout = dw::vk::DescriptorSetLayout::create(m_sample->m_vk_backend, desc);
    }

    for (int i = 0; i < 2; i++)
    {
        m_write_ds[i] = m_sample->m_vk_backend->allocate_descriptor_set(m_write_ds_layout);
        m_read_ds[i]  = m_sample->m_vk_backend->allocate_descriptor_set(m_read_ds_layout);
    }

    {
        dw::vk::PipelineLayout::Desc desc;

        desc.add_descriptor_set_layout(m_write_ds_layout);
        desc.add_descriptor_set_layout(m_sample->m_gpu_resources->g_buffer_ds_layout);
        desc.add_descriptor_set_layout(m_sample->m_gpu_resources->g_buffer_ds_layout);
        desc.add_descriptor_set_layout(m_sample->m_gpu_resources->combined_sampler_ds_layout);
        desc.add_descriptor_set_layout(m_read_ds_layout);

        desc.add_push_constant_range(VK_SHADER_STAGE_COMPUTE_BIT, 0, sizeof(ReprojectionPushConstants));

        m_pipeline_layout = dw::vk::PipelineLayout::create(m_sample->m_vk_backend, desc);

        dw::vk::ShaderModule::Ptr module = dw::vk::ShaderModule::create_from_file(m_sample->m_vk_backend, "shaders/temporal_reprojection.comp.spv");

        dw::vk::ComputePipeline::Desc comp_desc;

        comp_desc.set_pipeline_layout(m_pipeline_layout);
        comp_desc.set_shader_stage(module, "main");

        m_pipeline = dw::vk::ComputePipeline::create(m_sample->m_vk_backend, comp_desc);
    }

    for (int i = 0; i < 2; i++)
    {
        m_color_image[i] = dw::vk::Image::create(m_sample->m_vk_backend, VK_IMAGE_TYPE_2D, m_input_width, m_input_height, 1, 1, 1, VK_FORMAT_R16G16B16A16_SFLOAT, VMA_MEMORY_USAGE_GPU_ONLY, VK_IMAGE_USAGE_TRANSFER_DST_BIT | VK_IMAGE_USAGE_TRANSFER_SRC_BIT | VK_IMAGE_USAGE_SAMPLED_BIT | VK_IMAGE_USAGE_STORAGE_BIT, VK_SAMPLE_COUNT_1_BIT);
        m_color_image[i]->set_name(m_name + " Reprojection Color");

        m_color_view[i] = dw::vk::ImageView::create(m_sample->m_vk_backend, m_color_image[i], VK_IMAGE_VIEW_TYPE_2D, VK_IMAGE_ASPECT_COLOR_BIT);
        m_color_view[i]->set_name(m_name + " Reprojection Color");

        m_history_length_image[i] = dw::vk::Image::create(m_sample->m_vk_backend, VK_IMAGE_TYPE_2D, m_input_width, m_input_height, 1, 1, 1, VK_FORMAT_R16_SFLOAT, VMA_MEMORY_USAGE_GPU_ONLY, VK_IMAGE_USAGE_TRANSFER_DST_BIT | VK_IMAGE_USAGE_SAMPLED_BIT | VK_IMAGE_USAGE_STORAGE_BIT, VK_SAMPLE_COUNT_1_BIT);
        m_history_length_image[i]->set_name(m_name + " Reprojection History");

        m_history_length_view[i] = dw::vk::ImageView::create(m_sample->m_vk_backend, m_history_length_image[i], VK_IMAGE_VIEW_TYPE_2D, VK_IMAGE_ASPECT_COLOR_BIT);
        m_history_length_view[i]->set_name(m_name + " Reprojection History");
    }

    // Reprojection write
    {
        std::vector<VkDescriptorImageInfo> image_infos;
        std::vector<VkWriteDescriptorSet>  write_datas;
        VkWriteDescriptorSet               write_data;

        image_infos.reserve(4);
        write_datas.reserve(4);

        for (int i = 0; i < 2; i++)
        {
            {
                VkDescriptorImageInfo storage_image_info;

                storage_image_info.sampler     = VK_NULL_HANDLE;
                storage_image_info.imageView   = m_color_view[i]->handle();
                storage_image_info.imageLayout = VK_IMAGE_LAYOUT_GENERAL;

                image_infos.push_back(storage_image_info);

                DW_ZERO_MEMORY(write_data);

                write_data.sType           = VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET;
                write_data.descriptorCount = 1;
                write_data.descriptorType  = VK_DESCRIPTOR_TYPE_STORAGE_IMAGE;
                write_data.pImageInfo      = &image_infos.back();
                write_data.dstBinding      = 0;
                write_data.dstSet          = m_write_ds[i]->handle();

                write_datas.push_back(write_data);
            }

            {
                VkDescriptorImageInfo storage_image_info;

                storage_image_info.sampler     = VK_NULL_HANDLE;
                storage_image_info.imageView   = m_history_length_view[i]->handle();
                storage_image_info.imageLayout = VK_IMAGE_LAYOUT_GENERAL;

                image_infos.push_back(storage_image_info);

                DW_ZERO_MEMORY(write_data);

                write_data.sType           = VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET;
                write_data.descriptorCount = 1;
                write_data.descriptorType  = VK_DESCRIPTOR_TYPE_STORAGE_IMAGE;
                write_data.pImageInfo      = &image_infos.back();
                write_data.dstBinding      = 1;
                write_data.dstSet          = m_write_ds[i]->handle();

                write_datas.push_back(write_data);
            }
        }

        vkUpdateDescriptorSets(m_sample->m_vk_backend->device(), write_datas.size(), write_datas.data(), 0, nullptr);
    }

    // Reprojection read
    {
        std::vector<VkDescriptorImageInfo> image_infos;
        std::vector<VkWriteDescriptorSet>  write_datas;
        VkWriteDescriptorSet               write_data;

        image_infos.reserve(4);
        write_datas.reserve(4);

        for (int i = 0; i < 2; i++)
        {
            {
                VkDescriptorImageInfo sampler_image_info;

                sampler_image_info.sampler     = m_sample->m_vk_backend->nearest_sampler()->handle();
                sampler_image_info.imageView   = m_color_view[i]->handle();
                sampler_image_info.imageLayout = VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL;

                image_infos.push_back(sampler_image_info);

                DW_ZERO_MEMORY(write_data);

                write_data.sType           = VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET;
                write_data.descriptorCount = 1;
                write_data.descriptorType  = VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER;
                write_data.pImageInfo      = &image_infos.back();
                write_data.dstBinding      = 0;
                write_data.dstSet          = m_read_ds[i]->handle();

                write_datas.push_back(write_data);
            }

            {
                VkDescriptorImageInfo sampler_image_info;

                sampler_image_info.sampler     = m_sample->m_vk_backend->nearest_sampler()->handle();
                sampler_image_info.imageView   = m_history_length_view[i]->handle();
                sampler_image_info.imageLayout = VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL;

                image_infos.push_back(sampler_image_info);

                DW_ZERO_MEMORY(write_data);

                write_data.sType           = VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET;
                write_data.descriptorCount = 1;
                write_data.descriptorType  = VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER;
                write_data.pImageInfo      = &image_infos.back();
                write_data.dstBinding      = 1;
                write_data.dstSet          = m_read_ds[i]->handle();

                write_datas.push_back(write_data);
            }
        }

        vkUpdateDescriptorSets(m_sample->m_vk_backend->device(), write_datas.size(), write_datas.data(), 0, nullptr);
    }
}

TemporalDenoiser::~TemporalDenoiser()
{
}

void TemporalDenoiser::reproject(dw::vk::CommandBuffer::Ptr cmd_buf, dw::vk::DescriptorSet::Ptr input)
{
    clear_images(cmd_buf);

    DW_SCOPED_SAMPLE(m_name + " Temporal Reprojection", cmd_buf);

    VkImageSubresourceRange subresource_range = { VK_IMAGE_ASPECT_COLOR_BIT, 0, 1, 0, 1 };

    {
        std::vector<VkMemoryBarrier> memory_barriers = {
            memory_barrier(VK_ACCESS_SHADER_READ_BIT, VK_ACCESS_SHADER_WRITE_BIT)
        };

        std::vector<VkImageMemoryBarrier> image_barriers = {
            image_memory_barrier(m_color_image[m_sample->m_ping_pong], VK_IMAGE_LAYOUT_UNDEFINED, VK_IMAGE_LAYOUT_GENERAL, subresource_range, VK_ACCESS_SHADER_READ_BIT, VK_ACCESS_SHADER_WRITE_BIT),
            image_memory_barrier(m_history_length_image[m_sample->m_ping_pong], VK_IMAGE_LAYOUT_UNDEFINED, VK_IMAGE_LAYOUT_GENERAL, subresource_range, VK_ACCESS_SHADER_READ_BIT, VK_ACCESS_SHADER_WRITE_BIT)
        };

        pipeline_barrier(cmd_buf, memory_barriers, image_barriers, VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT, VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT);
    }

    const uint32_t NUM_THREADS = 32;

    vkCmdBindPipeline(cmd_buf->handle(), VK_PIPELINE_BIND_POINT_COMPUTE, m_pipeline->handle());

    ReprojectionPushConstants push_constants;

    push_constants.alpha                 = m_alpha;
    push_constants.neighborhood_scale    = m_neighborhood_scale;
    push_constants.use_variance_clipping = (uint32_t)m_use_variance_clipping;

    vkCmdPushConstants(cmd_buf->handle(), m_pipeline_layout->handle(), VK_SHADER_STAGE_COMPUTE_BIT, 0, sizeof(push_constants), &push_constants);

    VkDescriptorSet descriptor_sets[] = {
        m_write_ds[m_sample->m_ping_pong]->handle(),
        m_sample->m_gpu_resources->downsampled_g_buffer_ds[m_sample->m_ping_pong]->handle(),
        m_sample->m_gpu_resources->downsampled_g_buffer_ds[!m_sample->m_ping_pong]->handle(),
        input->handle(),
        m_read_ds[!m_sample->m_ping_pong]->handle()
    };

    vkCmdBindDescriptorSets(cmd_buf->handle(), VK_PIPELINE_BIND_POINT_COMPUTE, m_pipeline_layout->handle(), 0, 5, descriptor_sets, 0, nullptr);

    vkCmdDispatch(cmd_buf->handle(), static_cast<uint32_t>(ceil(float(m_input_width) / float(NUM_THREADS))), static_cast<uint32_t>(ceil(float(m_input_height) / float(NUM_THREADS))), 1);

    {
        std::vector<VkMemoryBarrier> memory_barriers = {
            memory_barrier(VK_ACCESS_SHADER_WRITE_BIT, VK_ACCESS_SHADER_READ_BIT)
        };

        std::vector<VkImageMemoryBarrier> image_barriers = {
            image_memory_barrier(m_color_image[m_sample->m_ping_pong], VK_IMAGE_LAYOUT_GENERAL, VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL, subresource_range, VK_ACCESS_SHADER_WRITE_BIT, VK_ACCESS_SHADER_READ_BIT),
            image_memory_barrier(m_history_length_image[m_sample->m_ping_pong], VK_IMAGE_LAYOUT_GENERAL, VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL, subresource_range, VK_ACCESS_SHADER_WRITE_BIT, VK_ACCESS_SHADER_READ_BIT)
        };

        pipeline_barrier(cmd_buf, memory_barriers, image_barriers, VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT, VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT);
    }
}

void TemporalDenoiser::clear_images(dw::vk::CommandBuffer::Ptr cmd_buf)
{
    if (m_sample->m_first_frame)
    {
        VkImageSubresourceRange subresource_range = { VK_IMAGE_ASPECT_COLOR_BIT, 0, 1, 0, 1 };

        VkClearColorValue color;

        color.float32[0] = 0.0f;
        color.float32[1] = 0.0f;
        color.float32[2] = 0.0f;
        color.float32[3] = 0.0f;

        dw::vk::utilities::set_image_layout(
            cmd_buf->handle(),
            m_history_length_image[!m_sample->m_ping_pong]->handle(),
            VK_IMAGE_LAYOUT_UNDEFINED,
            VK_IMAGE_LAYOUT_GENERAL,
            subresource_range);

        dw::vk::utilities::set_image_layout(
            cmd_buf->handle(),
            m_color_image[!m_sample->m_ping_pong]->handle(),
            VK_IMAGE_LAYOUT_UNDEFINED,
            VK_IMAGE_LAYOUT_GENERAL,
            subresource_range);

        vkCmdClearColorImage(cmd_buf->handle(), m_history_length_image[!m_sample->m_ping_pong]->handle(), VK_IMAGE_LAYOUT_GENERAL, &color, 1, &subresource_range);
        vkCmdClearColorImage(cmd_buf->handle(), m_color_image[!m_sample->m_ping_pong]->handle(), VK_IMAGE_LAYOUT_GENERAL, &color, 1, &subresource_range);

        dw::vk::utilities::set_image_layout(
            cmd_buf->handle(),
            m_history_length_image[!m_sample->m_ping_pong]->handle(),
            VK_IMAGE_LAYOUT_GENERAL,
            VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL,
            subresource_range);

        dw::vk::utilities::set_image_layout(
            cmd_buf->handle(),
            m_color_image[!m_sample->m_ping_pong]->handle(),
            VK_IMAGE_LAYOUT_GENERAL,
            VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL,
            subresource_range);
    }
}

dw::vk::DescriptorSet::Ptr TemporalDenoiser::output_image_ds()
{
    return m_read_ds[m_sample->m_ping_pong];
}

ReflectionDenoiser::ReflectionDenoiser(HybridRendering* sample, std::string name, uint32_t input_width, uint32_t input_height, uint32_t blur_radius) :
    m_name(name), m_sample(sample), m_input_width(input_width), m_input_height(input_height), m_blur_radius(blur_radius)
{
    m_reconstruction_image = dw::vk::Image::create(sample->m_vk_backend, VK_IMAGE_TYPE_2D, input_width, input_height, 1, 1, 1, VK_FORMAT_R16G16B16A16_SFLOAT, VMA_MEMORY_USAGE_GPU_ONLY, VK_IMAGE_USAGE_TRANSFER_DST_BIT | VK_IMAGE_USAGE_SAMPLED_BIT | VK_IMAGE_USAGE_STORAGE_BIT, VK_SAMPLE_COUNT_1_BIT);
    m_reconstruction_image->set_name(name + " Reconstructed");

    m_reconstruction_image_view = dw::vk::ImageView::create(sample->m_vk_backend, m_reconstruction_image, VK_IMAGE_VIEW_TYPE_2D, VK_IMAGE_ASPECT_COLOR_BIT);
    m_reconstruction_image_view->set_name(name + " Reconstructed");

    m_reconstruction_write_ds = sample->m_vk_backend->allocate_descriptor_set(sample->m_gpu_resources->storage_image_ds_layout);
    m_reconstruction_read_ds  = sample->m_vk_backend->allocate_descriptor_set(sample->m_gpu_resources->combined_sampler_ds_layout);

    // Reconstructed write
    {
        VkDescriptorImageInfo storage_image_info;

        storage_image_info.sampler     = VK_NULL_HANDLE;
        storage_image_info.imageView   = m_reconstruction_image_view->handle();
        storage_image_info.imageLayout = VK_IMAGE_LAYOUT_GENERAL;

        VkWriteDescriptorSet write_data;

        DW_ZERO_MEMORY(write_data);

        write_data.sType           = VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET;
        write_data.descriptorCount = 1;
        write_data.descriptorType  = VK_DESCRIPTOR_TYPE_STORAGE_IMAGE;
        write_data.pImageInfo      = &storage_image_info;
        write_data.dstBinding      = 0;
        write_data.dstSet          = m_reconstruction_write_ds->handle();

        vkUpdateDescriptorSets(sample->m_vk_backend->device(), 1, &write_data, 0, nullptr);
    }

    // Reconstructed read
    {
        VkDescriptorImageInfo sampler_image_info;

        sampler_image_info.sampler     = sample->m_vk_backend->bilinear_sampler()->handle();
        sampler_image_info.imageView   = m_reconstruction_image_view->handle();
        sampler_image_info.imageLayout = VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL;

        VkWriteDescriptorSet write_data;

        DW_ZERO_MEMORY(write_data);

        write_data.sType           = VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET;
        write_data.descriptorCount = 1;
        write_data.descriptorType  = VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER;
        write_data.pImageInfo      = &sampler_image_info;
        write_data.dstBinding      = 0;
        write_data.dstSet          = m_reconstruction_read_ds->handle();

        vkUpdateDescriptorSets(sample->m_vk_backend->device(), 1, &write_data, 0, nullptr);
    }

    {
        dw::vk::PipelineLayout::Desc desc;

        desc.add_descriptor_set_layout(sample->m_gpu_resources->storage_image_ds_layout);
        desc.add_descriptor_set_layout(sample->m_gpu_resources->combined_sampler_ds_layout);
        desc.add_descriptor_set_layout(m_sample->m_gpu_resources->g_buffer_ds_layout);
        desc.add_descriptor_set_layout(m_sample->m_gpu_resources->per_frame_ds_layout);

        desc.add_push_constant_range(VK_SHADER_STAGE_COMPUTE_BIT, 0, sizeof(ReconstructionPushConstants));

        m_reconstruction_layout = dw::vk::PipelineLayout::create(m_sample->m_vk_backend, desc);

        dw::vk::ShaderModule::Ptr module = dw::vk::ShaderModule::create_from_file(m_sample->m_vk_backend, "shaders/denoiser_reconstruction.comp.spv");

        dw::vk::ComputePipeline::Desc comp_desc;

        comp_desc.set_pipeline_layout(m_reconstruction_layout);
        comp_desc.set_shader_stage(module, "main");

        m_reconstruction_pipeline = dw::vk::ComputePipeline::create(m_sample->m_vk_backend, comp_desc);
    }
}

ReflectionDenoiser::~ReflectionDenoiser()
{
}

void ReflectionDenoiser::denoise(dw::vk::CommandBuffer::Ptr cmd_buf, dw::vk::DescriptorSet::Ptr input)
{
}

void ReflectionDenoiser::reconstruct_pass(dw::vk::CommandBuffer::Ptr cmd_buf, dw::vk::DescriptorSet::Ptr input)
{
    VkImageSubresourceRange subresource_range = { VK_IMAGE_ASPECT_COLOR_BIT, 0, 1, 0, 1 };

    dw::vk::utilities::set_image_layout(
        cmd_buf->handle(),
        m_reconstruction_image->handle(),
        VK_IMAGE_LAYOUT_UNDEFINED,
        VK_IMAGE_LAYOUT_GENERAL,
        subresource_range);

    const uint32_t NUM_THREADS = 32;

    vkCmdBindPipeline(cmd_buf->handle(), VK_PIPELINE_BIND_POINT_COMPUTE, m_reconstruction_pipeline->handle());

    ReconstructionPushConstants push_constants;

    float z_buffer_params_x = -1.0 + (m_sample->m_near_plane / m_sample->m_far_plane);

    push_constants.z_buffer_params = glm::vec4(z_buffer_params_x, 1.0f, z_buffer_params_x / m_sample->m_near_plane, 1.0f / m_sample->m_near_plane);
    push_constants.num_frames      = m_sample->m_num_frames;

    vkCmdPushConstants(cmd_buf->handle(), m_reconstruction_layout->handle(), VK_SHADER_STAGE_COMPUTE_BIT, 0, sizeof(push_constants), &push_constants);

    const uint32_t dynamic_offset = m_sample->m_ubo_size * m_sample->m_vk_backend->current_frame_idx();

    VkDescriptorSet descriptor_sets[] = {
        m_reconstruction_write_ds->handle(),
        input->handle(),
        m_sample->m_gpu_resources->g_buffer_ds[m_sample->m_ping_pong]->handle(),
        m_sample->m_gpu_resources->per_frame_ds->handle()
    };

    vkCmdBindDescriptorSets(cmd_buf->handle(), VK_PIPELINE_BIND_POINT_COMPUTE, m_reconstruction_layout->handle(), 0, 4, descriptor_sets, 1, &dynamic_offset);

    vkCmdDispatch(cmd_buf->handle(), static_cast<uint32_t>(ceil(float(m_reconstruction_image->width()) / float(NUM_THREADS))), static_cast<uint32_t>(ceil(float(m_reconstruction_image->height()) / float(NUM_THREADS))), 1);

    dw::vk::utilities::set_image_layout(
        cmd_buf->handle(),
        m_reconstruction_image->handle(),
        VK_IMAGE_LAYOUT_GENERAL,
        VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL,
        subresource_range);
}

void ReflectionDenoiser::temporal_pre_pass(dw::vk::CommandBuffer::Ptr cmd_buf)
{
}

void ReflectionDenoiser::temporal_main_pass(dw::vk::CommandBuffer::Ptr cmd_buf)
{
}

void ReflectionDenoiser::bilateral_blur()
{
}

dw::vk::DescriptorSet::Ptr ReflectionDenoiser::denoised_image_ds()
{
    return m_blur_read_ds;
}

DW_DECLARE_MAIN(HybridRendering)