#include <application.h>
#include <camera.h>
#include <material.h>
#include <mesh.h>
#include <vk.h>
#include <profiler.h>
#include <assimp/scene.h>
#include <vk_mem_alloc.h>
#include <scene.h>

// Uniform buffer data structure.
struct Transforms
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
    glm::mat4 model;
    DW_ALIGNED(16)
    glm::mat4 view;
    DW_ALIGNED(16)
    glm::mat4 proj;
    DW_ALIGNED(16)
    glm::vec4 cam_pos;
    DW_ALIGNED(16)
    glm::vec4 light_dir;
};

struct ShadowPushConstants
{
    float    light_radius;
    float    alpha;
    uint32_t num_frames;
};

struct BilateralBlurPushConstants
{
    glm::vec4 z_buffer_params;
    glm::vec2 direction;
    glm::vec2 pixel_size;
};

class Sample : public dw::Application
{
protected:
    // -----------------------------------------------------------------------------------------------------------------------------------

    bool init(int argc, const char* argv[]) override
    {
        if (!create_uniform_buffer())
            return false;

        // Load mesh.
        if (!load_mesh())
        {
            DW_LOG_INFO("Failed to load mesh");
            return false;
        }

        dw::vk::Sampler::Desc sampler_desc;

        sampler_desc.mag_filter        = VK_FILTER_LINEAR;
        sampler_desc.min_filter        = VK_FILTER_LINEAR;
        sampler_desc.mipmap_mode       = VK_SAMPLER_MIPMAP_MODE_NEAREST;
        sampler_desc.address_mode_u    = VK_SAMPLER_ADDRESS_MODE_CLAMP_TO_EDGE;
        sampler_desc.address_mode_v    = VK_SAMPLER_ADDRESS_MODE_CLAMP_TO_EDGE;
        sampler_desc.address_mode_w    = VK_SAMPLER_ADDRESS_MODE_CLAMP_TO_EDGE;
        sampler_desc.mip_lod_bias      = 0.0f;
        sampler_desc.anisotropy_enable = VK_FALSE;
        sampler_desc.max_anisotropy    = 1.0f;
        sampler_desc.compare_enable    = false;
        sampler_desc.compare_op        = VK_COMPARE_OP_NEVER;
        sampler_desc.min_lod           = 0.0f;
        sampler_desc.max_lod           = 0.0f;
        sampler_desc.border_color      = VK_BORDER_COLOR_FLOAT_OPAQUE_WHITE;
        sampler_desc.unnormalized_coordinates;

        m_bilinear_sampler = dw::vk::Sampler::create(m_vk_backend, sampler_desc);

        load_blue_noise();
        create_output_images();
        create_render_passes();
        create_framebuffers();
        create_descriptor_set_layouts();
        create_descriptor_sets();
        write_descriptor_sets();
        create_deferred_pipeline();
        create_gbuffer_pipeline();
        create_shadow_mask_ray_tracing_pipeline();
        //create_reflection_ray_tracing_pipeline();
        create_compute_pipeline();

        // Create camera.
        create_camera();

        m_light_direction = glm::normalize(glm::vec3(0.2f, 0.9770f, 0.2f));

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
            DW_SCOPED_SAMPLE("update", cmd_buf);

            if (m_debug_gui)
            {
                ImGui::InputFloat("Light Radius", &m_light_radius);
                ImGui::SliderFloat("Alpha", &m_alpha, 0.0f, 1.0f);
                ImGui::Checkbox("Shadow Temporal Filter", &m_temporal_accumulation);
                ImGui::Checkbox("Shadow Spatial Blur", &m_use_bilateral_blur);
            }

            // Render profiler.
            //dw::profiler::ui();

            // Update camera.
            update_camera();

            // Update uniforms.
            update_uniforms(cmd_buf);

            // Render.
            render_gbuffer(cmd_buf);
            ray_trace_shadow_mask(cmd_buf);
            if (m_use_bilateral_blur)
                bilateral_blur_shadows(cmd_buf);
            //ray_trace_reflection(cmd_buf);

            render(cmd_buf);
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
        m_blue_noise.reset();
        m_blue_noise_view.reset();

        for (int i = 0; i < 2; i++)
        {
            m_reflection_write_ds[i].reset();
            m_reflection_read_ds[i].reset();
            m_reflection_view[i].reset();
            m_reflection_image[i].reset();
        }

        m_visibility_write_ds.reset();
        m_visibility_read_ds.reset();
        m_visibility_filtered_write_ds.reset();
        m_visibility_filtered_read_ds.reset();
        m_visibility_view.reset();
        m_visibility_image.reset();
        m_visibility_filtered_view.reset();
        m_visibility_filtered_image.reset();
        m_bilateral_blur_pipeline.reset();
        m_bilateral_blur_pipeline_layout.reset();
        m_temp_blur_write_ds.reset();
        m_temp_blur_read_ds.reset();
        m_temp_blur_view.reset();
        m_temp_blur.reset();
        m_bilinear_sampler.reset();
        m_acceleration_structure_ds.reset();
        m_per_frame_ds.reset();
        m_g_buffer_ds.reset();
        m_per_frame_ds_layout.reset();
        m_g_buffer_ds_layout.reset();
        m_storage_image_ds_layout.reset();
        m_acceleration_structure_ds_layout.reset();
        m_combined_sampler_ds_layout.reset();
        m_shadow_mask_pipeline_layout.reset();
        m_reflection_pipeline_layout.reset();
        m_g_buffer_pipeline_layout.reset();
        m_deferred_pipeline_layout.reset();
        m_ubo.reset();
        m_deferred_pipeline.reset();
        m_shadow_mask_pipeline.reset();
        m_g_buffer_pipeline.reset();
        m_reflection_pipeline.reset();
        m_g_buffer_fbo.reset();
        m_g_buffer_rp.reset();
        m_g_buffer_1_view.reset();
        m_g_buffer_2_view.reset();
        m_g_buffer_3_view.reset();
        m_blue_noise_view.reset();
        m_g_buffer_depth_view.reset();
        m_blue_noise.reset();
        m_g_buffer_1.reset();
        m_g_buffer_2.reset();
        m_g_buffer_3.reset();
        m_g_buffer_depth.reset();
        m_shadow_mask_sbt.reset();
        m_reflection_sbt.reset();

        // Unload assets.
        m_scene.reset();
        m_mesh.reset();
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
        m_main_camera->update_projection(60.0f, 0.1f, 10000.0f, float(m_width) / float(m_height));

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
        m_g_buffer_1.reset();
        m_g_buffer_2.reset();
        m_g_buffer_3.reset();
        m_g_buffer_depth.reset();
        m_g_buffer_1_view.reset();
        m_g_buffer_2_view.reset();
        m_g_buffer_3_view.reset();
        m_g_buffer_depth_view.reset();
        m_visibility_image.reset();
        m_visibility_view.reset();
        m_visibility_filtered_image.reset();
        m_visibility_filtered_view.reset();

        m_visibility_image          = dw::vk::Image::create(m_vk_backend, VK_IMAGE_TYPE_2D, m_width, m_height, 1, 1, 1, VK_FORMAT_R16_SFLOAT, VMA_MEMORY_USAGE_GPU_ONLY, VK_IMAGE_USAGE_STORAGE_BIT | VK_IMAGE_USAGE_SAMPLED_BIT, VK_SAMPLE_COUNT_1_BIT);
        m_visibility_filtered_image = dw::vk::Image::create(m_vk_backend, VK_IMAGE_TYPE_2D, m_width, m_height, 1, 1, 1, VK_FORMAT_R16_SFLOAT, VMA_MEMORY_USAGE_GPU_ONLY, VK_IMAGE_USAGE_STORAGE_BIT | VK_IMAGE_USAGE_SAMPLED_BIT, VK_SAMPLE_COUNT_1_BIT);

        m_visibility_view          = dw::vk::ImageView::create(m_vk_backend, m_visibility_image, VK_IMAGE_VIEW_TYPE_2D, VK_IMAGE_ASPECT_COLOR_BIT);
        m_visibility_filtered_view = dw::vk::ImageView::create(m_vk_backend, m_visibility_filtered_image, VK_IMAGE_VIEW_TYPE_2D, VK_IMAGE_ASPECT_COLOR_BIT);

        for (int i = 0; i < 2; i++)
        {
            m_reflection_image[i].reset();
            m_reflection_view[i].reset();

            m_reflection_image[i] = dw::vk::Image::create(m_vk_backend, VK_IMAGE_TYPE_2D, m_width, m_height, 1, 1, 1, VK_FORMAT_R16G16B16A16_SFLOAT, VMA_MEMORY_USAGE_GPU_ONLY, VK_IMAGE_USAGE_STORAGE_BIT | VK_IMAGE_USAGE_SAMPLED_BIT, VK_SAMPLE_COUNT_1_BIT);
            m_reflection_view[i]  = dw::vk::ImageView::create(m_vk_backend, m_reflection_image[i], VK_IMAGE_VIEW_TYPE_2D, VK_IMAGE_ASPECT_COLOR_BIT);
        }

        m_g_buffer_1     = dw::vk::Image::create(m_vk_backend, VK_IMAGE_TYPE_2D, m_width, m_height, 1, 1, 1, VK_FORMAT_R8G8B8A8_UNORM, VMA_MEMORY_USAGE_GPU_ONLY, VK_IMAGE_USAGE_STORAGE_BIT | VK_IMAGE_USAGE_SAMPLED_BIT | VK_IMAGE_USAGE_COLOR_ATTACHMENT_BIT, VK_SAMPLE_COUNT_1_BIT);
        m_g_buffer_2     = dw::vk::Image::create(m_vk_backend, VK_IMAGE_TYPE_2D, m_width, m_height, 1, 1, 1, VK_FORMAT_R16G16B16A16_SFLOAT, VMA_MEMORY_USAGE_GPU_ONLY, VK_IMAGE_USAGE_STORAGE_BIT | VK_IMAGE_USAGE_SAMPLED_BIT | VK_IMAGE_USAGE_COLOR_ATTACHMENT_BIT, VK_SAMPLE_COUNT_1_BIT);
        m_g_buffer_3     = dw::vk::Image::create(m_vk_backend, VK_IMAGE_TYPE_2D, m_width, m_height, 1, 1, 1, VK_FORMAT_R32G32B32A32_SFLOAT, VMA_MEMORY_USAGE_GPU_ONLY, VK_IMAGE_USAGE_STORAGE_BIT | VK_IMAGE_USAGE_SAMPLED_BIT | VK_IMAGE_USAGE_COLOR_ATTACHMENT_BIT, VK_SAMPLE_COUNT_1_BIT);
        m_g_buffer_depth = dw::vk::Image::create(m_vk_backend, VK_IMAGE_TYPE_2D, m_width, m_height, 1, 1, 1, m_vk_backend->swap_chain_depth_format(), VMA_MEMORY_USAGE_GPU_ONLY, VK_IMAGE_USAGE_SAMPLED_BIT | VK_IMAGE_USAGE_DEPTH_STENCIL_ATTACHMENT_BIT, VK_SAMPLE_COUNT_1_BIT);

        m_g_buffer_1_view     = dw::vk::ImageView::create(m_vk_backend, m_g_buffer_1, VK_IMAGE_VIEW_TYPE_2D, VK_IMAGE_ASPECT_COLOR_BIT);
        m_g_buffer_2_view     = dw::vk::ImageView::create(m_vk_backend, m_g_buffer_2, VK_IMAGE_VIEW_TYPE_2D, VK_IMAGE_ASPECT_COLOR_BIT);
        m_g_buffer_3_view     = dw::vk::ImageView::create(m_vk_backend, m_g_buffer_3, VK_IMAGE_VIEW_TYPE_2D, VK_IMAGE_ASPECT_COLOR_BIT);
        m_g_buffer_depth_view = dw::vk::ImageView::create(m_vk_backend, m_g_buffer_depth, VK_IMAGE_VIEW_TYPE_2D, VK_IMAGE_ASPECT_DEPTH_BIT);

        m_temp_blur      = dw::vk::Image::create(m_vk_backend, VK_IMAGE_TYPE_2D, m_width, m_height, 1, 1, 1, VK_FORMAT_R16_SFLOAT, VMA_MEMORY_USAGE_GPU_ONLY, VK_IMAGE_USAGE_STORAGE_BIT | VK_IMAGE_USAGE_SAMPLED_BIT, VK_SAMPLE_COUNT_1_BIT);
        m_temp_blur_view = dw::vk::ImageView::create(m_vk_backend, m_temp_blur, VK_IMAGE_VIEW_TYPE_2D, VK_IMAGE_ASPECT_COLOR_BIT);
    }

    // -----------------------------------------------------------------------------------------------------------------------------------

    void create_render_passes()
    {
        std::vector<VkAttachmentDescription> attachments(4);

        // GBuffer1 attachment
        attachments[0].format         = VK_FORMAT_R8G8B8A8_UNORM;
        attachments[0].samples        = VK_SAMPLE_COUNT_1_BIT;
        attachments[0].loadOp         = VK_ATTACHMENT_LOAD_OP_CLEAR;
        attachments[0].storeOp        = VK_ATTACHMENT_STORE_OP_STORE;
        attachments[0].stencilLoadOp  = VK_ATTACHMENT_LOAD_OP_DONT_CARE;
        attachments[0].stencilStoreOp = VK_ATTACHMENT_STORE_OP_DONT_CARE;
        attachments[0].initialLayout  = VK_IMAGE_LAYOUT_UNDEFINED;
        attachments[0].finalLayout    = VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL;

        // GBuffer2 attachment
        attachments[1].format         = VK_FORMAT_R16G16B16A16_SFLOAT;
        attachments[1].samples        = VK_SAMPLE_COUNT_1_BIT;
        attachments[1].loadOp         = VK_ATTACHMENT_LOAD_OP_CLEAR;
        attachments[1].storeOp        = VK_ATTACHMENT_STORE_OP_STORE;
        attachments[1].stencilLoadOp  = VK_ATTACHMENT_LOAD_OP_DONT_CARE;
        attachments[1].stencilStoreOp = VK_ATTACHMENT_STORE_OP_DONT_CARE;
        attachments[1].initialLayout  = VK_IMAGE_LAYOUT_UNDEFINED;
        attachments[1].finalLayout    = VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL;

        // GBuffer3 attachment
        attachments[2].format         = VK_FORMAT_R32G32B32A32_SFLOAT;
        attachments[2].samples        = VK_SAMPLE_COUNT_1_BIT;
        attachments[2].loadOp         = VK_ATTACHMENT_LOAD_OP_CLEAR;
        attachments[2].storeOp        = VK_ATTACHMENT_STORE_OP_STORE;
        attachments[2].stencilLoadOp  = VK_ATTACHMENT_LOAD_OP_DONT_CARE;
        attachments[2].stencilStoreOp = VK_ATTACHMENT_STORE_OP_DONT_CARE;
        attachments[2].initialLayout  = VK_IMAGE_LAYOUT_UNDEFINED;
        attachments[2].finalLayout    = VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL;

        // Depth attachment
        attachments[3].format         = m_vk_backend->swap_chain_depth_format();
        attachments[3].samples        = VK_SAMPLE_COUNT_1_BIT;
        attachments[3].loadOp         = VK_ATTACHMENT_LOAD_OP_CLEAR;
        attachments[3].storeOp        = VK_ATTACHMENT_STORE_OP_STORE;
        attachments[3].stencilLoadOp  = VK_ATTACHMENT_LOAD_OP_CLEAR;
        attachments[3].stencilStoreOp = VK_ATTACHMENT_STORE_OP_DONT_CARE;
        attachments[3].initialLayout  = VK_IMAGE_LAYOUT_UNDEFINED;
        attachments[3].finalLayout    = VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL;

        VkAttachmentReference gbuffer_references[3];

        gbuffer_references[0].attachment = 0;
        gbuffer_references[0].layout     = VK_IMAGE_LAYOUT_COLOR_ATTACHMENT_OPTIMAL;

        gbuffer_references[1].attachment = 1;
        gbuffer_references[1].layout     = VK_IMAGE_LAYOUT_COLOR_ATTACHMENT_OPTIMAL;

        gbuffer_references[2].attachment = 2;
        gbuffer_references[2].layout     = VK_IMAGE_LAYOUT_COLOR_ATTACHMENT_OPTIMAL;

        VkAttachmentReference depth_reference;
        depth_reference.attachment = 3;
        depth_reference.layout     = VK_IMAGE_LAYOUT_DEPTH_STENCIL_ATTACHMENT_OPTIMAL;

        std::vector<VkSubpassDescription> subpass_description(1);

        subpass_description[0].pipelineBindPoint       = VK_PIPELINE_BIND_POINT_GRAPHICS;
        subpass_description[0].colorAttachmentCount    = 3;
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

        m_g_buffer_rp = dw::vk::RenderPass::create(m_vk_backend, attachments, subpass_description, dependencies);
    }

    // -----------------------------------------------------------------------------------------------------------------------------------

    void create_framebuffers()
    {
        m_g_buffer_fbo.reset();
        m_g_buffer_fbo = dw::vk::Framebuffer::create(m_vk_backend, m_g_buffer_rp, { m_g_buffer_1_view, m_g_buffer_2_view, m_g_buffer_3_view, m_g_buffer_depth_view }, m_width, m_height, 1);
    }

    // -----------------------------------------------------------------------------------------------------------------------------------

    bool create_uniform_buffer()
    {
        m_ubo_size = m_vk_backend->aligned_dynamic_ubo_size(sizeof(Transforms));
        m_ubo      = dw::vk::Buffer::create(m_vk_backend, VK_BUFFER_USAGE_UNIFORM_BUFFER_BIT, m_ubo_size * dw::vk::Backend::kMaxFramesInFlight, VMA_MEMORY_USAGE_CPU_TO_GPU, VMA_ALLOCATION_CREATE_MAPPED_BIT);

        return true;
    }

    // -----------------------------------------------------------------------------------------------------------------------------------

    void create_descriptor_set_layouts()
    {
        {
            dw::vk::DescriptorSetLayout::Desc desc;

            desc.add_binding(0, VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER_DYNAMIC, 1, VK_SHADER_STAGE_RAYGEN_BIT_NV | VK_SHADER_STAGE_CLOSEST_HIT_BIT_NV | VK_SHADER_STAGE_FRAGMENT_BIT | VK_SHADER_STAGE_VERTEX_BIT);
            desc.add_binding(1, VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER, 1, VK_SHADER_STAGE_RAYGEN_BIT_NV | VK_SHADER_STAGE_CLOSEST_HIT_BIT_NV | VK_SHADER_STAGE_FRAGMENT_BIT | VK_SHADER_STAGE_VERTEX_BIT);

            m_per_frame_ds_layout = dw::vk::DescriptorSetLayout::create(m_vk_backend, desc);
        }

        {
            dw::vk::DescriptorSetLayout::Desc desc;

            desc.add_binding(0, VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER, 1, VK_SHADER_STAGE_RAYGEN_BIT_NV | VK_SHADER_STAGE_CLOSEST_HIT_BIT_NV | VK_SHADER_STAGE_FRAGMENT_BIT | VK_SHADER_STAGE_COMPUTE_BIT);
            desc.add_binding(1, VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER, 1, VK_SHADER_STAGE_RAYGEN_BIT_NV | VK_SHADER_STAGE_CLOSEST_HIT_BIT_NV | VK_SHADER_STAGE_FRAGMENT_BIT | VK_SHADER_STAGE_COMPUTE_BIT);
            desc.add_binding(2, VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER, 1, VK_SHADER_STAGE_RAYGEN_BIT_NV | VK_SHADER_STAGE_CLOSEST_HIT_BIT_NV | VK_SHADER_STAGE_FRAGMENT_BIT | VK_SHADER_STAGE_COMPUTE_BIT);
            desc.add_binding(3, VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER, 1, VK_SHADER_STAGE_RAYGEN_BIT_NV | VK_SHADER_STAGE_CLOSEST_HIT_BIT_NV | VK_SHADER_STAGE_FRAGMENT_BIT | VK_SHADER_STAGE_COMPUTE_BIT);

            m_g_buffer_ds_layout = dw::vk::DescriptorSetLayout::create(m_vk_backend, desc);
        }

        {
            dw::vk::DescriptorSetLayout::Desc desc;

            desc.add_binding(0, VK_DESCRIPTOR_TYPE_ACCELERATION_STRUCTURE_NV, 1, VK_SHADER_STAGE_RAYGEN_BIT_NV | VK_SHADER_STAGE_CLOSEST_HIT_BIT_NV);

            m_acceleration_structure_ds_layout = dw::vk::DescriptorSetLayout::create(m_vk_backend, desc);
        }

        {
            dw::vk::DescriptorSetLayout::Desc desc;

            desc.add_binding(0, VK_DESCRIPTOR_TYPE_STORAGE_IMAGE, 1, VK_SHADER_STAGE_RAYGEN_BIT_NV | VK_SHADER_STAGE_COMPUTE_BIT);

            m_storage_image_ds_layout = dw::vk::DescriptorSetLayout::create(m_vk_backend, desc);
        }

        {
            dw::vk::DescriptorSetLayout::Desc desc;

            desc.add_binding(0, VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER, 1, VK_SHADER_STAGE_RAYGEN_BIT_NV | VK_SHADER_STAGE_FRAGMENT_BIT | VK_SHADER_STAGE_COMPUTE_BIT);

            m_combined_sampler_ds_layout = dw::vk::DescriptorSetLayout::create(m_vk_backend, desc);
        }
    }

    // -----------------------------------------------------------------------------------------------------------------------------------

    void create_descriptor_sets()
    {
        m_per_frame_ds                 = m_vk_backend->allocate_descriptor_set(m_per_frame_ds_layout);
        m_acceleration_structure_ds    = m_vk_backend->allocate_descriptor_set(m_acceleration_structure_ds_layout);
        m_g_buffer_ds                  = m_vk_backend->allocate_descriptor_set(m_g_buffer_ds_layout);
        m_temp_blur_write_ds           = m_vk_backend->allocate_descriptor_set(m_storage_image_ds_layout);
        m_temp_blur_read_ds            = m_vk_backend->allocate_descriptor_set(m_combined_sampler_ds_layout);
        m_visibility_write_ds          = m_vk_backend->allocate_descriptor_set(m_storage_image_ds_layout);
        m_visibility_read_ds           = m_vk_backend->allocate_descriptor_set(m_combined_sampler_ds_layout);
        m_visibility_filtered_read_ds  = m_vk_backend->allocate_descriptor_set(m_combined_sampler_ds_layout);
        m_visibility_filtered_write_ds = m_vk_backend->allocate_descriptor_set(m_storage_image_ds_layout);

        for (int i = 0; i < 2; i++)
        {
            m_reflection_write_ds[i] = m_vk_backend->allocate_descriptor_set(m_storage_image_ds_layout);
            m_reflection_read_ds[i]  = m_vk_backend->allocate_descriptor_set(m_combined_sampler_ds_layout);
        }
    }

    // -----------------------------------------------------------------------------------------------------------------------------------

    void write_descriptor_sets()
    {
        // Per-frame
        {
            VkDescriptorBufferInfo buffer_info;

            buffer_info.range  = VK_WHOLE_SIZE;
            buffer_info.offset = 0;
            buffer_info.buffer = m_ubo->handle();

            VkWriteDescriptorSet write_data[2];
            DW_ZERO_MEMORY(write_data[0]);
            DW_ZERO_MEMORY(write_data[1]);

            write_data[0].sType           = VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET;
            write_data[0].descriptorCount = 1;
            write_data[0].descriptorType  = VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER_DYNAMIC;
            write_data[0].pBufferInfo     = &buffer_info;
            write_data[0].dstBinding      = 0;
            write_data[0].dstSet          = m_per_frame_ds->handle();

            VkDescriptorImageInfo blue_noise_image;
            blue_noise_image.sampler     = m_bilinear_sampler->handle();
            blue_noise_image.imageView   = m_blue_noise_view->handle();
            blue_noise_image.imageLayout = VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL;

            write_data[1].sType           = VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET;
            write_data[1].descriptorCount = 1;
            write_data[1].descriptorType  = VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER;
            write_data[1].pImageInfo      = &blue_noise_image;
            write_data[1].dstBinding      = 1;
            write_data[1].dstSet          = m_per_frame_ds->handle();

            vkUpdateDescriptorSets(m_vk_backend->device(), 2, write_data, 0, nullptr);
        }

        // G-Buffer
        {
            VkDescriptorImageInfo image_info[4];

            image_info[0].sampler     = dw::Material::common_sampler()->handle();
            image_info[0].imageView   = m_g_buffer_1_view->handle();
            image_info[0].imageLayout = VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL;

            image_info[1].sampler     = dw::Material::common_sampler()->handle();
            image_info[1].imageView   = m_g_buffer_2_view->handle();
            image_info[1].imageLayout = VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL;

            image_info[2].sampler     = dw::Material::common_sampler()->handle();
            image_info[2].imageView   = m_g_buffer_3_view->handle();
            image_info[2].imageLayout = VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL;

            image_info[3].sampler     = dw::Material::common_sampler()->handle();
            image_info[3].imageView   = m_g_buffer_depth_view->handle();
            image_info[3].imageLayout = VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL;

            VkWriteDescriptorSet write_data[4];
            DW_ZERO_MEMORY(write_data[0]);
            DW_ZERO_MEMORY(write_data[1]);
            DW_ZERO_MEMORY(write_data[2]);
            DW_ZERO_MEMORY(write_data[3]);

            write_data[0].sType           = VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET;
            write_data[0].descriptorCount = 1;
            write_data[0].descriptorType  = VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER;
            write_data[0].pImageInfo      = &image_info[0];
            write_data[0].dstBinding      = 0;
            write_data[0].dstSet          = m_g_buffer_ds->handle();

            write_data[1].sType           = VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET;
            write_data[1].descriptorCount = 1;
            write_data[1].descriptorType  = VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER;
            write_data[1].pImageInfo      = &image_info[1];
            write_data[1].dstBinding      = 1;
            write_data[1].dstSet          = m_g_buffer_ds->handle();

            write_data[2].sType           = VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET;
            write_data[2].descriptorCount = 1;
            write_data[2].descriptorType  = VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER;
            write_data[2].pImageInfo      = &image_info[2];
            write_data[2].dstBinding      = 2;
            write_data[2].dstSet          = m_g_buffer_ds->handle();

            write_data[3].sType           = VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET;
            write_data[3].descriptorCount = 1;
            write_data[3].descriptorType  = VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER;
            write_data[3].pImageInfo      = &image_info[3];
            write_data[3].dstBinding      = 3;
            write_data[3].dstSet          = m_g_buffer_ds->handle();

            vkUpdateDescriptorSets(m_vk_backend->device(), 4, &write_data[0], 0, nullptr);
        }

        // Acceleration Structure
        {
            VkWriteDescriptorSet write_data;
            DW_ZERO_MEMORY(write_data);

            VkWriteDescriptorSetAccelerationStructureNV descriptor_as;

            descriptor_as.sType                      = VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET_ACCELERATION_STRUCTURE_NV;
            descriptor_as.pNext                      = nullptr;
            descriptor_as.accelerationStructureCount = 1;
            descriptor_as.pAccelerationStructures    = &m_scene->acceleration_structure()->handle();

            write_data.sType           = VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET;
            write_data.pNext           = &descriptor_as;
            write_data.descriptorCount = 1;
            write_data.descriptorType  = VK_DESCRIPTOR_TYPE_ACCELERATION_STRUCTURE_NV;
            write_data.dstBinding      = 0;
            write_data.dstSet          = m_acceleration_structure_ds->handle();

            vkUpdateDescriptorSets(m_vk_backend->device(), 1, &write_data, 0, nullptr);
        }

        // Visibility write
        {
            VkWriteDescriptorSet write_data;
            DW_ZERO_MEMORY(write_data);

            VkDescriptorImageInfo storage_image_info;

            storage_image_info.sampler     = VK_NULL_HANDLE;
            storage_image_info.imageView   = m_visibility_view->handle();
            storage_image_info.imageLayout = VK_IMAGE_LAYOUT_GENERAL;

            write_data.sType           = VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET;
            write_data.descriptorCount = 1;
            write_data.descriptorType  = VK_DESCRIPTOR_TYPE_STORAGE_IMAGE;
            write_data.pImageInfo      = &storage_image_info;
            write_data.dstBinding      = 0;
            write_data.dstSet          = m_visibility_write_ds->handle();

            vkUpdateDescriptorSets(m_vk_backend->device(), 1, &write_data, 0, nullptr);
        }

        // Visibility read
        {
            VkWriteDescriptorSet write_data;
            DW_ZERO_MEMORY(write_data);

            VkDescriptorImageInfo sampler_image_info;

            sampler_image_info.sampler     = m_bilinear_sampler->handle();
            sampler_image_info.imageView   = m_visibility_view->handle();
            sampler_image_info.imageLayout = VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL;

            write_data.sType           = VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET;
            write_data.descriptorCount = 1;
            write_data.descriptorType  = VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER;
            write_data.pImageInfo      = &sampler_image_info;
            write_data.dstBinding      = 0;
            write_data.dstSet          = m_visibility_read_ds->handle();

            vkUpdateDescriptorSets(m_vk_backend->device(), 1, &write_data, 0, nullptr);
        }

        // Filtered visibility write
        {
            VkWriteDescriptorSet write_data;
            DW_ZERO_MEMORY(write_data);

            VkDescriptorImageInfo storage_image_info;

            storage_image_info.sampler     = VK_NULL_HANDLE;
            storage_image_info.imageView   = m_visibility_filtered_view->handle();
            storage_image_info.imageLayout = VK_IMAGE_LAYOUT_GENERAL;

            write_data.sType           = VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET;
            write_data.descriptorCount = 1;
            write_data.descriptorType  = VK_DESCRIPTOR_TYPE_STORAGE_IMAGE;
            write_data.pImageInfo      = &storage_image_info;
            write_data.dstBinding      = 0;
            write_data.dstSet          = m_visibility_filtered_write_ds->handle();

            vkUpdateDescriptorSets(m_vk_backend->device(), 1, &write_data, 0, nullptr);
        }

        // Filtered visibility read
        {
            VkWriteDescriptorSet write_data;
            DW_ZERO_MEMORY(write_data);

            VkDescriptorImageInfo sampler_image_info;

            sampler_image_info.sampler     = m_bilinear_sampler->handle();
            sampler_image_info.imageView   = m_visibility_filtered_view->handle();
            sampler_image_info.imageLayout = VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL;

            write_data.sType           = VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET;
            write_data.descriptorCount = 1;
            write_data.descriptorType  = VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER;
            write_data.pImageInfo      = &sampler_image_info;
            write_data.dstBinding      = 0;
            write_data.dstSet          = m_visibility_filtered_read_ds->handle();

            vkUpdateDescriptorSets(m_vk_backend->device(), 1, &write_data, 0, nullptr);
        }

        // Reflection write
        //{
        //    VkWriteDescriptorSet write_data[4];
        //    DW_ZERO_MEMORY(write_data[0]);
        //    DW_ZERO_MEMORY(write_data[1]);
        //    DW_ZERO_MEMORY(write_data[2]);
        //    DW_ZERO_MEMORY(write_data[3]);

        //    VkDescriptorImageInfo storage_image_info[2];
        //    VkDescriptorImageInfo sampler_image_info[2];

        //    for (int i = 0; i < 2; i++)
        //    {
        //        storage_image_info[i].sampler     = VK_NULL_HANDLE;
        //        storage_image_info[i].imageView   = m_reflection_view[i]->handle();
        //        storage_image_info[i].imageLayout = VK_IMAGE_LAYOUT_GENERAL;

        //        sampler_image_info[i].sampler     = m_bilinear_sampler->handle();
        //        sampler_image_info[i].imageView   = m_reflection_view[i]->handle();
        //        sampler_image_info[i].imageLayout = VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL;
        //    }

        //    for (int i = 0; i < 2; i++)
        //    {
        //        bool idx = (bool)i;

        //        int write_idx = 2 * i;

        //        write_data[write_idx].sType           = VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET;
        //        write_data[write_idx].descriptorCount = 1;
        //        write_data[write_idx].descriptorType  = VK_DESCRIPTOR_TYPE_STORAGE_IMAGE;
        //        write_data[write_idx].pImageInfo      = &storage_image_info[idx];
        //        write_data[write_idx].dstBinding      = 0;
        //        write_data[write_idx].dstSet          = m_reflection_write_ds[i]->handle();

        //        write_data[write_idx + 1].sType           = VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET;
        //        write_data[write_idx + 1].descriptorCount = 1;
        //        write_data[write_idx + 1].descriptorType  = VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER;
        //        write_data[write_idx + 1].pImageInfo      = &sampler_image_info[!idx];
        //        write_data[write_idx + 1].dstBinding      = 1;
        //        write_data[write_idx + 1].dstSet          = m_reflection_write_ds[i]->handle();
        //    }

        //    vkUpdateDescriptorSets(m_vk_backend->device(), 4, &write_data[0], 0, nullptr);
        //}

        //// Reflection read
        //{
        //    VkWriteDescriptorSet write_data[2];
        //    DW_ZERO_MEMORY(write_data[0]);
        //    DW_ZERO_MEMORY(write_data[1]);

        //    VkDescriptorImageInfo image_info[2];

        //    for (int i = 0; i < 2; i++)
        //    {
        //        image_info[i].sampler     = VK_NULL_HANDLE;
        //        image_info[i].imageView   = m_reflection_view[i]->handle();
        //        image_info[i].imageLayout = VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL;
        //        image_info[i].sampler     = m_bilinear_sampler->handle();

        //        write_data[i].sType           = VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET;
        //        write_data[i].descriptorCount = 1;
        //        write_data[i].descriptorType  = VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER;
        //        write_data[i].pImageInfo      = &image_info[i];
        //        write_data[i].dstBinding      = 0;
        //        write_data[i].dstSet          = m_reflection_read_ds[i]->handle();
        //    }

        //    vkUpdateDescriptorSets(m_vk_backend->device(), 2, &write_data[0], 0, nullptr);
        //}

        // Temp blur
        {
            VkWriteDescriptorSet write_data[2];
            DW_ZERO_MEMORY(write_data[0]);
            DW_ZERO_MEMORY(write_data[1]);

            VkDescriptorImageInfo image_info[2];

            image_info[0].sampler     = VK_NULL_HANDLE;
            image_info[0].imageView   = m_temp_blur_view->handle();
            image_info[0].imageLayout = VK_IMAGE_LAYOUT_GENERAL;

            image_info[1].sampler     = m_bilinear_sampler->handle();
            image_info[1].imageView   = m_temp_blur_view->handle();
            image_info[1].imageLayout = VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL;

            write_data[0].sType           = VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET;
            write_data[0].descriptorCount = 1;
            write_data[0].descriptorType  = VK_DESCRIPTOR_TYPE_STORAGE_IMAGE;
            write_data[0].pImageInfo      = &image_info[0];
            write_data[0].dstBinding      = 0;
            write_data[0].dstSet          = m_temp_blur_write_ds->handle();

            write_data[1].sType           = VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET;
            write_data[1].descriptorCount = 1;
            write_data[1].descriptorType  = VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER;
            write_data[1].pImageInfo      = &image_info[1];
            write_data[1].dstBinding      = 0;
            write_data[1].dstSet          = m_temp_blur_read_ds->handle();

            vkUpdateDescriptorSets(m_vk_backend->device(), 2, &write_data[0], 0, nullptr);
        }
    }

    // -----------------------------------------------------------------------------------------------------------------------------------

    void create_deferred_pipeline()
    {
        dw::vk::PipelineLayout::Desc desc;

        desc.add_descriptor_set_layout(m_g_buffer_ds_layout);
        desc.add_descriptor_set_layout(m_combined_sampler_ds_layout);
        desc.add_descriptor_set_layout(m_per_frame_ds_layout);

        m_deferred_pipeline_layout = dw::vk::PipelineLayout::create(m_vk_backend, desc);
        m_deferred_pipeline        = dw::vk::GraphicsPipeline::create_for_post_process(m_vk_backend, "shaders/triangle.vert.spv", "shaders/deferred.frag.spv", m_deferred_pipeline_layout, m_vk_backend->swapchain_render_pass());
    }

    // -----------------------------------------------------------------------------------------------------------------------------------

    void create_compute_pipeline()
    {
        dw::vk::PipelineLayout::Desc desc;

        desc.add_descriptor_set_layout(m_storage_image_ds_layout);
        desc.add_descriptor_set_layout(m_combined_sampler_ds_layout);
        desc.add_descriptor_set_layout(m_g_buffer_ds_layout);

        desc.add_push_constant_range(VK_SHADER_STAGE_COMPUTE_BIT, 0, sizeof(BilateralBlurPushConstants));

        m_bilateral_blur_pipeline_layout = dw::vk::PipelineLayout::create(m_vk_backend, desc);

        dw::vk::ShaderModule::Ptr module = dw::vk::ShaderModule::create_from_file(m_vk_backend, "shaders/bilateral_blur.comp.spv");

        dw::vk::ComputePipeline::Desc comp_desc;

        comp_desc.set_pipeline_layout(m_bilateral_blur_pipeline_layout);
        comp_desc.set_shader_stage(module, "main");

        m_bilateral_blur_pipeline = dw::vk::ComputePipeline::create(m_vk_backend, comp_desc);
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

        m_shadow_mask_sbt = dw::vk::ShaderBindingTable::create(m_vk_backend, sbt_desc);

        dw::vk::RayTracingPipeline::Desc desc;

        desc.set_recursion_depth(1);
        desc.set_shader_binding_table(m_shadow_mask_sbt);

        // ---------------------------------------------------------------------------
        // Create pipeline layout
        // ---------------------------------------------------------------------------

        dw::vk::PipelineLayout::Desc pl_desc;

        pl_desc.add_descriptor_set_layout(m_acceleration_structure_ds_layout);
        pl_desc.add_descriptor_set_layout(m_storage_image_ds_layout);
        pl_desc.add_descriptor_set_layout(m_combined_sampler_ds_layout);
        pl_desc.add_descriptor_set_layout(m_per_frame_ds_layout);
        pl_desc.add_descriptor_set_layout(m_g_buffer_ds_layout);

        pl_desc.add_push_constant_range(VK_SHADER_STAGE_RAYGEN_BIT_NV, 0, sizeof(ShadowPushConstants));

        m_shadow_mask_pipeline_layout = dw::vk::PipelineLayout::create(m_vk_backend, pl_desc);

        desc.set_pipeline_layout(m_shadow_mask_pipeline_layout);

        m_shadow_mask_pipeline = dw::vk::RayTracingPipeline::create(m_vk_backend, desc);
    }

    // -----------------------------------------------------------------------------------------------------------------------------------

    void create_reflection_ray_tracing_pipeline()
    {
        // ---------------------------------------------------------------------------
        // Create shader modules
        // ---------------------------------------------------------------------------

        dw::vk::ShaderModule::Ptr rgen  = dw::vk::ShaderModule::create_from_file(m_vk_backend, "shaders/reflection.rgen.spv");
        dw::vk::ShaderModule::Ptr rchit = dw::vk::ShaderModule::create_from_file(m_vk_backend, "shaders/reflection.rchit.spv");
        dw::vk::ShaderModule::Ptr rmiss = dw::vk::ShaderModule::create_from_file(m_vk_backend, "shaders/reflection.rmiss.spv");

        dw::vk::ShaderBindingTable::Desc sbt_desc;

        sbt_desc.add_ray_gen_group(rgen, "main");
        sbt_desc.add_hit_group(rchit, "main");
        sbt_desc.add_miss_group(rmiss, "main");

        m_reflection_sbt = dw::vk::ShaderBindingTable::create(m_vk_backend, sbt_desc);

        dw::vk::RayTracingPipeline::Desc desc;

        desc.set_recursion_depth(1);
        desc.set_shader_binding_table(m_reflection_sbt);

        // ---------------------------------------------------------------------------
        // Create pipeline layout
        // ---------------------------------------------------------------------------

        dw::vk::PipelineLayout::Desc pl_desc;

        pl_desc.add_descriptor_set_layout(m_acceleration_structure_ds_layout);
        pl_desc.add_descriptor_set_layout(m_storage_image_ds_layout);
        pl_desc.add_descriptor_set_layout(m_per_frame_ds_layout);
        pl_desc.add_descriptor_set_layout(m_g_buffer_ds_layout);
        pl_desc.add_descriptor_set_layout(m_scene->ray_tracing_geometry_descriptor_set_layout());
        pl_desc.add_descriptor_set_layout(m_scene->material_descriptor_set_layout());
        pl_desc.add_descriptor_set_layout(m_scene->material_descriptor_set_layout());
        pl_desc.add_descriptor_set_layout(m_scene->material_descriptor_set_layout());
        pl_desc.add_descriptor_set_layout(m_scene->material_descriptor_set_layout());

        m_reflection_pipeline_layout = dw::vk::PipelineLayout::create(m_vk_backend, pl_desc);

        desc.set_pipeline_layout(m_reflection_pipeline_layout);

        m_reflection_pipeline = dw::vk::RayTracingPipeline::create(m_vk_backend, desc);
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

        pso_desc.set_vertex_input_state(m_mesh->vertex_input_state_desc());

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
            .add_attachment(blend_att_desc);

        pso_desc.set_color_blend_state(blend_state);

        // ---------------------------------------------------------------------------
        // Create pipeline layout
        // ---------------------------------------------------------------------------

        dw::vk::PipelineLayout::Desc pl_desc;

        pl_desc.add_descriptor_set_layout(m_per_frame_ds_layout)
            .add_descriptor_set_layout(dw::Material::pbr_descriptor_set_layout());

        m_g_buffer_pipeline_layout = dw::vk::PipelineLayout::create(m_vk_backend, pl_desc);

        pso_desc.set_pipeline_layout(m_g_buffer_pipeline_layout);

        // ---------------------------------------------------------------------------
        // Create dynamic state
        // ---------------------------------------------------------------------------

        pso_desc.add_dynamic_state(VK_DYNAMIC_STATE_VIEWPORT)
            .add_dynamic_state(VK_DYNAMIC_STATE_SCISSOR);

        // ---------------------------------------------------------------------------
        // Create pipeline
        // ---------------------------------------------------------------------------

        pso_desc.set_render_pass(m_g_buffer_rp);

        m_g_buffer_pipeline = dw::vk::GraphicsPipeline::create(m_vk_backend, pso_desc);
    }

    // -----------------------------------------------------------------------------------------------------------------------------------

    bool load_mesh()
    {
        m_mesh = dw::Mesh::load(m_vk_backend, "mesh/sponza.obj");
        m_mesh->initialize_for_ray_tracing(m_vk_backend);

        m_scene = dw::Scene::create();
        m_scene->add_instance(m_mesh, glm::mat4(1.0f));

        m_scene->initialize_for_ray_tracing(m_vk_backend);

        return m_mesh != nullptr;
    }

    // -----------------------------------------------------------------------------------------------------------------------------------

    void load_blue_noise()
    {
        m_blue_noise      = dw::vk::Image::create_from_file(m_vk_backend, "texture/LDR_RGBA_0.png");
        m_blue_noise_view = dw::vk::ImageView::create(m_vk_backend, m_blue_noise, VK_IMAGE_VIEW_TYPE_2D, VK_IMAGE_ASPECT_COLOR_BIT);
    }

    // -----------------------------------------------------------------------------------------------------------------------------------

    void create_camera()
    {
        m_main_camera = std::make_unique<dw::Camera>(
            60.0f, 1.0f, 10000.0f, float(m_width) / float(m_height), glm::vec3(0.0f, 35.0f, 125.0f), glm::vec3(0.0f, 0.0, -1.0f));
    }

    // -----------------------------------------------------------------------------------------------------------------------------------

    void check_camera_movement()
    {
        if (m_mouse_look && (m_mouse_delta_x != 0 || m_mouse_delta_y != 0) || m_heading_speed != 0.0f || m_sideways_speed != 0.0f)
            m_camera_moved = true;
        else
            m_camera_moved = false;
    }

    // -----------------------------------------------------------------------------------------------------------------------------------

    void ray_trace_shadow_mask(dw::vk::CommandBuffer::Ptr cmd_buf)
    {
        DW_SCOPED_SAMPLE("ray-tracing-shadows", cmd_buf);

        VkImageSubresourceRange subresource_range = { VK_IMAGE_ASPECT_COLOR_BIT, 0, 1, 0, 1 };

        // Transition ray tracing output image back to general layout
        dw::vk::utilities::set_image_layout(
            cmd_buf->handle(),
            m_visibility_image->handle(),
            VK_IMAGE_LAYOUT_UNDEFINED,
            VK_IMAGE_LAYOUT_GENERAL,
            subresource_range);

        if (m_first_frame)
        {
            dw::vk::utilities::set_image_layout(
                cmd_buf->handle(),
                m_visibility_filtered_image->handle(),
                VK_IMAGE_LAYOUT_UNDEFINED,
                VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL,
                subresource_range);
        }

        auto& rt_props = m_vk_backend->ray_tracing_properties();

        vkCmdBindPipeline(cmd_buf->handle(), VK_PIPELINE_BIND_POINT_RAY_TRACING_NV, m_shadow_mask_pipeline->handle());

        ShadowPushConstants push_constants;

        push_constants.num_frames   = m_num_frames;
        push_constants.alpha        = m_temporal_accumulation ? m_alpha : 1.0f;
        push_constants.light_radius = m_light_radius;

        vkCmdPushConstants(cmd_buf->handle(), m_shadow_mask_pipeline_layout->handle(), VK_SHADER_STAGE_RAYGEN_BIT_NV, 0, sizeof(push_constants), &push_constants);

        const uint32_t dynamic_offset = m_ubo_size * m_vk_backend->current_frame_idx();

        VkDescriptorSet descriptor_sets[] = {
            m_acceleration_structure_ds->handle(),
            m_visibility_write_ds->handle(),
            m_visibility_filtered_read_ds->handle(),
            m_per_frame_ds->handle(),
            m_g_buffer_ds->handle()
        };

        vkCmdBindDescriptorSets(cmd_buf->handle(), VK_PIPELINE_BIND_POINT_RAY_TRACING_NV, m_shadow_mask_pipeline_layout->handle(), 0, 5, descriptor_sets, 1, &dynamic_offset);

        vkCmdTraceRaysNV(cmd_buf->handle(),
                         m_shadow_mask_pipeline->shader_binding_table_buffer()->handle(),
                         0,
                         m_shadow_mask_pipeline->shader_binding_table_buffer()->handle(),
                         m_shadow_mask_sbt->miss_group_offset(),
                         rt_props.shaderGroupBaseAlignment,
                         m_shadow_mask_pipeline->shader_binding_table_buffer()->handle(),
                         m_shadow_mask_sbt->hit_group_offset(),
                         rt_props.shaderGroupBaseAlignment,
                         VK_NULL_HANDLE,
                         0,
                         0,
                         m_width,
                         m_height,
                         1);

        // Prepare ray tracing output image as transfer source
        dw::vk::utilities::set_image_layout(
            cmd_buf->handle(),
            m_visibility_image->handle(),
            VK_IMAGE_LAYOUT_GENERAL,
            VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL,
            subresource_range);
    }

    // -----------------------------------------------------------------------------------------------------------------------------------

    void ray_trace_reflection(dw::vk::CommandBuffer::Ptr cmd_buf)
    {
        DW_SCOPED_SAMPLE("ray-tracing-reflections", cmd_buf);

        VkImageSubresourceRange subresource_range = { VK_IMAGE_ASPECT_COLOR_BIT, 0, 1, 0, 1 };

        // Transition ray tracing output image back to general layout
        dw::vk::utilities::set_image_layout(
            cmd_buf->handle(),
            m_reflection_image[m_ping_pong]->handle(),
            VK_IMAGE_LAYOUT_UNDEFINED,
            VK_IMAGE_LAYOUT_GENERAL,
            subresource_range);

        if (m_first_frame)
        {
            dw::vk::utilities::set_image_layout(
                cmd_buf->handle(),
                m_reflection_image[!m_ping_pong]->handle(),
                VK_IMAGE_LAYOUT_UNDEFINED,
                VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL,
                subresource_range);
        }

        auto& rt_props = m_vk_backend->ray_tracing_properties();

        vkCmdBindPipeline(cmd_buf->handle(), VK_PIPELINE_BIND_POINT_RAY_TRACING_NV, m_reflection_pipeline->handle());

        const uint32_t dynamic_offset = m_ubo_size * m_vk_backend->current_frame_idx();

        VkDescriptorSet descriptor_sets[] = {
            m_acceleration_structure_ds->handle(),
            m_reflection_write_ds[m_ping_pong]->handle(),
            m_per_frame_ds->handle(),
            m_g_buffer_ds->handle(),
            m_scene->ray_tracing_geometry_descriptor_set()->handle(),
            m_scene->albedo_descriptor_set()->handle(),
            m_scene->normal_descriptor_set()->handle(),
            m_scene->roughness_descriptor_set()->handle(),
            m_scene->metallic_descriptor_set()->handle()
        };

        vkCmdBindDescriptorSets(cmd_buf->handle(), VK_PIPELINE_BIND_POINT_RAY_TRACING_NV, m_reflection_pipeline_layout->handle(), 0, 9, descriptor_sets, 1, &dynamic_offset);

        vkCmdTraceRaysNV(cmd_buf->handle(),
                         m_reflection_pipeline->shader_binding_table_buffer()->handle(),
                         0,
                         m_reflection_pipeline->shader_binding_table_buffer()->handle(),
                         m_reflection_sbt->miss_group_offset(),
                         rt_props.shaderGroupBaseAlignment,
                         m_reflection_pipeline->shader_binding_table_buffer()->handle(),
                         m_reflection_sbt->hit_group_offset(),
                         rt_props.shaderGroupBaseAlignment,
                         VK_NULL_HANDLE,
                         0,
                         0,
                         m_width,
                         m_height,
                         1);

        // Prepare ray tracing output image as transfer source
        dw::vk::utilities::set_image_layout(
            cmd_buf->handle(),
            m_reflection_image[m_ping_pong]->handle(),
            VK_IMAGE_LAYOUT_GENERAL,
            VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL,
            subresource_range);
    }

    // -----------------------------------------------------------------------------------------------------------------------------------

    void render_gbuffer(dw::vk::CommandBuffer::Ptr cmd_buf)
    {
        DW_SCOPED_SAMPLE("render_gbuffer", cmd_buf);

        VkClearValue clear_values[4];

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

        clear_values[3].color.float32[0] = 1.0f;
        clear_values[3].color.float32[1] = 1.0f;
        clear_values[3].color.float32[2] = 1.0f;
        clear_values[3].color.float32[3] = 1.0f;

        VkRenderPassBeginInfo info    = {};
        info.sType                    = VK_STRUCTURE_TYPE_RENDER_PASS_BEGIN_INFO;
        info.renderPass               = m_g_buffer_rp->handle();
        info.framebuffer              = m_g_buffer_fbo->handle();
        info.renderArea.extent.width  = m_width;
        info.renderArea.extent.height = m_height;
        info.clearValueCount          = 4;
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

        vkCmdBindPipeline(cmd_buf->handle(), VK_PIPELINE_BIND_POINT_GRAPHICS, m_g_buffer_pipeline->handle());

        VkDeviceSize offset = 0;
        vkCmdBindVertexBuffers(cmd_buf->handle(), 0, 1, &m_mesh->vertex_buffer()->handle(), &offset);
        vkCmdBindIndexBuffer(cmd_buf->handle(), m_mesh->index_buffer()->handle(), 0, VK_INDEX_TYPE_UINT32);

        const uint32_t dynamic_offset = m_ubo_size * m_vk_backend->current_frame_idx();

        vkCmdBindDescriptorSets(cmd_buf->handle(), VK_PIPELINE_BIND_POINT_GRAPHICS, m_g_buffer_pipeline_layout->handle(), 0, 1, &m_per_frame_ds->handle(), 1, &dynamic_offset);

        for (uint32_t i = 0; i < m_mesh->sub_mesh_count(); i++)
        {
            auto& submesh = m_mesh->sub_meshes()[i];
            auto& mat     = m_mesh->material(submesh.mat_idx);

            if (mat->pbr_descriptor_set())
                vkCmdBindDescriptorSets(cmd_buf->handle(), VK_PIPELINE_BIND_POINT_GRAPHICS, m_g_buffer_pipeline_layout->handle(), 1, 1, &mat->pbr_descriptor_set()->handle(), 0, nullptr);

            // Issue draw call.
            vkCmdDrawIndexed(cmd_buf->handle(), submesh.index_count, 1, submesh.base_index, submesh.base_vertex, 0);
        }

        vkCmdEndRenderPass(cmd_buf->handle());
    }

    // -----------------------------------------------------------------------------------------------------------------------------------

    void bilateral_blur_shadows(dw::vk::CommandBuffer::Ptr cmd_buf)
    {
        const uint32_t NUM_THREADS = 32;

        VkImageSubresourceRange subresource_range = { VK_IMAGE_ASPECT_COLOR_BIT, 0, 1, 0, 1 };

        dw::vk::utilities::set_image_layout(
            cmd_buf->handle(),
            m_temp_blur->handle(),
            VK_IMAGE_LAYOUT_UNDEFINED,
            VK_IMAGE_LAYOUT_GENERAL,
            subresource_range);

        BilateralBlurPushConstants push_constants;

        float z_buffer_params_x = -1.0 + (m_main_camera->m_near / m_main_camera->m_far);

        push_constants.z_buffer_params = glm::vec4(z_buffer_params_x, 1.0f, z_buffer_params_x / m_main_camera->m_near, 1.0f / m_main_camera->m_near);
        push_constants.pixel_size      = glm::vec2(1.0f / float(m_width), 1.0f / float(m_height));

        {
            DW_SCOPED_SAMPLE("Bilateral Blur Horizontal", cmd_buf);

            vkCmdBindPipeline(cmd_buf->handle(), VK_PIPELINE_BIND_POINT_COMPUTE, m_bilateral_blur_pipeline->handle());

            push_constants.direction = glm::vec2(1.0f, 0.0f);
            vkCmdPushConstants(cmd_buf->handle(), m_bilateral_blur_pipeline_layout->handle(), VK_SHADER_STAGE_COMPUTE_BIT, 0, sizeof(push_constants), &push_constants);

            VkDescriptorSet descriptor_sets[] = {
                m_temp_blur_write_ds->handle(),
                m_visibility_read_ds->handle(),
                m_g_buffer_ds->handle()
            };

            vkCmdBindDescriptorSets(cmd_buf->handle(), VK_PIPELINE_BIND_POINT_COMPUTE, m_bilateral_blur_pipeline_layout->handle(), 0, 3, descriptor_sets, 0, nullptr);

            vkCmdDispatch(cmd_buf->handle(), m_width / NUM_THREADS, m_height / NUM_THREADS, 1);
        }

        dw::vk::utilities::set_image_layout(
            cmd_buf->handle(),
            m_visibility_filtered_image->handle(),
            VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL,
            VK_IMAGE_LAYOUT_GENERAL,
            subresource_range);

        dw::vk::utilities::set_image_layout(
            cmd_buf->handle(),
            m_temp_blur->handle(),
            VK_IMAGE_LAYOUT_GENERAL,
            VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL,
            subresource_range);

        {
            DW_SCOPED_SAMPLE("Bilateral Blur Vertical", cmd_buf);

            vkCmdBindPipeline(cmd_buf->handle(), VK_PIPELINE_BIND_POINT_COMPUTE, m_bilateral_blur_pipeline->handle());

            push_constants.direction = glm::vec2(0.0f, 1.0f);
            vkCmdPushConstants(cmd_buf->handle(), m_bilateral_blur_pipeline_layout->handle(), VK_SHADER_STAGE_COMPUTE_BIT, 0, sizeof(push_constants), &push_constants);

            VkDescriptorSet descriptor_sets[] = {
                m_visibility_filtered_write_ds->handle(),
                m_temp_blur_read_ds->handle(),
                m_g_buffer_ds->handle()
            };

            vkCmdBindDescriptorSets(cmd_buf->handle(), VK_PIPELINE_BIND_POINT_COMPUTE, m_bilateral_blur_pipeline_layout->handle(), 0, 3, descriptor_sets, 0, nullptr);

            vkCmdDispatch(cmd_buf->handle(), m_width / NUM_THREADS, m_height / NUM_THREADS, 1);
        }

        dw::vk::utilities::set_image_layout(
            cmd_buf->handle(),
            m_visibility_filtered_image->handle(),
            VK_IMAGE_LAYOUT_GENERAL,
            VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL,
            subresource_range);
    }

    // -----------------------------------------------------------------------------------------------------------------------------------

    void render(dw::vk::CommandBuffer::Ptr cmd_buf)
    {
        DW_SCOPED_SAMPLE("deferred", cmd_buf);

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

        vkCmdBindPipeline(cmd_buf->handle(), VK_PIPELINE_BIND_POINT_GRAPHICS, m_deferred_pipeline->handle());

        const uint32_t dynamic_offset = m_ubo_size * m_vk_backend->current_frame_idx();

        VkDescriptorSet descriptor_sets[] = {
            m_g_buffer_ds->handle(),
            m_visibility_filtered_read_ds->handle(),
            m_per_frame_ds->handle()
        };

        vkCmdBindDescriptorSets(cmd_buf->handle(), VK_PIPELINE_BIND_POINT_GRAPHICS, m_deferred_pipeline_layout->handle(), 0, 3, descriptor_sets, 1, &dynamic_offset);

        vkCmdDraw(cmd_buf->handle(), 3, 1, 0, 0);

        render_gui(cmd_buf);

        vkCmdEndRenderPass(cmd_buf->handle());
    }

    // -----------------------------------------------------------------------------------------------------------------------------------

    void update_uniforms(dw::vk::CommandBuffer::Ptr cmd_buf)
    {
        DW_SCOPED_SAMPLE("update_uniforms", cmd_buf);

        if (m_first_frame)
            m_prev_view_proj = m_main_camera->m_view_projection;

        m_transforms.proj_inverse      = glm::inverse(m_main_camera->m_projection);
        m_transforms.view_inverse      = glm::inverse(m_main_camera->m_view);
        m_transforms.view_proj_inverse = glm::inverse(m_main_camera->m_projection * m_main_camera->m_view);
        m_transforms.prev_view_proj    = m_prev_view_proj;
        m_transforms.proj              = m_main_camera->m_projection;
        m_transforms.view              = m_main_camera->m_view;
        m_transforms.model             = glm::mat4(1.0f);
        m_transforms.cam_pos           = glm::vec4(m_main_camera->m_position, 0.0f);
        m_transforms.light_dir         = glm::vec4(m_light_direction, 0.0f);

        m_prev_view_proj = m_main_camera->m_view_projection;

        uint8_t* ptr = (uint8_t*)m_ubo->mapped_ptr();
        memcpy(ptr + m_ubo_size * m_vk_backend->current_frame_idx(), &m_transforms, sizeof(Transforms));
    }

    // -----------------------------------------------------------------------------------------------------------------------------------

    void update_camera()
    {
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

private:
    // GPU resources.
    size_t m_ubo_size;

    // Common
    dw::vk::DescriptorSet::Ptr       m_per_frame_ds;
    dw::vk::DescriptorSetLayout::Ptr m_per_frame_ds_layout;
    dw::vk::DescriptorSet::Ptr       m_g_buffer_ds;
    dw::vk::DescriptorSetLayout::Ptr m_g_buffer_ds_layout;
    dw::vk::Buffer::Ptr              m_ubo;
    dw::vk::Image::Ptr               m_blue_noise;
    dw::vk::ImageView::Ptr           m_blue_noise_view;

    dw::vk::DescriptorSetLayout::Ptr m_combined_sampler_ds_layout;
    dw::vk::DescriptorSetLayout::Ptr m_acceleration_structure_ds_layout;
    dw::vk::DescriptorSetLayout::Ptr m_storage_image_ds_layout;

    dw::vk::DescriptorSet::Ptr m_acceleration_structure_ds;
    dw::vk::Sampler::Ptr       m_bilinear_sampler;

    // Shadow mask pass
    dw::vk::DescriptorSet::Ptr m_visibility_write_ds;
    dw::vk::DescriptorSet::Ptr m_visibility_read_ds;
    dw::vk::DescriptorSet::Ptr m_visibility_filtered_read_ds;
    dw::vk::DescriptorSet::Ptr m_visibility_filtered_write_ds;

    dw::vk::RayTracingPipeline::Ptr m_shadow_mask_pipeline;
    dw::vk::PipelineLayout::Ptr     m_shadow_mask_pipeline_layout;
    dw::vk::ShaderBindingTable::Ptr m_shadow_mask_sbt;
    dw::vk::Image::Ptr              m_visibility_image;
    dw::vk::ImageView::Ptr          m_visibility_view;
    dw::vk::Image::Ptr              m_visibility_filtered_image;
    dw::vk::ImageView::Ptr          m_visibility_filtered_view;

    // Reflection pass
    dw::vk::DescriptorSet::Ptr      m_reflection_write_ds[2];
    dw::vk::DescriptorSet::Ptr      m_reflection_read_ds[2];
    dw::vk::RayTracingPipeline::Ptr m_reflection_pipeline;
    dw::vk::PipelineLayout::Ptr     m_reflection_pipeline_layout;
    dw::vk::Image::Ptr              m_reflection_image[2];
    dw::vk::ImageView::Ptr          m_reflection_view[2];
    dw::vk::ShaderBindingTable::Ptr m_reflection_sbt;

    // Deferred pass
    dw::vk::GraphicsPipeline::Ptr m_deferred_pipeline;
    dw::vk::PipelineLayout::Ptr   m_deferred_pipeline_layout;

    // G-Buffer pass
    dw::vk::Image::Ptr            m_g_buffer_1; // RGB: Albedo, A: Metallic
    dw::vk::Image::Ptr            m_g_buffer_2; // RGB: Normal, A: Roughness
    dw::vk::Image::Ptr            m_g_buffer_3; // RGB: Position, A: -
    dw::vk::Image::Ptr            m_g_buffer_depth;
    dw::vk::ImageView::Ptr        m_g_buffer_1_view;
    dw::vk::ImageView::Ptr        m_g_buffer_2_view;
    dw::vk::ImageView::Ptr        m_g_buffer_3_view;
    dw::vk::ImageView::Ptr        m_g_buffer_depth_view;
    dw::vk::Framebuffer::Ptr      m_g_buffer_fbo;
    dw::vk::RenderPass::Ptr       m_g_buffer_rp;
    dw::vk::GraphicsPipeline::Ptr m_g_buffer_pipeline;
    dw::vk::PipelineLayout::Ptr   m_g_buffer_pipeline_layout;

    // Bilateral Blur
    dw::vk::DescriptorSet::Ptr   m_temp_blur_write_ds;
    dw::vk::DescriptorSet::Ptr   m_temp_blur_read_ds;
    dw::vk::Image::Ptr           m_temp_blur;
    dw::vk::ImageView::Ptr       m_temp_blur_view;
    dw::vk::ComputePipeline::Ptr m_bilateral_blur_pipeline;
    dw::vk::PipelineLayout::Ptr  m_bilateral_blur_pipeline_layout;

    // Camera.
    std::unique_ptr<dw::Camera> m_main_camera;
    glm::mat4                   m_prev_view_proj;

    // Camera controls.
    bool      m_mouse_look         = false;
    float     m_heading_speed      = 0.0f;
    float     m_sideways_speed     = 0.0f;
    float     m_camera_sensitivity = 0.05f;
    float     m_camera_speed       = 0.2f;
    float     m_offset             = 0.1f;
    bool      m_debug_gui          = false;
    glm::vec3 m_light_direction    = glm::vec3(0.0f);

    // Camera orientation.
    float m_camera_x;
    float m_camera_y;

    // Assets.
    dw::Mesh::Ptr  m_mesh;
    dw::Scene::Ptr m_scene;

    // Path Tracer
    bool    m_path_trace_mode       = false;
    bool    m_ping_pong             = false;
    bool    m_camera_moved          = false;
    bool    m_first_frame           = true;
    bool    m_use_bilateral_blur    = true;
    bool    m_temporal_accumulation = true;
    int32_t m_num_frames            = 0;
    float   m_light_radius          = 0.01f;
    float   m_alpha                 = 0.15f;
    int32_t m_max_samples           = 10000;

    // Uniforms.
    Transforms m_transforms;
};

DW_DECLARE_MAIN(Sample)
