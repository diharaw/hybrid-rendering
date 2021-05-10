#include "reflection_denoiser.h"
#include "spatial_reconstruction.h"
#include "temporal_reprojection.h"
#include "bilateral_blur.h"
#include "g_buffer.h"
#include "common_resources.h"
#include "utilities.h"
#include <macros.h>
#include <profiler.h>
#include <imgui.h>

ReflectionDenoiser::ReflectionDenoiser(std::weak_ptr<dw::vk::Backend> backend, CommonResources* common_resources, GBuffer* g_buffer, std::string name, uint32_t input_width, uint32_t input_height) :
    m_name(name), m_common_resources(common_resources), m_input_width(input_width), m_input_height(input_height)
{
    m_spatial_reconstruction = std::unique_ptr<SpatialReconstruction>(new SpatialReconstruction(backend, common_resources, g_buffer, name, input_width, input_height));
    m_temporal_pre_pass      = std::unique_ptr<TemporalReprojection>(new TemporalReprojection(backend, common_resources, g_buffer, name, input_width * 2, input_height * 2));
    m_temporal_main_pass     = std::unique_ptr<TemporalReprojection>(new TemporalReprojection(backend, common_resources, g_buffer, name, input_width * 2, input_height * 2));
    m_bilateral_blur         = std::unique_ptr<BilateralBlur>(new BilateralBlur(backend, common_resources, g_buffer, name, input_width * 2, input_height * 2));

    m_use_blur_as_temporal_input = true;
    m_temporal_pre_pass->set_variance_clipping(true);
    m_temporal_pre_pass->set_neighborhood_scale(3.5f);
    m_temporal_pre_pass->set_alpha(0.05f);
    m_temporal_main_pass->set_variance_clipping(true);
    m_bilateral_blur->set_blur_radius(1);
}

ReflectionDenoiser::~ReflectionDenoiser()
{
}

void ReflectionDenoiser::denoise(dw::vk::CommandBuffer::Ptr cmd_buf, dw::vk::DescriptorSet::Ptr input)
{
    if (m_common_resources->first_frame)
        m_bilateral_blur->prepare_first_frame(cmd_buf);

    m_spatial_reconstruction->reconstruct(cmd_buf, input);

    if (m_use_temporal_pre_pass)
        m_temporal_pre_pass->reproject(cmd_buf, m_spatial_reconstruction->output_ds(), (m_use_blur_as_temporal_input && m_use_bilateral_blur) ? m_bilateral_blur->output_ds() : nullptr);

    m_temporal_main_pass->reproject(cmd_buf, m_use_temporal_pre_pass ? m_temporal_pre_pass->output_ds() : m_spatial_reconstruction->output_ds(), m_use_blur_as_temporal_input ? m_bilateral_blur->output_ds() : nullptr);

    if (m_use_bilateral_blur)
        m_bilateral_blur->blur(cmd_buf, m_temporal_main_pass->output_ds());
}

void ReflectionDenoiser::gui()
{
    ImGui::Checkbox("Use Blur as Temporal Input", &m_use_blur_as_temporal_input);
    {
        //ImGui::Text("Spatial Reconstruction");
        //m_spatial_reconstruction->gui();
    } 
    {
        ImGui::PushID("TemporalPrePass");
        ImGui::Separator();
        ImGui::Text("Temporal Pre Pass");
        ImGui::Checkbox("Enable", &m_use_temporal_pre_pass);
        m_temporal_pre_pass->gui();
        ImGui::PopID();
    }
    {
        ImGui::PushID("TemporalMainPass");
        ImGui::Separator();
        ImGui::Text("Temporal Main Pass");
        m_temporal_main_pass->gui();
        ImGui::PopID();
    }
    {
        ImGui::PushID("BilateralBlurPass");
        ImGui::Separator();
        ImGui::Text("Bilateral Blur");
        ImGui::Checkbox("Enable", &m_use_bilateral_blur);
        m_bilateral_blur->gui();
        ImGui::PopID();
    }
}

dw::vk::DescriptorSet::Ptr ReflectionDenoiser::output_ds()
{
    if (m_use_bilateral_blur)
        return m_bilateral_blur->output_ds();
    else
        return m_temporal_main_pass->output_ds();
}