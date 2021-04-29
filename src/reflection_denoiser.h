#pragma once

#include <vk.h>
#include <glm.hpp>

class CommonResources;
class GBuffer;
class SpatialReconstruction;
class TemporalReprojection;
class BilateralBlur;

class ReflectionDenoiser
{
public:
    ReflectionDenoiser(std::weak_ptr<dw::vk::Backend> backend, CommonResources* common_resources, GBuffer* g_buffer, std::string name, uint32_t input_width, uint32_t input_height);
    ~ReflectionDenoiser();

    void                       denoise(dw::vk::CommandBuffer::Ptr cmd_buf, dw::vk::DescriptorSet::Ptr input);
    void                       gui();
    dw::vk::DescriptorSet::Ptr output_ds();

    inline bool temporal_pre_pass() { return m_use_temporal_pre_pass; }
    inline void set_temporal_pre_pass(bool value) { m_use_temporal_pre_pass = value; }

private:
    std::string      m_name;
    CommonResources* m_common_resources;
    uint32_t         m_input_width;
    uint32_t         m_input_height;
    bool             m_use_temporal_pre_pass      = true;
    bool             m_use_blur_as_temporal_input = false;
    bool             m_use_bilateral_blur         = true;

    // Reconstruction
    std::unique_ptr<SpatialReconstruction> m_spatial_reconstruction;

    // Temporal
    std::unique_ptr<TemporalReprojection> m_temporal_pre_pass;
    std::unique_ptr<TemporalReprojection> m_temporal_main_pass;

    // Bilateral Blur
    std::unique_ptr<BilateralBlur> m_bilateral_blur;
};
