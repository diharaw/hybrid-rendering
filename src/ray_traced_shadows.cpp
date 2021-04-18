#include "ray_traced_shadows.h"
#include "common_resources.h"
#include "g_buffer.h"
#include <profiler.h>
#include <macros.h>

RayTracedShadows::RayTracedShadows(std::weak_ptr<dw::vk::Backend> backend, CommonResources* common_resources, GBuffer* g_buffer, uint32_t width, uint32_t height) :
    m_backend(backend), m_common_resources(common_resources), m_g_buffer(g_buffer), m_width(width), m_height(height)
{
    create_images();
    create_descriptor_sets();
    write_descriptor_sets();
    create_pipeline();
}

RayTracedShadows::~RayTracedShadows()
{

}

void RayTracedShadows::create_images()
{

}

void RayTracedShadows::create_descriptor_sets()
{

}

void RayTracedShadows::write_descriptor_sets()
{

}

void RayTracedShadows::create_pipeline()
{

}