#include "common.h"
#include <logger.h>
#include <stdexcept>
#include <gtc/matrix_transform.hpp>
#include <equirectangular_to_cubemap.h>

// -----------------------------------------------------------------------------------------------------------------------------------

namespace constants
{
const std::vector<std::string>            environment_map_images        = { "textures/Arches_E_PineTree_3k.hdr", "textures/BasketballCourt_3k.hdr", "textures/Etnies_Park_Center_3k.hdr", "textures/LA_Downtown_Helipad_GoldenHour_3k.hdr" };
const std::vector<std::string>            environment_types             = { "None", "Procedural Sky", "Arches Pine Tree", "Basketball Court", "Etnies Park Central", "LA Downtown Helipad" };
const std::vector<std::string>            visualization_types           = { "Final", "Shadows", "Ambient Occlusion", "Reflections", "Global Illumination", "Ground Truth" };
const std::vector<std::string>            scene_types                   = { "Shadows Test", "Reflections Test", "Global Illumination Test", "Pica Pica", "Sponza" };
const std::vector<std::string>            ray_trace_scales              = { "Full-Res", "Half-Res", "Quarter-Res" };
const std::vector<std::string>            light_types                   = { "Directional", "Point", "Spot" };
const std::vector<std::string>            camera_types                  = { "Free", "Animated", "Fixed" };
const std::vector<std::vector<glm::vec3>> fixed_camera_position_vectors = {
    { glm::vec3(-22.061460f, 16.624475f, 23.893597f),
      glm::vec3(-0.337131f, 15.421529f, 39.524925f),
      glm::vec3(9.907501f, 8.313064f, -18.302629f),
      glm::vec3(10.431265f, 4.411600f, -6.578662f) },

    { glm::vec3(-42.599087f, 5.077470f, 3.662686f),
      glm::vec3(42.569202f, 5.680231f, 1.135333f),
      glm::vec3(-0.088592f, 18.299492f, 31.712112f),
      glm::vec3(-10.971692f, 4.040000f, -1.099626f) },

    { glm::vec3(5.589866f, 5.565732f, 8.097699f),
      glm::vec3(-8.616280f, 4.837910f, 9.911291f),
      glm::vec3(10.316762f, 5.566028f, 1.504511f),
      glm::vec3(11.364138f, 9.322021f, -6.157114f) },

    { glm::vec3(-2.880592f, 12.838152f, 28.133095f),
      glm::vec3(-4.044456f, 3.885819f, 14.471013f),
      glm::vec3(-10.408246f, 4.111171f, 8.519235f),
      glm::vec3(-10.283543f, 6.659785f, 2.117568f) },

    { glm::vec3(349.689911f, 50.013187f, -47.142761f),
      glm::vec3(255.940018f, 181.126541f, -14.534848f),
      glm::vec3(25.954714f, 36.763203f, 153.194244f),
      glm::vec3(-391.294556f, 179.648758f, 141.655914f) }
};
const std::vector<std::vector<glm::vec3>> fixed_camera_forward_vectors = {
    { glm::vec3(0.593151f, -0.521760f, -0.613138f),
      glm::vec3(-0.006306f, -0.425798f, -0.904796f),
      glm::vec3(-0.353051f, -0.351048f, 0.867249f),
      glm::vec3(-0.800752f, -0.151261f, 0.579584f) },

    { glm::vec3(0.926363f, -0.233447f, -0.295558f),
      glm::vec3(-0.956285f, -0.235149f, -0.173853f),
      glm::vec3(0.003158f, -0.457108f, -0.889406f),
      glm::vec3(-0.593590f, -0.287377f, -0.751709f) },

    { glm::vec3(-0.747366f, -0.139174f, -0.649672f),
      glm::vec3(0.361111f, -0.127066f, -0.923825f),
      glm::vec3(-0.994886f, -0.098450f, -0.022578f),
      glm::vec3(-0.760210f, -0.417866f, 0.497463f) },

    { glm::vec3(-0.005560f, -0.393157f, -0.919454f),
      glm::vec3(0.725216f, -0.146966f, -0.672653f),
      glm::vec3(-0.739586f, -0.270623f, -0.616259f),
      glm::vec3(0.787594f, -0.314029f, -0.530172f) },

    { glm::vec3(-0.927807f, -0.008728f, 0.372960f),
      glm::vec3(-0.890209f, -0.455542f, -0.003118f),
      glm::vec3(0.932927f, -0.008722f, -0.359960f),
      glm::vec3(0.723851f, -0.095842f, -0.683267f) }
};
const std::vector<std::vector<glm::vec3>> fixed_camera_right_vectors = {
    { glm::vec3(0.718724f, -0.000000f, 0.695295f),
      glm::vec3(0.999976f, 0.000000f, -0.006970f),
      glm::vec3(-0.926194f, 0.000000f, -0.377048f),
      glm::vec3(-0.586330f, 0.000000f, -0.810072f) },

    { glm::vec3(0.303957f, -0.000000f, 0.952686f),
      glm::vec3(0.178869f, 0.000000f, -0.983873f),
      glm::vec3(0.999994f, -0.000000f, 0.003551f),
      glm::vec3(0.784814f, 0.000000f, -0.619732f) },

    { glm::vec3(0.656057f, 0.000000f, -0.754711f),
      glm::vec3(0.931375f, -0.000000f, 0.364062f),
      glm::vec3(0.022688f, 0.000000f, -0.999743f),
      glm::vec3(-0.547560f, 0.000000f, -0.836766f) },

    { glm::vec3(0.999982f, 0.000000f, -0.006047f),
      glm::vec3(0.680037f, -0.000000f, 0.733178f),
      glm::vec3(0.640146f, 0.000000f, -0.768253f),
      glm::vec3(0.558420f, -0.000000f, 0.829558f) },

    { glm::vec3(-0.372974f, 0.000000f, -0.927842f),
      glm::vec3(0.003502f, 0.000000f, -0.999994f),
      glm::vec3(0.359974f, -0.000000f, 0.932963f),
      glm::vec3(0.686427f, -0.000000f, 0.727199f) }
};
const std::vector<std::vector<glm::vec3>> animated_camera_position_vectors = {
    { glm::vec3(-2.053485, 17.298836, 30.651987),
      glm::vec3(-17.733454, 17.524971, 19.784597),
      glm::vec3(-23.405531, 17.524971, -2.044511),
      glm::vec3(-9.641323, 17.524971, -19.843979),
      glm::vec3(9.528142, 17.524971, -22.101870),
      glm::vec3(22.545725, 18.766420, -11.367293),
      glm::vec3(18.993521, 18.766420, 14.301329),
      glm::vec3(1.044318, 10.293690, 31.108456),
      glm::vec3(1.055899, 6.021045, 21.854504),
      glm::vec3(6.531604, 4.438575, 12.356213),
      glm::vec3(24.359325, 3.234370, 8.643531),
      glm::vec3(29.245045, 7.622394, -2.224115),
      glm::vec3(24.680267, 11.480971, -31.513523),
      glm::vec3(-0.891728, 10.484192, -33.953403),
      glm::vec3(-24.790842, 10.846797, -25.491060),
      glm::vec3(-29.894993, 10.846797, -3.001314),
      glm::vec3(-23.293041, 12.024170, 29.659746),
      glm::vec3(2.173189, 9.871284, 37.758049) },
    { glm::vec3(-42.047676, 5.609881, 2.562008),
      glm::vec3(-22.662962, 6.125844, 9.230613),
      glm::vec3(0.765859, 6.125844, 10.087609),
      glm::vec3(18.091362, 6.125844, 9.800380),
      glm::vec3(40.892811, 6.125844, 2.576573) },
    { glm::vec3(9.634456, 5.721654, 8.176250),
      glm::vec3(0.835031, 4.663226, 6.983876),
      glm::vec3(-9.140702, 4.568583, 8.695939),
      glm::vec3(-4.281561, 6.994284, 5.569273),
      glm::vec3(3.310796, 6.340317, -0.976028),
      glm::vec3(9.808627, 7.746489, -2.045464),
      glm::vec3(10.834608, 6.933594, 8.102142) },
    { glm::vec3(-15.785997, 11.891207, 24.087767),
      glm::vec3(-19.417524, 6.099357, 11.969102),
      glm::vec3(-11.975905, 5.179130, 0.234051),
      glm::vec3(1.423192, 11.926380, 9.830714),
      glm::vec3(16.197201, 19.097288, 26.328077),
      glm::vec3(-3.237705, 24.273561, 39.714752),
      glm::vec3(-14.591802, 8.839928, 20.456472) },
    { glm::vec3(349.886139, 44.537258, -29.406055),
      glm::vec3(159.558609, 40.026180, -8.913517),
      glm::vec3(-166.946274, 39.137112, -9.046158),
      glm::vec3(-393.339844, 37.502037, -18.350918),
      glm::vec3(-399.373962, 60.521072, -85.331512),
      glm::vec3(-231.727509, 47.810944, -155.838043),
      glm::vec3(208.506546, 53.873413, -176.414337),
      glm::vec3(272.163635, 77.822922, -182.078278),
      glm::vec3(364.375824, 68.312172, -99.257278),
      glm::vec3(360.336212, 61.498547, -4.966379),
      glm::vec3(257.223083, 76.369492, 6.302059),
      glm::vec3(182.961060, 164.536682, -6.476255),
      glm::vec3(81.959976, 180.288940, -41.259853),
      glm::vec3(65.144699, 177.133667, -146.537628),
      glm::vec3(21.156075, 168.451660, -152.319733),
      glm::vec3(-129.477036, 168.355881, -149.554825),
      glm::vec3(-246.375076, 168.495789, -165.307434),
      glm::vec3(-397.199646, 172.113068, -73.152016),
      glm::vec3(-389.553802, 172.113068, 10.852755),
      glm::vec3(-272.403931, 178.293121, 11.517756),
      glm::vec3(-108.214783, 107.747192, -6.823005),
      glm::vec3(64.845596, 58.646214, -20.116652),
      glm::vec3(182.857437, 44.446678, -57.136696),
      glm::vec3(285.696503, 36.827293, -43.684582),
      glm::vec3(338.671600, 35.484882, -21.409927),
      glm::vec3(335.790222, 35.701035, 56.135605) }
};
const std::vector<std::vector<glm::vec3>> animated_camera_forward_vectors = {
    { glm::vec3(0.055552, -0.414693, -0.908264),
      glm::vec3(0.603634, -0.541709, -0.584959),
      glm::vec3(0.847531, -0.530659, -0.009613),
      glm::vec3(0.405785, -0.584250, 0.702844),
      glm::vec3(-0.179740, -0.605295, 0.775444),
      glm::vec3(-0.699132, -0.614976, 0.364719),
      glm::vec3(-0.613705, -0.626606, -0.480345),
      glm::vec3(0.010284, -0.422622, -0.906248),
      glm::vec3(-0.026531, -0.194238, -0.980596),
      glm::vec3(-0.531659, -0.134855, -0.836154),
      glm::vec3(-0.992098, -0.088898, -0.088541),
      glm::vec3(-0.972391, -0.217294, 0.085080),
      glm::vec3(-0.528165, -0.410722, 0.743202),
      glm::vec3(0.032904, -0.392341, 0.919231),
      glm::vec3(0.749475, -0.397152, 0.529677),
      glm::vec3(0.910138, -0.413902, -0.018271),
      glm::vec3(0.539066, -0.424993, -0.727179),
      glm::vec3(-0.055169, -0.329695, -0.942474) },
    { glm::vec3(0.671119, -0.241076, -0.701057),
      glm::vec3(0.002524, -0.270601, -0.962688),
      glm::vec3(-0.016052, -0.249536, -0.968232),
      glm::vec3(-0.016052, -0.249536, -0.968232),
      glm::vec3(-0.702000, -0.272282, -0.658072) },
    { glm::vec3(-0.826647, -0.223250, -0.516541),
      glm::vec3(-0.605107, -0.119271, -0.787159),
      glm::vec3(-0.055601, -0.089764, -0.994410),
      glm::vec3(-0.467805, -0.268080, -0.842195),
      glm::vec3(-0.949390, -0.180521, -0.257042),
      glm::vec3(-0.952666, -0.282343, -0.112741),
      glm::vec3(-0.903509, -0.196804, -0.380710) },
    { glm::vec3(0.335764, -0.371370, -0.865648),
      glm::vec3(0.688170, -0.233449, -0.686967),
      glm::vec3(0.928188, -0.094112, -0.360013),
      glm::vec3(0.129433, -0.381881, -0.915103),
      glm::vec3(-0.313897, -0.528443, -0.788807),
      glm::vec3(0.019821, -0.487105, -0.873118),
      glm::vec3(0.460709, -0.292381, -0.838010) },
    { glm::vec3(-0.947293, -0.094977, 0.305966),
      glm::vec3(-0.999099, -0.006981, 0.041873),
      glm::vec3(-0.998480, 0.004363, 0.054948),
      glm::vec3(0.430484, -0.011345, -0.902527),
      glm::vec3(0.856712, -0.003491, -0.515782),
      glm::vec3(0.999097, -0.040132, -0.013950),
      glm::vec3(0.887265, 0.018325, 0.460896),
      glm::vec3(0.535261, -0.153848, 0.830558),
      glm::vec3(-0.698998, -0.145220, 0.700223),
      glm::vec3(-0.978501, -0.167629, 0.120148),
      glm::vec3(-0.844890, 0.534352, 0.025079),
      glm::vec3(-0.985548, 0.157296, 0.062875),
      glm::vec3(-0.595818, -0.061049, -0.800796),
      glm::vec3(-0.994881, -0.074109, 0.068704),
      glm::vec3(0.000879, -0.035772, 0.999360),
      glm::vec3(0.004366, -0.040132, 0.999185),
      glm::vec3(0.320194, -0.051465, 0.945953),
      glm::vec3(0.992814, -0.078459, -0.090362),
      glm::vec3(0.992814, -0.078459, -0.090362),
      glm::vec3(0.993551, -0.054951, -0.099180),
      glm::vec3(0.862905, -0.498487, -0.083098),
      glm::vec3(0.984163, -0.158157, -0.080060),
      glm::vec3(0.932298, -0.120136, 0.341157),
      glm::vec3(0.316201, -0.042746, 0.947729),
      glm::vec3(-0.664873, -0.032282, 0.746259),
      glm::vec3(-0.918267, -0.020069, -0.395452) }
};
const std::vector<std::vector<glm::vec3>> animated_camera_right_vectors = {
    { glm::vec3(0.998135, -0.000000, 0.061049),
      glm::vec3(0.695911, -0.000000, 0.718128),
      glm::vec3(0.011341, -0.000000, 0.999936),
      glm::vec3(-0.866026, 0.000000, 0.499998),
      glm::vec3(-0.974173, 0.000000, -0.225803),
      glm::vec3(-0.462521, 0.000000, -0.886609),
      glm::vec3(0.616351, 0.000000, -0.787471),
      glm::vec3(0.999936, -0.000000, 0.011347),
      glm::vec3(0.999634, 0.000000, -0.027046),
      glm::vec3(0.843862, 0.000000, -0.536560),
      glm::vec3(0.088893, 0.000000, -0.996041),
      glm::vec3(-0.087162, 0.000000, -0.996194),
      glm::vec3(-0.815128, 0.000000, -0.579280),
      glm::vec3(-0.999360, 0.000000, 0.035772),
      glm::vec3(-0.577145, 0.000000, 0.816642),
      glm::vec3(0.020071, -0.000000, 0.999799),
      glm::vec3(0.803338, -0.000000, 0.595523),
      glm::vec3(0.998291, 0.000000, -0.058436) },
    { glm::vec3(0.722362, -0.000000, 0.691515),
      glm::vec3(0.999996, -0.000000, 0.002621),
      glm::vec3(0.999863, 0.000000, -0.016576),
      glm::vec3(0.999863, 0.000000, -0.016576),
      glm::vec3(0.683912, 0.000000, -0.729565) },
    { glm::vec3(0.529915, 0.000000, -0.848051),
      glm::vec3(0.792819, 0.000000, -0.609458),
      glm::vec3(0.998440, 0.000000, -0.055827),
      glm::vec3(0.874193, 0.000000, -0.485578),
      glm::vec3(0.261335, 0.000000, -0.965248),
      glm::vec3(0.117523, 0.000000, -0.993070),
      glm::vec3(0.388304, 0.000000, -0.921531) },
    { glm::vec3(0.932323, -0.000000, 0.361625),
      glm::vec3(0.706488, -0.000000, 0.707725),
      glm::vec3(0.361618, -0.000000, 0.932326),
      glm::vec3(0.990145, -0.000000, 0.140047),
      glm::vec3(0.929136, 0.000000, -0.369739),
      glm::vec3(0.999742, -0.000000, 0.022696),
      glm::vec3(0.876303, -0.000000, 0.481761) },
    { glm::vec3(-0.307355, 0.000000, -0.951595),
      glm::vec3(-0.041874, 0.000000, -0.999123),
      glm::vec3(-0.054949, 0.000000, -0.998489),
      glm::vec3(0.902585, -0.000000, 0.430512),
      glm::vec3(0.515786, -0.000000, 0.856718),
      glm::vec3(0.013961, -0.000000, 0.999903),
      glm::vec3(-0.460973, 0.000000, 0.887414),
      glm::vec3(-0.840565, 0.000000, 0.541711),
      glm::vec3(-0.707726, 0.000000, -0.706487),
      glm::vec3(-0.121872, 0.000000, -0.992546),
      glm::vec3(-0.029670, 0.000000, -0.999560),
      glm::vec3(-0.063668, 0.000000, -0.997971),
      glm::vec3(0.802292, 0.000000, -0.596931),
      glm::vec3(-0.068893, 0.000000, -0.997624),
      glm::vec3(-1.000000, 0.000000, 0.000879),
      glm::vec3(-0.999990, 0.000000, 0.004370),
      glm::vec3(-0.947208, 0.000000, 0.320619),
      glm::vec3(0.090641, -0.000000, 0.995884),
      glm::vec3(0.090641, -0.000000, 0.995884),
      glm::vec3(0.099330, -0.000000, 0.995055),
      glm::vec3(0.095857, -0.000000, 0.995395),
      glm::vec3(0.081081, -0.000000, 0.996708),
      glm::vec3(-0.343646, 0.000000, 0.939099),
      glm::vec3(-0.948596, 0.000000, 0.316491),
      glm::vec3(-0.746648, 0.000000, -0.665220),
      glm::vec3(0.395532, 0.000000, -0.918452) }
};
const std::vector<float> animated_camera_speeds = {
    5.0f,
    5.0f,
    2.0f,
    2.0f,
    35.0f
};
} // namespace constants

// -----------------------------------------------------------------------------------------------------------------------------------

CommonResources::CommonResources(dw::vk::Backend::Ptr backend)
{
    create_uniform_buffer(backend);
    load_mesh(backend);

    brdf_preintegrate_lut = std::unique_ptr<dw::BRDFIntegrateLUT>(new dw::BRDFIntegrateLUT(backend));
    blue_noise            = std::unique_ptr<BlueNoise>(new BlueNoise(backend));

    create_environment_resources(backend);
    create_descriptor_set_layouts(backend);
    create_descriptor_sets(backend);
    write_descriptor_sets(backend);

    demo_players.resize(SCENE_TYPE_COUNT);

    for (int i = 0; i < SCENE_TYPE_COUNT; i++)
    {
        demo_players[i] = std::unique_ptr<dw::DemoPlayer>(new dw::DemoPlayer(constants::animated_camera_position_vectors[i], constants::animated_camera_forward_vectors[i], constants::animated_camera_right_vectors[i]));
        demo_players[i]->set_speed(constants::animated_camera_speeds[i]);
    }
}

// -----------------------------------------------------------------------------------------------------------------------------------

CommonResources::~CommonResources()
{
}

// -----------------------------------------------------------------------------------------------------------------------------------

void CommonResources::create_uniform_buffer(dw::vk::Backend::Ptr backend)
{
    ubo_size = backend->aligned_dynamic_ubo_size(sizeof(UBO));
    ubo      = dw::vk::Buffer::create(backend, VK_BUFFER_USAGE_UNIFORM_BUFFER_BIT, ubo_size * dw::vk::Backend::kMaxFramesInFlight, VMA_MEMORY_USAGE_CPU_TO_GPU, VMA_ALLOCATION_CREATE_MAPPED_BIT);
}

// -----------------------------------------------------------------------------------------------------------------------------------

void CommonResources::load_mesh(dw::vk::Backend::Ptr backend)
{
    scenes.reserve(SCENE_TYPE_COUNT);

    {
        std::vector<dw::RayTracedScene::Instance> instances;

        dw::Mesh::Ptr pillar = dw::Mesh::load(backend, "mesh/pillar.gltf");

        if (!pillar)
        {
            DW_LOG_ERROR("Failed to load mesh");
            throw std::runtime_error("Failed to load mesh");
        }

        pillar->initialize_for_ray_tracing(backend);

        meshes.push_back(pillar);

        dw::Mesh::Ptr bunny = dw::Mesh::load(backend, "mesh/bunny.gltf");

        if (!bunny)
        {
            DW_LOG_ERROR("Failed to load mesh");
            throw std::runtime_error("Failed to load mesh");
        }

        bunny->initialize_for_ray_tracing(backend);

        meshes.push_back(bunny);

        dw::Mesh::Ptr ground = dw::Mesh::load(backend, "mesh/ground.gltf");

        if (!ground)
        {
            DW_LOG_ERROR("Failed to load mesh");
            throw std::runtime_error("Failed to load mesh");
        }

        ground->initialize_for_ray_tracing(backend);

        meshes.push_back(ground);

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

        scenes.push_back(dw::RayTracedScene::create(backend, instances));
    }

    {
        std::vector<dw::RayTracedScene::Instance> instances;

        dw::Mesh::Ptr reflections_test = dw::Mesh::load(backend, "mesh/reflections_test.gltf");

        if (!reflections_test)
        {
            DW_LOG_ERROR("Failed to load mesh");
            throw std::runtime_error("Failed to load mesh");
        }

        reflections_test->initialize_for_ray_tracing(backend);

        meshes.push_back(reflections_test);

        dw::RayTracedScene::Instance reflections_test_instance;

        reflections_test_instance.mesh      = reflections_test;
        reflections_test_instance.transform = glm::mat4(1.0f);

        instances.push_back(reflections_test_instance);

        scenes.push_back(dw::RayTracedScene::create(backend, instances));
    }

    {
        std::vector<dw::RayTracedScene::Instance> instances;

        dw::Mesh::Ptr gi_test = dw::Mesh::load(backend, "mesh/global_illumination_test.gltf");

        if (!gi_test)
        {
            DW_LOG_ERROR("Failed to load mesh");
            throw std::runtime_error("Failed to load mesh");
        }

        gi_test->initialize_for_ray_tracing(backend);

        meshes.push_back(gi_test);

        dw::RayTracedScene::Instance gi_test_instance;

        gi_test_instance.mesh      = gi_test;
        gi_test_instance.transform = glm::mat4(1.0f);

        instances.push_back(gi_test_instance);

        scenes.push_back(dw::RayTracedScene::create(backend, instances));
    }

    {
        std::vector<dw::RayTracedScene::Instance> instances;

        dw::Mesh::Ptr pica_pica = dw::Mesh::load(backend, "scene.gltf");

        if (!pica_pica)
        {
            DW_LOG_ERROR("Failed to load mesh");
            throw std::runtime_error("Failed to load mesh");
        }

        pica_pica->initialize_for_ray_tracing(backend);

        meshes.push_back(pica_pica);

        dw::RayTracedScene::Instance pica_pica_instance;

        pica_pica_instance.mesh      = pica_pica;
        pica_pica_instance.transform = glm::scale(glm::mat4(1.0f), glm::vec3(1.0f));

        instances.push_back(pica_pica_instance);

        scenes.push_back(dw::RayTracedScene::create(backend, instances));
    }

    {
        std::vector<dw::RayTracedScene::Instance> instances;

        dw::Mesh::Ptr sponza = dw::Mesh::load(backend, "mesh/sponza.obj");

        if (!sponza)
        {
            DW_LOG_ERROR("Failed to load mesh");
            throw std::runtime_error("Failed to load mesh");
        }

        sponza->initialize_for_ray_tracing(backend);

        meshes.push_back(sponza);

        dw::RayTracedScene::Instance sponza_instance;

        sponza_instance.mesh      = sponza;
        sponza_instance.transform = glm::scale(glm::mat4(1.0f), glm::vec3(0.3f));

        instances.push_back(sponza_instance);

        scenes.push_back(dw::RayTracedScene::create(backend, instances));
    }
}

// -----------------------------------------------------------------------------------------------------------------------------------

void CommonResources::create_environment_resources(dw::vk::Backend::Ptr backend)
{
    // Create procedural sky
    {
        sky_environment                         = std::unique_ptr<SkyEnvironment>(new SkyEnvironment());
        sky_environment->hosek_wilkie_sky_model = std::unique_ptr<dw::HosekWilkieSkyModel>(new dw::HosekWilkieSkyModel(backend));
        sky_environment->cubemap_sh_projection  = std::unique_ptr<dw::CubemapSHProjection>(new dw::CubemapSHProjection(backend, sky_environment->hosek_wilkie_sky_model->image()));
        sky_environment->cubemap_prefilter      = std::unique_ptr<dw::CubemapPrefiler>(new dw::CubemapPrefiler(backend, sky_environment->hosek_wilkie_sky_model->image()));
    }

    // Create blank SH image
    {
        blank_sh_image = dw::vk::Image::create(backend, VK_IMAGE_TYPE_2D, 9, 1, 1, 1, 1, VK_FORMAT_R32G32B32A32_SFLOAT, VMA_MEMORY_USAGE_GPU_ONLY, VK_IMAGE_USAGE_SAMPLED_BIT | VK_IMAGE_USAGE_STORAGE_BIT | VK_IMAGE_USAGE_TRANSFER_DST_BIT, VK_SAMPLE_COUNT_1_BIT, VK_IMAGE_LAYOUT_UNDEFINED);
        blank_sh_image->set_name("Blank SH Projection Image");

        blank_sh_image_view = dw::vk::ImageView::create(backend, blank_sh_image, VK_IMAGE_VIEW_TYPE_2D, VK_IMAGE_ASPECT_COLOR_BIT, 0, 1, 0, 1);
        blank_sh_image_view->set_name("Blank SH Projection Image View");

        std::vector<glm::vec4> sh_data(9);
        std::vector<size_t>    sh_sizes(1);

        for (int i = 0; i < 9; i++)
            sh_data[i] = glm::vec4(0.0f);

        sh_sizes[0] = sizeof(glm::vec4) * 9;

        dw::vk::BatchUploader uploader(backend);

        uploader.upload_image_data(blank_sh_image, sh_data.data(), sh_sizes);

        uploader.submit();
    }

    // Create blank environment map
    {
        blank_cubemap_image      = dw::vk::Image::create(backend, VK_IMAGE_TYPE_2D, 2, 2, 1, 1, 6, VK_FORMAT_R32G32B32A32_SFLOAT, VMA_MEMORY_USAGE_GPU_ONLY, VK_IMAGE_USAGE_SAMPLED_BIT | VK_IMAGE_USAGE_TRANSFER_DST_BIT, VK_SAMPLE_COUNT_1_BIT, VK_IMAGE_LAYOUT_UNDEFINED, 0, nullptr, VK_IMAGE_CREATE_CUBE_COMPATIBLE_BIT);
        blank_cubemap_image_view = dw::vk::ImageView::create(backend, blank_cubemap_image, VK_IMAGE_VIEW_TYPE_CUBE, VK_IMAGE_ASPECT_COLOR_BIT, 0, 1, 0, 6);

        std::vector<glm::vec4> cubemap_data(2 * 2 * 6);
        std::vector<size_t>    cubemap_sizes(6);

        int idx = 0;

        for (int layer = 0; layer < 6; layer++)
        {
            cubemap_sizes[layer] = sizeof(glm::vec4) * 4;

            for (int i = 0; i < 4; i++)
                cubemap_data[idx++] = glm::vec4(0.0f);
        }

        dw::vk::BatchUploader uploader(backend);

        uploader.upload_image_data(blank_cubemap_image, cubemap_data.data(), cubemap_sizes);

        uploader.submit();
    }

    // Load environment maps
    std::unique_ptr<dw::EquirectangularToCubemap> equirectangular_to_cubemap = std::unique_ptr<dw::EquirectangularToCubemap>(new dw::EquirectangularToCubemap(backend, VK_FORMAT_R32G32B32A32_SFLOAT));

    hdr_environments.resize(constants::environment_map_images.size());

    for (int i = 0; i < constants::environment_map_images.size(); i++)
    {
        std::shared_ptr<HDREnvironment> environment = std::shared_ptr<HDREnvironment>(new HDREnvironment());

        auto input_image = dw::vk::Image::create_from_file(backend, constants::environment_map_images[i], true);

        environment->image                 = dw::vk::Image::create(backend, VK_IMAGE_TYPE_2D, 1024, 1024, 1, 5, 6, VK_FORMAT_R32G32B32A32_SFLOAT, VMA_MEMORY_USAGE_GPU_ONLY, VK_IMAGE_USAGE_SAMPLED_BIT | VK_IMAGE_USAGE_COLOR_ATTACHMENT_BIT | VK_IMAGE_USAGE_TRANSFER_DST_BIT | VK_IMAGE_USAGE_TRANSFER_SRC_BIT, VK_SAMPLE_COUNT_1_BIT, VK_IMAGE_LAYOUT_UNDEFINED, 0, nullptr, VK_IMAGE_CREATE_CUBE_COMPATIBLE_BIT);
        environment->image_view            = dw::vk::ImageView::create(backend, environment->image, VK_IMAGE_VIEW_TYPE_CUBE, VK_IMAGE_ASPECT_COLOR_BIT, 0, 1, 0, 6);
        environment->cubemap_sh_projection = std::unique_ptr<dw::CubemapSHProjection>(new dw::CubemapSHProjection(backend, environment->image));
        environment->cubemap_prefilter     = std::unique_ptr<dw::CubemapPrefiler>(new dw::CubemapPrefiler(backend, environment->image));

        equirectangular_to_cubemap->convert(input_image, environment->image);

        auto cmd_buf = backend->allocate_graphics_command_buffer(true);

        environment->image->generate_mipmaps(cmd_buf);
        environment->cubemap_sh_projection->update(cmd_buf);
        environment->cubemap_prefilter->update(cmd_buf);

        vkEndCommandBuffer(cmd_buf->handle());

        backend->flush_graphics({ cmd_buf });

        hdr_environments[i] = environment;
    }
}

// -----------------------------------------------------------------------------------------------------------------------------------

void CommonResources::create_descriptor_set_layouts(dw::vk::Backend::Ptr backend)
{
    {
        dw::vk::DescriptorSetLayout::Desc desc;

        desc.add_binding(0, VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER_DYNAMIC, 1, VK_SHADER_STAGE_RAYGEN_BIT_KHR | VK_SHADER_STAGE_CLOSEST_HIT_BIT_KHR | VK_SHADER_STAGE_FRAGMENT_BIT | VK_SHADER_STAGE_VERTEX_BIT | VK_SHADER_STAGE_COMPUTE_BIT);

        per_frame_ds_layout = dw::vk::DescriptorSetLayout::create(backend, desc);
        per_frame_ds_layout->set_name("Per Frame DS Layout");
    }

    {
        dw::vk::DescriptorSetLayout::Desc desc;

        desc.add_binding(0, VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER, 1, VK_SHADER_STAGE_RAYGEN_BIT_KHR | VK_SHADER_STAGE_CLOSEST_HIT_BIT_KHR | VK_SHADER_STAGE_MISS_BIT_KHR | VK_SHADER_STAGE_FRAGMENT_BIT | VK_SHADER_STAGE_COMPUTE_BIT);
        desc.add_binding(1, VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER, 1, VK_SHADER_STAGE_RAYGEN_BIT_KHR | VK_SHADER_STAGE_CLOSEST_HIT_BIT_KHR | VK_SHADER_STAGE_MISS_BIT_KHR | VK_SHADER_STAGE_FRAGMENT_BIT | VK_SHADER_STAGE_COMPUTE_BIT);

        blue_noise_ds_layout = dw::vk::DescriptorSetLayout::create(backend, desc);
        blue_noise_ds_layout->set_name("Blue Noise DS Layout");
    }

    {
        dw::vk::DescriptorSetLayout::Desc desc;

        desc.add_binding(0, VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER, 1, VK_SHADER_STAGE_FRAGMENT_BIT | VK_SHADER_STAGE_CLOSEST_HIT_BIT_KHR | VK_SHADER_STAGE_MISS_BIT_KHR);
        desc.add_binding(1, VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER, 1, VK_SHADER_STAGE_FRAGMENT_BIT | VK_SHADER_STAGE_CLOSEST_HIT_BIT_KHR | VK_SHADER_STAGE_MISS_BIT_KHR);
        desc.add_binding(2, VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER, 1, VK_SHADER_STAGE_FRAGMENT_BIT | VK_SHADER_STAGE_CLOSEST_HIT_BIT_KHR | VK_SHADER_STAGE_MISS_BIT_KHR);
        desc.add_binding(3, VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER, 1, VK_SHADER_STAGE_FRAGMENT_BIT | VK_SHADER_STAGE_CLOSEST_HIT_BIT_KHR | VK_SHADER_STAGE_MISS_BIT_KHR);

        skybox_ds_layout = dw::vk::DescriptorSetLayout::create(backend, desc);
        skybox_ds_layout->set_name("Skybox DS Layout");
    }

    {
        dw::vk::DescriptorSetLayout::Desc desc;

        desc.add_binding(0, VK_DESCRIPTOR_TYPE_STORAGE_IMAGE, 1, VK_SHADER_STAGE_RAYGEN_BIT_KHR | VK_SHADER_STAGE_COMPUTE_BIT);

        storage_image_ds_layout = dw::vk::DescriptorSetLayout::create(backend, desc);
        storage_image_ds_layout->set_name("Storage Image DS Layout");
    }

    {
        dw::vk::DescriptorSetLayout::Desc desc;

        desc.add_binding(0, VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER, 1, VK_SHADER_STAGE_RAYGEN_BIT_KHR | VK_SHADER_STAGE_CLOSEST_HIT_BIT_KHR | VK_SHADER_STAGE_MISS_BIT_KHR | VK_SHADER_STAGE_FRAGMENT_BIT | VK_SHADER_STAGE_COMPUTE_BIT);

        combined_sampler_ds_layout = dw::vk::DescriptorSetLayout::create(backend, desc);
        combined_sampler_ds_layout->set_name("Combined Sampler DS Layout");
    }

    {
        dw::vk::DescriptorSetLayout::Desc desc;

        desc.add_binding(0, VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER, 1, VK_SHADER_STAGE_RAYGEN_BIT_KHR | VK_SHADER_STAGE_CLOSEST_HIT_BIT_KHR | VK_SHADER_STAGE_COMPUTE_BIT | VK_SHADER_STAGE_FRAGMENT_BIT);
        desc.add_binding(1, VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER, 1, VK_SHADER_STAGE_RAYGEN_BIT_KHR | VK_SHADER_STAGE_CLOSEST_HIT_BIT_KHR | VK_SHADER_STAGE_COMPUTE_BIT | VK_SHADER_STAGE_FRAGMENT_BIT);
        desc.add_binding(2, VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER_DYNAMIC, 1, VK_SHADER_STAGE_RAYGEN_BIT_KHR | VK_SHADER_STAGE_CLOSEST_HIT_BIT_KHR | VK_SHADER_STAGE_COMPUTE_BIT | VK_SHADER_STAGE_VERTEX_BIT | VK_SHADER_STAGE_FRAGMENT_BIT);

        ddgi_read_ds_layout = dw::vk::DescriptorSetLayout::create(backend, desc);
    }
}

// -----------------------------------------------------------------------------------------------------------------------------------

void CommonResources::create_descriptor_sets(dw::vk::Backend::Ptr backend)
{
    per_frame_ds = backend->allocate_descriptor_set(per_frame_ds_layout);

    for (int i = 0; i < 9; i++)
        blue_noise_ds[i] = backend->allocate_descriptor_set(blue_noise_ds_layout);

    int num_environment_map_images = constants::environment_map_images.size() + 2;

    skybox_ds.resize(num_environment_map_images);

    for (int i = 0; i < num_environment_map_images; i++)
        skybox_ds[i] = backend->allocate_descriptor_set(skybox_ds_layout);
}

// -----------------------------------------------------------------------------------------------------------------------------------

void CommonResources::write_descriptor_sets(dw::vk::Backend::Ptr backend)
{
    // Per-frame
    {
        std::vector<VkWriteDescriptorSet> write_datas;

        {
            VkDescriptorBufferInfo buffer_info;

            buffer_info.range  = sizeof(UBO);
            buffer_info.offset = 0;
            buffer_info.buffer = ubo->handle();

            VkWriteDescriptorSet write_data;
            DW_ZERO_MEMORY(write_data);

            write_data.sType           = VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET;
            write_data.descriptorCount = 1;
            write_data.descriptorType  = VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER_DYNAMIC;
            write_data.pBufferInfo     = &buffer_info;
            write_data.dstBinding      = 0;
            write_data.dstSet          = per_frame_ds->handle();

            write_datas.push_back(write_data);
        }

        vkUpdateDescriptorSets(backend->device(), write_datas.size(), write_datas.data(), 0, nullptr);
    }

    // Skybox resources
    int num_environment_map_images = constants::environment_map_images.size() + 2;

    for (int i = 0; i < num_environment_map_images; i++)
    {
        VkDescriptorImageInfo image_info[4];

        image_info[0].sampler = backend->bilinear_sampler()->handle();
        if (i == ENVIRONMENT_TYPE_NONE)
            image_info[0].imageView = blank_cubemap_image_view->handle();
        else if (i == ENVIRONMENT_TYPE_PROCEDURAL_SKY)
            image_info[0].imageView = sky_environment->hosek_wilkie_sky_model->image_view()->handle();
        else
            image_info[0].imageView = hdr_environments[i - 2]->image_view->handle();
        image_info[0].imageLayout = VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL;

        image_info[1].sampler = backend->trilinear_sampler()->handle();
        if (i == ENVIRONMENT_TYPE_NONE)
            image_info[1].imageView = blank_sh_image_view->handle();
        else if (i == ENVIRONMENT_TYPE_PROCEDURAL_SKY)
            image_info[1].imageView = sky_environment->cubemap_sh_projection->image_view()->handle();
        else
            image_info[1].imageView = hdr_environments[i - 2]->cubemap_sh_projection->image_view()->handle();
        image_info[1].imageLayout = VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL;

        image_info[2].sampler = backend->trilinear_sampler()->handle();
        if (i == ENVIRONMENT_TYPE_NONE)
            image_info[2].imageView = blank_cubemap_image_view->handle();
        else if (i == ENVIRONMENT_TYPE_PROCEDURAL_SKY)
            image_info[2].imageView = sky_environment->cubemap_prefilter->image_view()->handle();
        else
            image_info[2].imageView = hdr_environments[i - 2]->cubemap_prefilter->image_view()->handle();
        image_info[2].imageLayout = VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL;

        image_info[3].sampler     = backend->bilinear_sampler()->handle();
        image_info[3].imageView   = brdf_preintegrate_lut->image_view()->handle();
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
        write_data[0].dstSet          = skybox_ds[i]->handle();

        write_data[1].sType           = VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET;
        write_data[1].descriptorCount = 1;
        write_data[1].descriptorType  = VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER;
        write_data[1].pImageInfo      = &image_info[1];
        write_data[1].dstBinding      = 1;
        write_data[1].dstSet          = skybox_ds[i]->handle();

        write_data[2].sType           = VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET;
        write_data[2].descriptorCount = 1;
        write_data[2].descriptorType  = VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER;
        write_data[2].pImageInfo      = &image_info[2];
        write_data[2].dstBinding      = 2;
        write_data[2].dstSet          = skybox_ds[i]->handle();

        write_data[3].sType           = VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET;
        write_data[3].descriptorCount = 1;
        write_data[3].descriptorType  = VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER;
        write_data[3].pImageInfo      = &image_info[3];
        write_data[3].dstBinding      = 3;
        write_data[3].dstSet          = skybox_ds[i]->handle();

        vkUpdateDescriptorSets(backend->device(), 4, &write_data[0], 0, nullptr);
    }

    current_skybox_ds = skybox_ds[current_environment_type];

    // Blue Noise
    {
        for (int i = 0; i < 9; i++)
        {
            VkDescriptorImageInfo image_info[2];

            image_info[0].sampler     = backend->nearest_sampler()->handle();
            image_info[0].imageView   = blue_noise->m_sobol_image_view->handle();
            image_info[0].imageLayout = VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL;

            image_info[1].sampler     = backend->nearest_sampler()->handle();
            image_info[1].imageView   = blue_noise->m_scrambling_ranking_image_view[i]->handle();
            image_info[1].imageLayout = VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL;

            VkWriteDescriptorSet write_data[2];
            DW_ZERO_MEMORY(write_data[0]);
            DW_ZERO_MEMORY(write_data[1]);

            write_data[0].sType           = VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET;
            write_data[0].descriptorCount = 1;
            write_data[0].descriptorType  = VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER;
            write_data[0].pImageInfo      = &image_info[0];
            write_data[0].dstBinding      = 0;
            write_data[0].dstSet          = blue_noise_ds[i]->handle();

            write_data[1].sType           = VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET;
            write_data[1].descriptorCount = 1;
            write_data[1].descriptorType  = VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER;
            write_data[1].pImageInfo      = &image_info[1];
            write_data[1].dstBinding      = 1;
            write_data[1].dstSet          = blue_noise_ds[i]->handle();

            vkUpdateDescriptorSets(backend->device(), 2, &write_data[0], 0, nullptr);
        }
    }
}

// -----------------------------------------------------------------------------------------------------------------------------------