#pragma once

#include "VulkanFW/vks/vks.hpp"
#include "VulkanFW/vks/helpers.hpp"
#include "VulkanFW/vks/filesystem.hpp"
#include "VulkanFW/vks/shaders.hpp"
#include "VulkanFW/vks/model.hpp"
#include "VulkanFW/vks/pipelines.hpp"
#include "VulkanFW/vks/texture.hpp"

#ifdef WIN32
//#include <handleapi.h>
#else
Linux / MacOS support still not implemented
#endif

#include <vulkan/vulkan.hpp>

#include <glad/glad.h>
#include <GLFW/glfw3.h>


// TEST: create window?
constexpr bool RT_TEST_WINDOW = true;


constexpr int RT_STAGE_COUNT = 6;
constexpr int RT_GROUP_COUNT = 5;

struct VertexModel {
    glm::vec4 pos;
    glm::vec4 normal;
    glm::vec4 color;
};


class HWAccelerator
{
public:
    HWAccelerator(void);
    ~HWAccelerator(void) = default;

    void enqueueTraceRays();
    void finish();
    void debugPrintHit0();
    void createGLObjects();

private:
    void getRTDeviceInfo();
    void prepareTextureTarget(vks::Image& tex, uint32_t width, uint32_t height, vk::Format format);
    void updateDrawCommandBuffer(const vk::CommandBuffer& cmdBuffer);
    void updateRaytracingCommandBuffer();
    void createRaytracingCommandBuffer();
    void buildAccelerationStructure();
    void setupDescriptorPool();
    void prepareRaytracing();
    void updateRTDescriptorSets();
    void setupShaderBindingTable();
    void loadAssets();
    void updateUniformBuffers();
    void getRaytracingQueue();

    void draw();
    void prepareFrame();
    void submitFrame();
    void drawCurrentCommandBuffer();

private:
    struct InstanceNV {
        float transform[12];
        uint32_t instanceID : 24;
        uint32_t instanceMask : 8;
        uint32_t instanceContributionToHitGroupIndex : 24;
        uint32_t flags : 8;
        uint64_t accelerationStructureHandle;
    };

    // Order by size to avoid alignment mismatches between host and device
    struct UboCompute {
        glm::mat4 invR;
        glm::vec4 camPos = glm::vec4(0.5f, 0.0f, 0.0f, 1.0f);
        glm::vec4 lightPos;
        float aspectRatio;
        float fov = 90.0f;
        unsigned int numTasks = 0;
    } uboRT;

    struct {
        vks::model::Model quad; // for fragment shader
        vks::model::Model rtMesh; // for actual raytracing
    } meshes;

    struct {
        vk::Pipeline display;
        vk::Pipeline raytracing;
    } pipelines;

    struct ShareHandles {
        HANDLE memory = INVALID_HANDLE_VALUE;
        HANDLE glReady = INVALID_HANDLE_VALUE;
        HANDLE glComplete = INVALID_HANDLE_VALUE;
    } handles;

    struct GLHandles {
        GLuint semGlReady{ 0 };
        GLuint semGlComplete{ 0 };
        GLuint memShared{ 0 };
        GLuint hitBuffer{ 0 };
    } glHandles;
    

    vk::DispatchLoaderDynamic loaderNV;
    vks::Image textureRaytracingTarget;
    vks::Buffer uniformDataRaytracing;
    uint32_t rtDevMaxRecursionDepth = 0;
    uint32_t rtDevShaderGoupHandleSize = 0;
    uint32_t numMeshesNV = 1;
    uint32_t numInstancesNV = 1;
    std::vector<InstanceNV> instances;
    std::vector<vk::GeometryNV> geometries;
    vks::Buffer transform3x4;
    vks::Buffer scratchMem;
    vks::Buffer instancesNV;
    vks::Buffer shaderBindingTable;
    vks::Buffer topLevelAccBuff;
    vks::Buffer bottomLevelAccBuff;
    vk::AccelerationStructureNV topHandle;
    vk::AccelerationStructureNV bottomHandle;

    // Written to by Vulkan, accessed by OpenCL
    vks::Buffer hitBuffer;

    vk::Pipeline rtPipeline;
    vk::Queue raytracingQueue;
    vk::CommandBuffer raytracingCmdBuffer;
    vk::PipelineLayout raytracingPipelineLayout;
    vk::DescriptorSet raytracingDescriptorSet;
    vk::DescriptorSetLayout raytracingDescriptorSetLayout;

    vk::QueryPool rtPerfQueryPool;

    void setupWindow();
    void setupSwapchain();
    void initVulkan();
    void prepare();
    void setupDepthStencil();
    void setupRenderPass();
    void setupRenderPassBeginInfo();
    void setupFrameBuffer();
    void prepareUniformBuffers();
    void setupDescriptorSet();
    void setupDescriptorSetLayout();
    void updateDescriptorSets();
    void prepareRasterizationPipeline();
    void setupQueryPool();
    void setupSharedBuffers();

    /* 
      ExampleBase
    */

    void allocateCommandBuffers();
    void buildCommandBuffers();
    void clearCommandBuffers();

    GLFWwindow* window{ nullptr };
    std::vector<vk::CommandBuffer> commandBuffers;
    std::vector<vk::ClearValue> clearValues;
    vk::RenderPassBeginInfo renderPassBeginInfo;
    vk::Viewport viewport() { return vks::util::viewport(size); }
    vk::Rect2D scissor() { return vks::util::rect2D(size); }
    vk::RenderPass renderPass;
    std::vector<vk::Framebuffer> framebuffers;

    vk::PipelineLayout pipelineLayout;
    vk::DescriptorSet descriptorSetPostCompute;
    vk::DescriptorSetLayout descriptorSetLayout;

    // Color buffer format
    vk::Format colorformat{ vk::Format::eB8G8R8A8Unorm };

    // Depth buffer format...  selected during Vulkan initialization
    vk::Format depthFormat{ vk::Format::eUndefined };

    // Active frame buffer index
    uint32_t currentBuffer = 0;
    // Descriptor set pool
    vk::DescriptorPool descriptorPool;
    std::vector<vk::Semaphore> renderWaitSemaphores;
    std::vector<vk::PipelineStageFlags> renderWaitStages;
    std::vector<vk::Semaphore> renderSignalSemaphores;
    vks::Context context;
    const vk::PhysicalDevice& physicalDevice{ context.physicalDevice };
    const vk::Device& device{ context.device };
    const vk::Queue& queue{ context.queue };
    const vk::PhysicalDeviceFeatures& deviceFeatures{ context.deviceFeatures };
    vk::PhysicalDeviceFeatures& enabledFeatures{ context.enabledFeatures };
    //vkx::ui::UIOverlay ui{ context };
    vk::SurfaceKHR surface;
    // Wraps the swap chain to present images (framebuffers) to the windowing system
    vks::SwapChain swapChain;
    struct {
        vk::Semaphore acquireComplete; // swap chain image aquisition
        vk::Semaphore renderComplete;
        vk::Semaphore glReady;
        vk::Semaphore glComplete;
    } semaphores;
    // Command buffer pool
    
    vk::CommandPool cmdPool;
    bool prepared = false;
    uint32_t version = VK_MAKE_VERSION(1, 1, 0);
    vk::Extent2D size{ 1920, 1080 };
    uint32_t& width{ size.width };
    uint32_t& height{ size.height };

    vk::ClearColorValue defaultClearColor = vks::util::clearColor(glm::vec4({ 0.025f, 0.025f, 0.025f, 1.0f }));
    vk::ClearDepthStencilValue defaultClearDepth{ 1.0f, 0 };
    vks::Image depthStencil;

    bool enableVsync = false;
};