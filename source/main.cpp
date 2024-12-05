#define GLFW_INCLUDE_VULKAN
#include <GLFW/glfw3.h>

#define GLM_FORCE_RADIANS
#include <glm/glm.hpp>
#include <glm/gtc/matrix_transform.hpp>

#pragma warning(push, 0)
#define STB_IMAGE_IMPLEMENTATION
#include <stb_image.h>
#pragma warning(pop)

#include <iostream>
#include <stdexcept>
#include <cstdlib>
#include <vector>
#include <optional>
#include <set>
#include <algorithm>
#include <chrono>
#include <array>

#include "utils.h"

const uint32_t WIDTH = 800;
const uint32_t HEIGHT = 600;
const int MAX_FRAMES_IN_FLIGHT = 2;

// the validation layers we want to be using for debugging our program
const std::vector<const char*> validationLayers{
        "VK_LAYER_KHRONOS_validation"
    };

// the device extensions we require for this program to run
const std::vector<const char*> deviceExtensions{
        VK_KHR_SWAPCHAIN_EXTENSION_NAME
    };

// NDEBUG is part of the c++ spec, meaning not debug. we can use this for debug conditionals
#ifdef NDEBUG
const bool enableValidationLayers{ false };
#else
const bool enableValidationLayers{ true };
#endif

// proxy function to the built in "vkCreateDebugUtilsMessengerEXT"
VkResult CreateDebugUtilsMessengerEXT(VkInstance instance, const VkDebugUtilsMessengerCreateInfoEXT* pCreateInfo, const VkAllocationCallbacks* pAllocator, VkDebugUtilsMessengerEXT* pDebugMessenger) 
{
    auto func = (PFN_vkCreateDebugUtilsMessengerEXT)vkGetInstanceProcAddr(instance, "vkCreateDebugUtilsMessengerEXT");
    if (func != nullptr) 
    {
        return func(instance, pCreateInfo, pAllocator, pDebugMessenger);
    }
    else 
    {
        return VK_ERROR_EXTENSION_NOT_PRESENT;
    }
}

// proxy function to the built in "vkDestroyDebugUtilsMessengerEXT"
void DestroyDebugUtilsMessengerEXT(VkInstance instance, VkDebugUtilsMessengerEXT debugMessenger, const VkAllocationCallbacks* pAllocator)
{
    auto func = (PFN_vkDestroyDebugUtilsMessengerEXT)vkGetInstanceProcAddr(instance, "vkDestroyDebugUtilsMessengerEXT");

    if (func != nullptr)
    {
        func(instance, debugMessenger, pAllocator);
    }
}

class HelloTriangleApplication
{
private:
    // a structure to represent all the queue families we want to be using. also validates itself to make sure we support all the queue families we need
    struct QueueFamiyIndices
    {
        std::optional<uint32_t> graphicsFamily{ std::make_optional<uint32_t>() };
        std::optional<uint32_t> presentFamily{ std::make_optional<uint32_t>() };

        bool isComplete() const
        {
            return graphicsFamily.has_value() && presentFamily.has_value();
        }
    };


    // a structure to represent the implementation details of our swap chain
    struct SwapChainSupportDetails
    {
        VkSurfaceCapabilitiesKHR capabilities{ };
        std::vector<VkSurfaceFormatKHR> formats{ };
        std::vector<VkPresentModeKHR> presentModes{ };
    };

    // something to descripe the structure of our vertex in the shader
    struct Vertex
    {
        glm::vec2 pos{ 0.f, 0.f };
        glm::vec3 colour{ 0.f, 0.f, 0.f };

        // describes how we're going to load this specific vertex into our shaders
        static VkVertexInputBindingDescription getBindingDescription()
        {
            VkVertexInputBindingDescription bindingDescription{ };
            // this is the position of our binding description in an array of bindings, I assume for different ways
            // to describe vertices. we only have one binding, so this number gets to be 0.
            bindingDescription.binding = 0;
            // describes how large our vertices are, so the GPU knows when it's finished with one and can
            // move to the next. 
            bindingDescription.stride = sizeof(Vertex);
            // the other option here is VK_VERTEX_INPUT_RATE_INSTANCE. we're not using instanced rendering,
            // but I should look into it to see what it's actually doing over per-vertex data.
            // it's probably just a more convenient way to store a shitload of vertex data
            bindingDescription.inputRate = VK_VERTEX_INPUT_RATE_VERTEX;

            return bindingDescription;
        }

        static std::array<VkVertexInputAttributeDescription, 2> getAttributeDescriptions()
        {
            std::array<VkVertexInputAttributeDescription, 2> attributeDescriptions{ };
            // describe our first attribute in a Vertex, this being the 2d position
            
            // binding describes which binding we're using to hold this attribute in the first place,
            // and should match the binding description above
            attributeDescriptions[0].binding = 0;
            // this is the location that matches up to the location described in the shader.
            attributeDescriptions[0].location = 0;
            // this is the format we'll be passing into our shader.
            attributeDescriptions[0].format = VK_FORMAT_R32G32_SFLOAT;
            // this is where our data is located in the data structure we'll be passing into our shader
            attributeDescriptions[0].offset = offsetof(Vertex, pos);
            // describe our second attribute in a Vertex, this being our colour
            attributeDescriptions[1].binding = 0;
            attributeDescriptions[1].location = 1;
            attributeDescriptions[1].format = VK_FORMAT_R32G32B32_SFLOAT;
            attributeDescriptions[1].offset = offsetof(Vertex, colour);

            return attributeDescriptions;
        }
    };

    struct UniformBufferObject
    {
        alignas(16) glm::mat4 model;
        alignas(16) glm::mat4 view;
        alignas(16) glm::mat4 proj;
    };

private:
    GLFWwindow* window{ nullptr };

    VkInstance instance{ VK_NULL_HANDLE };

    VkDebugUtilsMessengerEXT debugMessenger{ VK_NULL_HANDLE };

    VkPhysicalDevice physicalDevice{ VK_NULL_HANDLE };
    VkDevice device{ VK_NULL_HANDLE };

    VkQueue graphicsQueue{ VK_NULL_HANDLE };
    VkQueue presentQueue{ VK_NULL_HANDLE };

    VkSurfaceKHR surface{ VK_NULL_HANDLE };

    VkSwapchainKHR swapChain{ VK_NULL_HANDLE };
    std::vector<VkImage> swapChainImages{ };
    VkFormat swapChainImageFormat{ VK_FORMAT_UNDEFINED };
    VkExtent2D swapChainExtent{ 0, 0 };
    std::vector<VkImageView> swapChainImageViews{ };
    std::vector<VkFramebuffer> swapChainFramebuffers{ };

    VkDescriptorSetLayout descriptorSetLayout{ VK_NULL_HANDLE };
    VkPipelineLayout pipelineLayout{ VK_NULL_HANDLE };
    VkRenderPass renderPass{ VK_NULL_HANDLE };
    VkPipeline graphicsPipeline{ VK_NULL_HANDLE };

    VkCommandPool commandPool{ VK_NULL_HANDLE };
    std::vector<VkCommandBuffer> commandBuffers{ };
    VkDescriptorPool descriptorPool{ VK_NULL_HANDLE };
    std::vector<VkDescriptorSet> descriptorSets{ };
    std::vector<VkSemaphore> imageAvailableSemaphores{ };
    std::vector<VkSemaphore> renderFinishedSemaphores{ };
    std::vector<VkFence> inFlightFences{ VK_NULL_HANDLE };
    bool frameBufferResized{ false };
    uint32_t currentFrame{ 0 };

    VkBuffer vertexBuffer{ VK_NULL_HANDLE };
    VkDeviceMemory vertexBufferMemory{ VK_NULL_HANDLE };
    const std::vector<Vertex> vertices{ {
        { { 0.5f, -0.5f }, { 1.f, 0.f, 0.f } },
        { { 0.5f, 0.5f }, { 0.f, 1.f, 0.f } },
        { { -0.5f, 0.5f }, { 0.f, 0.f, 1.f } },
        { { -0.5f, -0.5f }, { 1.f, 1.f, 1.f } },
    } };

    // use the smallest number possible, 16 bits goes up to 65535 vertices which
    // is good enough for most meshes. sometimes it won't be though
    VkBuffer indexBuffer{ VK_NULL_HANDLE };
    VkDeviceMemory indexBufferMemory{ VK_NULL_HANDLE };
    const std::vector<uint16_t> indices{ {
        0, 1, 2, 2, 3, 0
    } };

    std::vector<VkBuffer> uniformBuffers{ };
    std::vector<VkDeviceMemory> uniformBuffersMemory{ };
    std::vector<void*> uniformBuffersMapped{ };
    VkImage textureImage{ VK_NULL_HANDLE };
    VkDeviceMemory textureImageMemory{ VK_NULL_HANDLE };
    VkImageView textureImageView{ VK_NULL_HANDLE };
    VkSampler textureSampler{ VK_NULL_HANDLE };

private:
    static void framebufferResizeCallback(GLFWwindow* window, int width, int height)
    {
        HelloTriangleApplication* app{ reinterpret_cast<HelloTriangleApplication*>(glfwGetWindowUserPointer(window)) };
        app->frameBufferResized = true;
    }

    // this is how we display our debug messaged to the standard output stream
    static VKAPI_ATTR VkBool32 VKAPI_CALL debugCallback(VkDebugUtilsMessageSeverityFlagBitsEXT messageSeverity,
        VkDebugUtilsMessageTypeFlagsEXT messageType,
        const VkDebugUtilsMessengerCallbackDataEXT* pCallbackData,
        void* pUserData)
    {
        std::cerr << "validation layer: " << pCallbackData->pMessage << std::endl;

        return VK_FALSE;
    }

public:
    // the main entry point to this program
    void run() 
    {
        initWindow();
        initVulkan();
        mainLoop();
        cleanup();
    }

private:
    void mainLoop()
    {
        while (!glfwWindowShouldClose(window))
        {
            glfwPollEvents();
            drawFrame();
        }

        vkDeviceWaitIdle(device);
    }

    void initWindow() 
    {
        glfwInit();

        glfwWindowHint(GLFW_CLIENT_API, GLFW_NO_API);
         
        window = glfwCreateWindow(WIDTH, HEIGHT, "Vulkan", nullptr, nullptr);
        glfwSetWindowUserPointer(window, this);
        glfwSetFramebufferSizeCallback(window, framebufferResizeCallback);
    }

    void initVulkan() 
    {
        createInstance();
        setupDebugMessenger();
        createSurface();
        pickPhysicalDevice();
        createLogicalDevice();
        createSwapChain();
        createImageViews();
        createRenderPass();
        createDescriptorSetLayout();
        createGraphicsPipline();
        createFramebuffers();
        createCommandPool();
        createTextureImage();
        createTextureImageView();
        createTextureSampler();
        createVertexBuffer();
        createIndexBuffer();
        createUniformBuffers();
        createDescriptorPool();
        createDescriptorSets();
        createCommandBuffers();
        createSyncObjects();
    }

    void cleanup()
    {
        if (enableValidationLayers)
        {
            DestroyDebugUtilsMessengerEXT(instance, debugMessenger, nullptr);
        }

        cleanupSwapChain();

        vkDestroySampler(device, textureSampler, nullptr);
        vkDestroyImageView(device, textureImageView, nullptr);

        vkDestroyImage(device, textureImage, nullptr);
        vkFreeMemory(device, textureImageMemory, nullptr);

        vkDestroyBuffer(device, indexBuffer, nullptr);
        vkFreeMemory(device, indexBufferMemory, nullptr);

        vkDestroyBuffer(device, vertexBuffer, nullptr);
        vkFreeMemory(device, vertexBufferMemory, nullptr);

        for (size_t i = 0; i < MAX_FRAMES_IN_FLIGHT; ++i)
        {
            vkDestroySemaphore(device, imageAvailableSemaphores[i], nullptr);
            vkDestroySemaphore(device, renderFinishedSemaphores[i], nullptr);
            vkDestroyFence(device, inFlightFences[i], nullptr);
        }

        vkDestroyCommandPool(device, commandPool, nullptr);

        for (size_t i{ 0 }; i < MAX_FRAMES_IN_FLIGHT; ++i)
        {
            vkDestroyBuffer(device, uniformBuffers[i], nullptr);
            vkFreeMemory(device, uniformBuffersMemory[i], nullptr);
        }

        vkDestroyDescriptorPool(device, descriptorPool, nullptr);
        vkDestroyDescriptorSetLayout(device, descriptorSetLayout, nullptr);

        vkDestroyPipeline(device, graphicsPipeline, nullptr);
        vkDestroyPipelineLayout(device, pipelineLayout, nullptr);

        vkDestroyRenderPass(device, renderPass, nullptr);

        // we need to explicitly destroy most of the Vulkan objects we're using, and in an order that allows us to destroy them with defined behaviour
        vkDestroyDevice(device, nullptr);
        vkDestroySurfaceKHR(instance, surface, nullptr);
        vkDestroyInstance(instance, nullptr);

        glfwDestroyWindow(window);
        glfwTerminate();
    }

    void cleanupSwapChain()
    {
        for (VkFramebuffer framebuffer : swapChainFramebuffers)
        {
            vkDestroyFramebuffer(device, framebuffer, nullptr);
        }

        for (VkImageView imageView : swapChainImageViews)
        {
            vkDestroyImageView(device, imageView, nullptr);
        }

        vkDestroySwapchainKHR(device, swapChain, nullptr);
    }

    void recreateSwapChain()
    {
        // special case to handle minimizing. we'll just idle in this loop
        // until we have an event happen, then if that event triggers resizing the
        // window we'll back out of it.
        int width{ 0 };
        int height{ 0 };
        glfwGetFramebufferSize(window, &width, &height);
        while (width == 0 || height == 0)
        {
            glfwGetFramebufferSize(window, &width, &height);
            glfwWaitEvents();
        }

        vkDeviceWaitIdle(device);

        cleanupSwapChain();

        createSwapChain();
        createImageViews();
        createFramebuffers();
    }

    void drawFrame()
    {
        // we pass in our fences and wait for all of them to complete. we only have one, so it doesn't really matter,
        // but we may as well be explicit. we're also disabling the timeout, so it'll wait basically forever
        // vkWaitForFences is also a blocking call, so our program will be unable to do anything while it's in this state
        // inFlightFence will signal when the command buffer has finished executing and is ready to be written to again
        // trying to check if the frame is available, and if it's not, then we increase the frame count and try to write to
        // the other frame instead.
        vkWaitForFences(device, 1, &inFlightFences[currentFrame], VK_TRUE, UINT64_MAX);

        uint32_t imageIndex;
        // gets the next image and spits out an image that'll correspond to a VkImage in our swapChainImages. we'll signal imageAvailableSemaphore
        // here to say that our image is available, and allow our GPU to move on to things that might be locked behind this semaphore
        VkResult result{ vkAcquireNextImageKHR(device, swapChain, UINT64_MAX, imageAvailableSemaphores[currentFrame], VK_NULL_HANDLE, &imageIndex) };

        // vkAcquireNextImageKHR has a return code, if it's out of date we want to remake the swapchain
        // we're backing out BEFORE we reset our fence since we didn't want to draw this frame anyways, and if we
        // didn't do that, we'd run into a deadlock because the fence never gets a chance to reset.
        if (result == VK_ERROR_OUT_OF_DATE_KHR)
        {
            recreateSwapChain();
            return;
        }
        else if (result != VK_SUCCESS && result != VK_SUBOPTIMAL_KHR)
        {
            throw std::runtime_error("failed to acquire swap chain image!");
        }

        // reset the fence, it'll be signaled when we dequeue this request
        vkResetFences(device, 1, &inFlightFences[currentFrame]);

        // reset the command buffer so we can write to it. we don't need any special flags, so we'll leave it at 0
        vkResetCommandBuffer(commandBuffers[currentFrame], 0);
        recordCommandBuffer(commandBuffers[currentFrame], imageIndex);

        updateUniformBuffer(currentFrame);

        // our command buffer has been recorded, so now we can finally submit it
        VkSubmitInfo submitInfo{ };
        submitInfo.sType = VK_STRUCTURE_TYPE_SUBMIT_INFO;
        // we want to wait for our image to be available before we write the colour to the swapchain,
        // so we'll wait for the vkAcquireNextImageKHR to finish acquiring an image before we start
        // drawing to it. theoretically this implementation could start executing our vertex shader
        // before we actually acquire an image, which is proabably good? it means that the position
        // of an object could update while the image is being acquired, but the fragment shader can't
        // start until we have an image to draw to, I think. that sounds like it's good.
        VkSemaphore waitSemaphores[]{ imageAvailableSemaphores[currentFrame] };
        VkPipelineStageFlags waitStages[]{ VK_PIPELINE_STAGE_COLOR_ATTACHMENT_OUTPUT_BIT };
        submitInfo.waitSemaphoreCount = 1;
        submitInfo.pWaitSemaphores = waitSemaphores;
        submitInfo.pWaitDstStageMask = waitStages;
        submitInfo.commandBufferCount = 1;
        submitInfo.pCommandBuffers = &commandBuffers[currentFrame];
        VkSemaphore presentSignalSemaphores[]{ renderFinishedSemaphores[currentFrame] };
        submitInfo.signalSemaphoreCount = 1;
        submitInfo.pSignalSemaphores = presentSignalSemaphores;

        // signals the selected fence when this request gets dequeued
        if (vkQueueSubmit(graphicsQueue, 1, &submitInfo, inFlightFences[currentFrame]) != VK_SUCCESS)
        {
            throw std::runtime_error("failed to draw command buffer!");
        }

        VkPresentInfoKHR presentInfo{ };
        presentInfo.sType = VK_STRUCTURE_TYPE_PRESENT_INFO_KHR;
        // we want to wait until our presentation is allowed to happen, so we'll wait until all those
        // semaphores have fired
        presentInfo.waitSemaphoreCount = 1;
        presentInfo.pWaitSemaphores = presentSignalSemaphores;
        // now specify which swapchain will be presenting the images and the index we'd like to present
        // the image to
        VkSwapchainKHR swapChains[]{ swapChain };
        presentInfo.swapchainCount = 1;
        presentInfo.pSwapchains = swapChains;
        presentInfo.pImageIndices = &imageIndex;
        // this is an array of VkResult values for each swap chain we're giving it. it's not necessary
        // to give it anything unless we have more than one swap chain, so we won't use it right now.
        presentInfo.pResults = nullptr; // optional

        result = vkQueuePresentKHR(presentQueue, &presentInfo);

        if (result == VK_ERROR_OUT_OF_DATE_KHR || result == VK_SUBOPTIMAL_KHR || frameBufferResized)
        {
            frameBufferResized = false;
            recreateSwapChain();
        }
        else if (result != VK_SUCCESS)
        {
            throw std::runtime_error("failed to present swap chain image!");
        }

        currentFrame = (currentFrame + 1) % MAX_FRAMES_IN_FLIGHT;
    }

    void updateUniformBuffer(uint32_t currentImage)
    {
        static auto startTime{ std::chrono::high_resolution_clock::now() };

        auto currentTime{ std::chrono::high_resolution_clock::now() };
        float time{ std::chrono::duration<float, std::chrono::seconds::period>(currentTime - startTime).count() };

        UniformBufferObject ubo{
            .model = glm::rotate(glm::mat4(1.f), time * glm::radians(90.f), glm::vec3(0.f, 0.f, 1.f)),
            .view = glm::lookAt(glm::vec3(2.f, 2.f, 2.f), glm::vec3(0.f, 0.f, 0.f), glm::vec3(0.f, 0.f, 1.f)),
            .proj = glm::perspective(glm::radians(45.f), swapChainExtent.width / static_cast<float>(swapChainExtent.height), 0.1f, 10.f)
        };

        // glm is made for OpenGL, which has an inverted Y axis for clip coordinates. we'll just invert that here
        // so our image renders right side up
        ubo.proj[1][1] *= -1;

        // this is not the most efficient way to get frequent data through to shaders, and we should
        // be using push constants. we'll revisit those later
        memcpy(uniformBuffersMapped[currentImage], &ubo, sizeof(ubo));
    }

    void createTextureSampler()
    {
        VkPhysicalDeviceProperties properties;
        vkGetPhysicalDeviceProperties(physicalDevice, &properties);

        VkSamplerCreateInfo samplerInfo{
            .sType = VK_STRUCTURE_TYPE_SAMPLER_CREATE_INFO,
            // magnification filter, VK_FILTER_LINEAR is bilinear filtering. this should only come into play when 
            // the texture isn't big enough to fit the space its trying to fill, so it'll interpolate the four nearest
            // texels when computing the colour.
            .magFilter = VK_FILTER_LINEAR,
            // same as above, but the "minified" filtering behaviour. I think 
            .minFilter = VK_FILTER_LINEAR,
            // not using mipmaps right now, we'll come back to this later
            .mipmapMode = VK_SAMPLER_MIPMAP_MODE_LINEAR,
            // we're not planning on sampling outside the texture in this tutorial, but there's a few options to choose from here.
            // VK_SAMPLER_ADDRESS_MODE_REPEAT means that the texture will repeat when you sample past it (0 -> 100 -> 0 -> 100)
            // VK_SAMPLER_ADDRESS_MODE_MIRRORED_REPEAT means that the texture will mirror when you sample past it (0 -> 100 -> 100 -> 0)
            // VK_SAMPLER_ADDRESS_MODE_CLAMP_TO_EDGE means the texture will keep using the nearest edge colour (0 -> 100 -> 100 -> 100)
            // VK_SAMPLER_ADDRESS_MODE_MIRROR_CLAMP_TO_EDGE will do the same, but use the opposite edge (0 -> 100 -> 0 -> 0)
            // VK_SAMPLER_ADDRESS_MODE_CLAMP_TO_BORDER will return a solid colour past the edge of the image
            .addressModeU = VK_SAMPLER_ADDRESS_MODE_REPEAT,
            .addressModeV = VK_SAMPLER_ADDRESS_MODE_REPEAT,
            .addressModeW = VK_SAMPLER_ADDRESS_MODE_REPEAT,
            .mipLodBias = 0.f,
            .anisotropyEnable = VK_TRUE,
            .maxAnisotropy = properties.limits.maxSamplerAnisotropy,
            // texels will be compared to a value, then the result of that is used in filtering operations. not useful to us right now,
            // but comes up for shadow maps. we'll revisit this later
            .compareEnable = VK_FALSE,
            .compareOp = VK_COMPARE_OP_ALWAYS,
            .minLod = 0.f,
            .maxLod = 0.f,
            // borders can be black, white, or transparent as either floats or ints. this will only matter when the address mode is clamp
            .borderColor = VK_BORDER_COLOR_INT_OPAQUE_BLACK,
            // normalized (false) will be [0, 1) where unnormalized is [0, width / height). we want to use normalized so we can fit any texture
            .unnormalizedCoordinates = VK_FALSE,
        };

        if (vkCreateSampler(device, &samplerInfo, nullptr, &textureSampler) != VK_SUCCESS)
        {
            throw std::runtime_error("failed to create texture sampler!");
        }
    }

    void createTextureImageView()
    {
        textureImageView = createImageView(textureImage, VK_FORMAT_R8G8B8A8_SRGB);
    }

    void createTextureImage()
    {
        int texWidth;
        int texHeight;
        int texChannels;

        stbi_uc* pixels{ stbi_load("textures/nitropan1.png", &texWidth, &texHeight, &texChannels, STBI_rgb_alpha)};
        // our image is width * height * numChannels, which is 4 for STBI_rgb_alpha
        VkDeviceSize imageSize{ static_cast<uint64_t>(texWidth) * static_cast<uint64_t>(texHeight) * 4 };

        if (!pixels)
        {
            throw std::runtime_error("failed to load texture image!");
        }

        VkBuffer stagingBuffer;
        VkDeviceMemory stagingBufferMemory;

        createBuffer(imageSize, VK_BUFFER_USAGE_TRANSFER_SRC_BIT, VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT | VK_MEMORY_PROPERTY_HOST_COHERENT_BIT, stagingBuffer, stagingBufferMemory);

        void* data;
        vkMapMemory(device, stagingBufferMemory, 0, imageSize, 0, &data);
        memcpy(data, pixels, static_cast<size_t>(imageSize));
        vkUnmapMemory(device, stagingBufferMemory);

        stbi_image_free(pixels);

        // usage has the same semantics as a buffer, but we want to be able to transfer data to this memory, then access it from the shader (VK_IMAGE_USAGE_SAMPLED_BIT)
        // there's 2 options for tiling, VK_IMAGE_TILING_LINEAR and VK_IMAGE_TILING_OPTIMAL. you'd only want to use linear for
        // directly accessing texels in memory, so if we were setting this up in a staging image, we'd want to do that.
        // since we're using a staging buffer, we can use the optimal layout so the GPU has an easier time with it on the shader
        // for the format, we should use the same format for the texels as the pixels in the buffer, or the operation fails
        createImage(texWidth, texHeight, VK_FORMAT_R8G8B8A8_SRGB, VK_IMAGE_TILING_OPTIMAL, VK_IMAGE_USAGE_TRANSFER_DST_BIT | VK_IMAGE_USAGE_SAMPLED_BIT, VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT, textureImage, textureImageMemory);

        transitionImageLayout(textureImage, VK_FORMAT_R8G8B8A8_SRGB, VK_IMAGE_LAYOUT_UNDEFINED, VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL);
        copyBufferToImage(stagingBuffer, textureImage, static_cast<uint32_t>(texWidth), static_cast<uint32_t>(texHeight));

        transitionImageLayout(textureImage, VK_FORMAT_R8G8B8A8_SRGB, VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL, VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL);

        vkDestroyBuffer(device, stagingBuffer, nullptr);
        vkFreeMemory(device, stagingBufferMemory, nullptr);
    }

    void createImage(uint32_t width, uint32_t height, VkFormat format, VkImageTiling tiling, VkImageUsageFlags usage, VkMemoryPropertyFlags properties, VkImage& image, VkDeviceMemory& imageMemory)
    {
        VkExtent3D imageExtent{
            .width = static_cast<uint32_t>(width),
            .height = static_cast<uint32_t>(height),
            .depth = 1
        };

        VkImageCreateInfo imageInfo{
            .sType = VK_STRUCTURE_TYPE_IMAGE_CREATE_INFO,
            .flags = 0, // optional
            // 1D and 3D images are also supported, for whatever I might want those for
            .imageType = VK_IMAGE_TYPE_2D,
            .format = format,
            .extent = imageExtent,
            // we haven't generated mipmaps yet, so this should be 1
            .mipLevels = 1,
            // our texture isn't in an array, so we've only got 1 layer
            .arrayLayers = 1,
            // for multisampling, we're ignoring it right now so we can go down to 1 sample (off).
            // it's also only relevant to images used as attachments, not sure what that is but we don't need MSAA.
            .samples = VK_SAMPLE_COUNT_1_BIT,
            .tiling = tiling,
            .usage = usage,
            // only in the queue family that supports graphics, so we can use exclusive. it's the same rules as other exclusive / concurrent things
            .sharingMode = VK_SHARING_MODE_EXCLUSIVE,
            // there's 2 options here too, VK_IMAGE_LAYOUT_UNDEFINED and VK_IMAGE_LAYOUT_PREINITIALIZED. preinitalized preserves texture
            // data during the first transition, which isn't necessary for this use case. if we were using a staging image, we'd want
            // to upload the texel data, then transition the image to be a transfer source and preserve the data there. we aren't
            // doing that so we won't be using that option
            .initialLayout = VK_IMAGE_LAYOUT_UNDEFINED,
        };

        if (vkCreateImage(device, &imageInfo, nullptr, &image) != VK_SUCCESS)
        {
            throw std::runtime_error("failed to create image!");
        }

        VkMemoryRequirements memRequirements;
        vkGetImageMemoryRequirements(device, image, &memRequirements);

        VkMemoryAllocateInfo allocInfo{
            .sType = VK_STRUCTURE_TYPE_MEMORY_ALLOCATE_INFO,
            .allocationSize = memRequirements.size,
            .memoryTypeIndex = findMemoryType(memRequirements.memoryTypeBits, properties)
        };

        if (vkAllocateMemory(device, &allocInfo, nullptr, &textureImageMemory) != VK_SUCCESS)
        {
            throw std::runtime_error("failed to allocate image memory!");
        }

        vkBindImageMemory(device, image, imageMemory, 0);
    }

    void transitionImageLayout(VkImage image, VkFormat format, VkImageLayout oldLayout, VkImageLayout newLayout)
    {
        VkCommandBuffer commandBuffer{ beginSingleTimeCommands() };

        VkImageSubresourceRange subresourceRange{
            .aspectMask = VK_IMAGE_ASPECT_COLOR_BIT,
            // our image does not have mip levels, so the base is 0 and the count is 1
            .baseMipLevel = 0,
            .levelCount = 1,
            // our image is not an array, so the base is 0 and the count is 1
            .baseArrayLayer = 0,
            .layerCount = 1
        };

        VkImageMemoryBarrier barrier{
            .sType = VK_STRUCTURE_TYPE_IMAGE_MEMORY_BARRIER,
            // these will specify which operations that will have to wait on our resource, it's determined
            // by newLayout and oldLayout, which I won't know until I figure it out later
            .srcAccessMask = 0, // TODO
            .dstAccessMask = 0, // TODO
            .oldLayout = oldLayout,
            .newLayout = newLayout,
            // if this barrier is for queue ownership transfer, we would set these values to something else here.
            // since it isn't, we must explicitly set this to ignored so it works properly
            .srcQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED,
            .dstQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED,
            .image = image,
            .subresourceRange = subresourceRange
        };

        VkPipelineStageFlags sourceStage;
        VkPipelineStageFlags destinationStage;

        if (oldLayout == VK_IMAGE_LAYOUT_UNDEFINED && newLayout == VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL)
        {
            barrier.srcAccessMask = 0;
            barrier.dstAccessMask = VK_ACCESS_TRANSFER_WRITE_BIT;

            sourceStage = VK_PIPELINE_STAGE_TOP_OF_PIPE_BIT;
            destinationStage = VK_PIPELINE_STAGE_TRANSFER_BIT;
        }
        else if (oldLayout == VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL && newLayout == VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL)
        {
            barrier.srcAccessMask = VK_ACCESS_TRANSFER_WRITE_BIT;
            barrier.dstAccessMask = VK_ACCESS_SHADER_READ_BIT;

            sourceStage = VK_PIPELINE_STAGE_TRANSFER_BIT;
            destinationStage = VK_PIPELINE_STAGE_FRAGMENT_SHADER_BIT;
        }
        else
        { 
            throw std::invalid_argument("unsupported layout transition!");
        }

        // all pipeline barriers are submitted to the same function, so there's a couple empty fields here that
        // we need to populate anyways. srcStageMask says which pipeline stage occurs before the barrier, and 
        // dstStageMask states which stage must wait on the barrier to complete
        vkCmdPipelineBarrier(
            commandBuffer,
            sourceStage, destinationStage, // TODO HERE
            0,
            0, nullptr,
            0, nullptr,
            1, &barrier
        );

        endSingleTimeCommands(commandBuffer);
    }
    
    void copyBufferToImage(VkBuffer buffer, VkImage image, uint32_t width, uint32_t height)
    {
        VkCommandBuffer commandBuffer{ beginSingleTimeCommands() };

        VkBufferImageCopy region{
            .bufferOffset = 0,
            .bufferRowLength = 0,
            .bufferImageHeight = 0,
            .imageSubresource = {
                .aspectMask = VK_IMAGE_ASPECT_COLOR_BIT,
                .mipLevel = 0,
                .baseArrayLayer = 0,
                .layerCount = 1
            },
            .imageOffset = { 0, 0, 0 },
            .imageExtent = {
                .width = width,
                .height = height,
                .depth = 1
            }
        };

        vkCmdCopyBufferToImage(
            commandBuffer,
            buffer,
            image,
            VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL,
            1,
            &region
        );

        endSingleTimeCommands(commandBuffer);
    }

    void createBuffer(VkDeviceSize size, VkBufferUsageFlags usage, VkMemoryPropertyFlags properties, VkBuffer& buffer, VkDeviceMemory& bufferMemory)
    {
        VkBufferCreateInfo bufferInfo{ };
        bufferInfo.sType = VK_STRUCTURE_TYPE_BUFFER_CREATE_INFO;
        // says how large our buffer is
        bufferInfo.size = size;
        // since bufferInfo is generic, we need to describe how we're using this one
        bufferInfo.usage = usage;
        // same as the swap chain image view, this buffer could be accessed by a specific queue family or 
        // multiple. since this is just for the graphics queue, we can keep it exclusive
        bufferInfo.sharingMode = VK_SHARING_MODE_EXCLUSIVE;

        if (vkCreateBuffer(device, &bufferInfo, nullptr, &buffer) != VK_SUCCESS)
        {
            throw std::runtime_error("failed to create vertex buffer!");
        }

        // don't need to initialize, the next function sets it up
        VkMemoryRequirements memRequirements;
        vkGetBufferMemoryRequirements(device, buffer, &memRequirements);

        VkMemoryAllocateInfo allocInfo{ };
        allocInfo.sType = VK_STRUCTURE_TYPE_MEMORY_ALLOCATE_INFO;
        allocInfo.allocationSize = memRequirements.size;
        // try to find memory that matches our memory type requirements
        allocInfo.memoryTypeIndex = findMemoryType(memRequirements.memoryTypeBits, properties);

        // we shouldn't be using this for every single buffer we want to allocate. in an actual application,
        // we might be doing this thousands of times, maybe one for each object. in that case, we'd want to
        // use as few as possible, but instead use the offset parameter we've been sticking on 0 for this
        // whole program. The tutorial suggests creating a custom allocator (I will try this) or using
        // https://github.com/GPUOpen-LibrariesAndSDKs/VulkanMemoryAllocator instead as a time saver
        if (vkAllocateMemory(device, &allocInfo, nullptr, &bufferMemory) != VK_SUCCESS)
        {
            throw std::runtime_error("failed to allocate buffer memory!");
        }

        // memory is allocated specifically for this buffer, so we can use a 0 offset. if the offset is nonzero,
        // then the offset needs to be divisible by memRequirements.alignment. I'd assume that we use an offset if 
        // we have literally any other buffers, we just happen to not
        vkBindBufferMemory(device, buffer, bufferMemory, 0);
    }

    void copyBuffer(VkBuffer srcBuffer, VkBuffer dstBuffer, VkDeviceSize bufferSize)
    {
        VkCommandBuffer commandBuffer{ beginSingleTimeCommands() };

        VkBufferCopy copyRegion{
            .srcOffset = 0,
            .dstOffset = 0,
            .size = bufferSize
        };
        vkCmdCopyBuffer(commandBuffer, srcBuffer, dstBuffer, 1, &copyRegion);

        endSingleTimeCommands(commandBuffer);
    }

    VkCommandBuffer beginSingleTimeCommands()
    {
        // funky c++ stuff, all the member variables must be in order, but I like this style. have to
        // change the rest when I bring most of this code out into other spots
        VkCommandBufferAllocateInfo allocInfo{
            .sType = VK_STRUCTURE_TYPE_COMMAND_BUFFER_ALLOCATE_INFO,
            .commandPool = commandPool,
            .level = VK_COMMAND_BUFFER_LEVEL_PRIMARY,
            .commandBufferCount = 1
        };

        // alternatively, we could create a command pool so we could execute some number of memory transfer 
        // buffers at once, in addition to letting the driver do a little more optimisation. I'm just sticking
        // to the tutorial for now, but this is probably good to keep in mind.
        // it says we'd want to be using VK_COMMAND_POOL_CREATE_TRANSIENT_BIT as a flag for that
        VkCommandBuffer commandBuffer;
        vkAllocateCommandBuffers(device, &allocInfo, &commandBuffer);

        VkCommandBufferBeginInfo beginInfo{
            .sType = VK_STRUCTURE_TYPE_COMMAND_BUFFER_BEGIN_INFO,
            // we're only submitting this command buffer once, so we'll tell the driver about it
            .flags = VK_COMMAND_BUFFER_USAGE_ONE_TIME_SUBMIT_BIT
        };
        vkBeginCommandBuffer(commandBuffer, &beginInfo);

        return commandBuffer;
    }

    void endSingleTimeCommands(VkCommandBuffer commandBuffer)
    {
        vkEndCommandBuffer(commandBuffer);

        VkSubmitInfo submitInfo{
            .sType = VK_STRUCTURE_TYPE_SUBMIT_INFO,
            .commandBufferCount = 1,
            .pCommandBuffers = &commandBuffer
        };

        // we'll submit our commands here, but not wait on a fence. Instead, we'll halt the program here until the 
        // queue's finished going through to the GPU, then continue whatever else we were up to.
        // we have to wait for our command buffer to finish doing everything before we can free it up anyways
        // if we were submitting multiple buffers at the same time, it'd probably make more sense to use 
        // a fence so the driver can figure out the best way to submit these queues and save us a couple cycles
        vkQueueSubmit(graphicsQueue, 1, &submitInfo, VK_NULL_HANDLE);
        vkQueueWaitIdle(graphicsQueue);

        vkFreeCommandBuffers(device, commandPool, 1, &commandBuffer);
    }

    VkImageView createImageView(VkImage image, VkFormat format)
    {
        VkImageViewCreateInfo viewInfo{
            .sType = VK_STRUCTURE_TYPE_IMAGE_VIEW_CREATE_INFO,
            .image = image,
            .viewType = VK_IMAGE_VIEW_TYPE_2D,
            .format = format,
            .subresourceRange = {
                .aspectMask = VK_IMAGE_ASPECT_COLOR_BIT,
                .baseMipLevel = 0,
                .levelCount = 1,
                .baseArrayLayer = 0,
                .layerCount = 1
            }
        };

        VkImageView imageView;
        if (vkCreateImageView(device, &viewInfo, nullptr, &imageView) != VK_SUCCESS)
        {
            throw std::runtime_error("failed to create texture image vie");
        }

        return imageView;
    }

    void createIndexBuffer()
    {
        VkDeviceSize bufferSize{ sizeof(indices[0]) * indices.size() };
        VkBuffer stagingBuffer;
        VkDeviceMemory stagingBufferMemory;
        createBuffer(bufferSize, VK_BUFFER_USAGE_TRANSFER_SRC_BIT, VK_MEMORY_PROPERTY_HOST_COHERENT_BIT | VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT, stagingBuffer, stagingBufferMemory);

        void* data;
        vkMapMemory(device, stagingBufferMemory, 0, bufferSize, 0, &data);
        memcpy(data, indices.data(), (size_t) bufferSize);
        vkUnmapMemory(device, stagingBufferMemory);

        createBuffer(bufferSize, VK_BUFFER_USAGE_INDEX_BUFFER_BIT | VK_BUFFER_USAGE_TRANSFER_DST_BIT, VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT, indexBuffer, indexBufferMemory);
        copyBuffer(stagingBuffer, indexBuffer, bufferSize);

        vkDestroyBuffer(device, stagingBuffer, nullptr);
        vkFreeMemory(device, stagingBufferMemory, nullptr);
    }

    void createVertexBuffer()
    {
        VkDeviceSize bufferSize{ sizeof(vertices[0]) * vertices.size() };
        // create a vertex buffer that says we want to use it as a staging buffer, and we
        // can access and read the memory from our CPU
        VkBuffer stagingBuffer;
        VkDeviceMemory stagingBufferMemory;
        createBuffer(bufferSize, VK_BUFFER_USAGE_TRANSFER_SRC_BIT, VK_MEMORY_PROPERTY_HOST_COHERENT_BIT | VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT, stagingBuffer, stagingBufferMemory);

        // create a map to our vertex buffer memory, copy our vertex data into it, then unmap
        // the memory as we don't need to have access to it anymore
        void* data;
        vkMapMemory(device, stagingBufferMemory, 0, bufferSize, 0, &data);
        memcpy(data, vertices.data(), (size_t) bufferSize);
        vkUnmapMemory(device, stagingBufferMemory);

        createBuffer(bufferSize, VK_BUFFER_USAGE_VERTEX_BUFFER_BIT | VK_BUFFER_USAGE_TRANSFER_DST_BIT, VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT, vertexBuffer, vertexBufferMemory);
        copyBuffer(stagingBuffer, vertexBuffer, bufferSize);

        vkDestroyBuffer(device, stagingBuffer, nullptr);
        vkFreeMemory(device, stagingBufferMemory, nullptr);
    }

    uint32_t findMemoryType(uint32_t typeFilter, VkMemoryPropertyFlags properties)
    {
        // don't need to initialize, the next function sets it up
        VkPhysicalDeviceMemoryProperties memProperties;
        vkGetPhysicalDeviceMemoryProperties(physicalDevice, &memProperties);

        for (uint32_t i{ 0 }; i < memProperties.memoryTypeCount; i++)
        {
            // this is a little complicated. the memProperties check is just making sure that each property is filtered
            // to definitely match, and all of them match exactly. it'll find anything that at least matches our properties.
            // typeFilter is just going to say which memory type we want to be using. I'd imagine that it just works out
            // that each index is equal to 0b1, then 0b10, then 0b100, and so on. we're just looking for any one of those
            // types to match and it'll be valid memory
            if ((typeFilter & (1 << i)) && (memProperties.memoryTypes[i].propertyFlags & properties) == properties)
            {
                return i;
            }
        }

        throw std::runtime_error("failed to find suitable memory type!");
    }

    void createSyncObjects()
    {
        imageAvailableSemaphores.resize(MAX_FRAMES_IN_FLIGHT);
        renderFinishedSemaphores.resize(MAX_FRAMES_IN_FLIGHT);
        inFlightFences.resize(MAX_FRAMES_IN_FLIGHT);
        // both semaphores and fences need create info, but there are no useful operations
        // or definitions past this for semaphores. it seems to mostly be for consistency between object
        // creations and potential changes in the future
        VkSemaphoreCreateInfo semaphoreInfo{ };
        semaphoreInfo.sType = VK_STRUCTURE_TYPE_SEMAPHORE_CREATE_INFO;

        VkFenceCreateInfo fenceInfo{ };
        fenceInfo.sType = VK_STRUCTURE_TYPE_FENCE_CREATE_INFO;
        // this will start our fence as signalled, so our first frame should render without any issues.
        // then it'll auto handle it for each frame.
        fenceInfo.flags = VK_FENCE_CREATE_SIGNALED_BIT;

        for (size_t i = 0; i < MAX_FRAMES_IN_FLIGHT; ++i)
        {
            if (vkCreateSemaphore(device, &semaphoreInfo, nullptr, &imageAvailableSemaphores[i]) != VK_SUCCESS ||
                vkCreateSemaphore(device, &semaphoreInfo, nullptr, &renderFinishedSemaphores[i]) != VK_SUCCESS ||
                vkCreateFence(device, &fenceInfo, nullptr, &inFlightFences[i]) != VK_SUCCESS)
            {
                throw std::runtime_error("failed to create one or more syncronization objects");
            }
        }
    }

    void recordCommandBuffer(VkCommandBuffer commandBuffer, uint32_t imageIndex)
    {
        // always start recording a command buffer with begin info, it'll supply some extra info that we need.
        VkCommandBufferBeginInfo beginInfo{ };
        beginInfo.sType = VK_STRUCTURE_TYPE_COMMAND_BUFFER_BEGIN_INFO;
        // there's a few flags we could be using
        // VK_COMMAND_BUFFER_USAGE_ONE_TIME_SUBMIT_BIT - the command buffer will be rerecorded immediately after executing it once
        // VK_COMMAND_BUFFER_USAGE_RENDER_PASS_CONTINUE_BIT - marks it as a secondary command buffer that will be entirely in one render pass
        // VK_COMMAND_BUFFER_USAGE_SIMULTANEOUS_USE_BIT - the command buffer can be resubmitted while it's queued
        // none of these are relevant right now, but they're good to keep in mind
        beginInfo.flags = 0; // optional
        beginInfo.pInheritanceInfo = nullptr; // optional

        if (vkBeginCommandBuffer(commandBuffer, &beginInfo) != VK_SUCCESS)
        {
            throw std::runtime_error("failed to begin recording command buffer!");
        }

        VkRenderPassBeginInfo renderPassInfo{ };
        renderPassInfo.sType = VK_STRUCTURE_TYPE_RENDER_PASS_BEGIN_INFO;
        renderPassInfo.renderPass = renderPass;
        // we created a framebuffer for each swap chain image, and it's specified as a colour attachment.
        // we're going to need to bind the framebuffer for that swapchain image, and we do that here
        // using the imageIndex that we pass in to pick the correct framebuffer
        renderPassInfo.framebuffer = swapChainFramebuffers[imageIndex];
        // the render area should match the size of the attachments for best performance, so we'll
        // just use no offset and the size of our swap chain extent to determine the render area
        renderPassInfo.renderArea.offset = { 0, 0 };
        renderPassInfo.renderArea.extent = swapChainExtent;
        // these value will define clear values for VK_ATTACHMENT_LOAD_OP_CLEAR, which we're using for 
        // our load operation on a colour attachment
        VkClearValue clearColour{ {{ 0.f, 0.f, 0.f, 1.f }} };
        renderPassInfo.clearValueCount = 1;
        renderPassInfo.pClearValues = &clearColour;

        // starts the render pass! won't draw anything to the screen yet, but we've defined that 
        // the render pass may begin. there's a few alternatives for the VkSubpassContents value
        // VK_SUBPASS_CONTENTS_INLINE - the render pass commands will be embedded in the primary command buffer and there are no secondary buffers
        // VK_SUBPASS_CONTENTS_SECONDARY_COMMAND_BUFFERS - the render pass commands will be executed from secondary command buffers
        // this program has no secondary buffers, so this works for now.
        vkCmdBeginRenderPass(commandBuffer, &renderPassInfo, VK_SUBPASS_CONTENTS_INLINE);

        vkCmdBindPipeline(commandBuffer, VK_PIPELINE_BIND_POINT_GRAPHICS, graphicsPipeline);

        // define our viewport. I think it'll always have an extent of [0, swapChainExtent.w/h], and we should
        // just trust whatever that value is
        VkViewport viewport{ };
        viewport.x = 0.f;
        viewport.y = 0.f;
        viewport.width = static_cast<float>(swapChainExtent.width);
        viewport.height = static_cast<float>(swapChainExtent.height);
        viewport.minDepth = 0.f;
        viewport.maxDepth = 1.f;
        vkCmdSetViewport(commandBuffer, 0, 1, &viewport);

        // our scissor rect, which can clip parts of the screen. we'll keep it to the same size as our viewport rect for now.
        VkRect2D scissor{ };
        scissor.offset = { 0, 0 };
        scissor.extent = { swapChainExtent };
        vkCmdSetScissor(commandBuffer, 0, 1, &scissor);

        VkBuffer vertexBuffers[]{ { vertexBuffer } };
        VkDeviceSize vertexOffsets[]{ {0} };
        vkCmdBindVertexBuffers(commandBuffer, 0, 1, vertexBuffers, vertexOffsets);

        // bind our index buffer. we can only create one index buffer, so I'd assume that we need to 
        // latch each index buffer onto whatever vertex buffer it corresponds to. but now each vertex buffer
        // only has to list out its vertices once, which is pretty exciting!
        vkCmdBindIndexBuffer(commandBuffer, indexBuffer, 0, VK_INDEX_TYPE_UINT16);
        
        // bind our descriptor sets for our uniform buffers we described. in this case it's only an MVP transform
        vkCmdBindDescriptorSets(commandBuffer, VK_PIPELINE_BIND_POINT_GRAPHICS, pipelineLayout, 0, 1, &descriptorSets[currentFrame], 0, nullptr);

        // drawing indexed allows for me to specify the first index, the first vertex, and the first instance of the thing we're drawing.
        // I'm not exactly sure how that interacts with multiple vertex buffers, but with a single index buffer we should 
        // be able to describe multiple objects, probably just different vertex buffers for each of them, and then whatever the 
        // fuck instanced rendering is about we can definitely do something with that. all neat shit.
        // apparently this is called aliasing and some vulkan functions will have explicit flags for doing exactly that
        vkCmdDrawIndexed(commandBuffer, static_cast<uint32_t>(indices.size()), 1, 0, 0, 0);

        vkCmdEndRenderPass(commandBuffer);

        if (vkEndCommandBuffer(commandBuffer) != VK_SUCCESS)
        {
            throw std::runtime_error("failed to record command buffer!");
        }
    }

    void createFramebuffers()
    {
        swapChainFramebuffers.resize(swapChainImageViews.size());

        for (size_t i = 0; i < swapChainImageViews.size(); ++i)
        {
            VkImageView attachments[]{
                swapChainImageViews[i]
            };

            VkFramebufferCreateInfo framebufferInfo{ };
            framebufferInfo.sType = VK_STRUCTURE_TYPE_FRAMEBUFFER_CREATE_INFO;
            framebufferInfo.renderPass = renderPass;
            // the list of image views for some attachments. I believe they'll correspond to the attachment
            // indices in our renderPass, but we only have one anyways so it works out like this
            framebufferInfo.attachmentCount = 1;
            framebufferInfo.pAttachments = attachments;
            framebufferInfo.width = swapChainExtent.width;
            framebufferInfo.height = swapChainExtent.height;
            // how many layers are in the image arrays. our swapchain only has single images, so this is 1
            framebufferInfo.layers = 1;

            if (vkCreateFramebuffer(device, &framebufferInfo, nullptr, &swapChainFramebuffers[i]) != VK_SUCCESS)
            {
                throw std::runtime_error("failed to create framebuffer!");
            }
        }
    }
    
    void createCommandBuffers()
    {
        commandBuffers.resize(MAX_FRAMES_IN_FLIGHT);
        // command buffers are automatically freed when their pool is destroyed, so we
        // don't need to explicitly destroy it later
        VkCommandBufferAllocateInfo allocInfo{ };
        allocInfo.sType = VK_STRUCTURE_TYPE_COMMAND_BUFFER_ALLOCATE_INFO;
        allocInfo.commandPool = commandPool;
        // there's two different levels a command buffer can be. VK_COMMAND_BUFFER_LEVEL_PRIMARY and
        // VK_COMMAND_BUFFER_LEVEL_SECONDARY. primary command buffers can't be called from other command
        // buffers, but may be submitted to a queue. secondary command buffers may be called from primary
        // buffers, but may not be submitted to a pool directly. we're using a primary buffer since
        // there's only one.
        allocInfo.level = VK_COMMAND_BUFFER_LEVEL_PRIMARY;
        allocInfo.commandBufferCount = static_cast<uint32_t>(commandBuffers.size());

        if (vkAllocateCommandBuffers(device, &allocInfo, commandBuffers.data()) != VK_SUCCESS)
        {
            throw std::runtime_error("failed to allocate command buffers!");
        }
    }

    void createCommandPool()
    {
        QueueFamiyIndices queueFamilyIndices{ findQueueFamilies(physicalDevice) };

        VkCommandPoolCreateInfo poolInfo{ };
        poolInfo.sType = VK_STRUCTURE_TYPE_COMMAND_POOL_CREATE_INFO;
        // there are two available flags, VK_COMMAND_POOL_CREATE_RESET_COMMAND_BUFFER_BIT and
        // VK_COMMAND_POOL_CREATE_TRANSIENT_BIT. the former will allow command buffers to be rerecorded
        // individually, while the latter hints that command buffers will be rerecorded with new commands often
        // we're changing the command buffer every frame (probably like most renderers?) so we'd like to
        // be able to reset the command buffer and record over it. apparently, this isn't often enough
        // to require the other flag for performance issues, and the convenience is better
        poolInfo.flags = VK_COMMAND_POOL_CREATE_RESET_COMMAND_BUFFER_BIT;
        poolInfo.queueFamilyIndex = queueFamilyIndices.graphicsFamily.value();

        if (vkCreateCommandPool(device, &poolInfo, nullptr, &commandPool) != VK_SUCCESS)
        {
            throw std::runtime_error("failed to create command pool!");
        }
    }

    void createRenderPass()
    {
        // create a single colour buffer attachment, which will represent one image from the swap chain
        VkAttachmentDescription colorAttachment{ };
        // we just want to copy the image format that our swap chain describes
        colorAttachment.format = swapChainImageFormat;
        // we're also not interested in MSAA right now, so a sample of 1 is appropriate
        colorAttachment.samples = VK_SAMPLE_COUNT_1_BIT;
        // this will clear the framebuffer to black before drawing a new frame on it. should make it much easier
        // to actually render things to the screen without worrying about what the screen was before
        colorAttachment.loadOp = VK_ATTACHMENT_LOAD_OP_CLEAR;
        // this says that we're going to store our colour data into our colour buffer. exciting! triangles!
        colorAttachment.storeOp = VK_ATTACHMENT_STORE_OP_STORE;
        // we don't care about stencils right now
        colorAttachment.stencilLoadOp = VK_ATTACHMENT_LOAD_OP_DONT_CARE;
        colorAttachment.stencilStoreOp = VK_ATTACHMENT_STORE_OP_DONT_CARE;
        // this is saying that we're not expecting any image format when we recieve an image in this render pass
        // it doesn't matter, since we're clearing the whole image first anyways
        colorAttachment.initialLayout = VK_IMAGE_LAYOUT_UNDEFINED;
        // and this is saying we're going to be outputting an image that can go onto the swap chain
        colorAttachment.finalLayout = VK_IMAGE_LAYOUT_PRESENT_SRC_KHR;

        // we also need to describe sub passes for post processing effects. we're only gonna want one for 
        // this program since we're just trying to draw a triangle to the screen.

        // this is an attachment reference for our first subpass. it'll go into an index that corresponds to
        // our shader's output colour and have an optimal layout, which will provide the best performance for a colour buffer
        VkAttachmentReference colorAttachmentRef{ };
        // this is the index of our colour attachment in the render pass array, and also where our shaders can write to by location(n)
        colorAttachmentRef.attachment = 0; 
        colorAttachmentRef.layout = VK_IMAGE_LAYOUT_COLOR_ATTACHMENT_OPTIMAL;

        // we'll describe our subpass here, it's a graphics subpass so it'll have VK_PIPELINE_BIND_POINT_GRAPHICS
        // for its bind port. because we've put a colour attachment ref in slot 0, our fragment shader'll be able 
        // to output to it. sick!
        VkSubpassDescription subpass{ };
        subpass.pipelineBindPoint = VK_PIPELINE_BIND_POINT_GRAPHICS;
        subpass.colorAttachmentCount = 1;
        subpass.pColorAttachments = &colorAttachmentRef;
        // a subpass can also reference a bunch of other things, like
        // InputAttachments - read from a shader
        // ResolveAttachments - colour multisampling
        // DepthStencilAttachment - depth and stencil data
        // PreserveAttachments - not used by the subpass, but the data must be preserved through it

        // create a dependency for our renderpass to wait for our image to be available before trying to get it.
        // we know our image is available as soon as we get to the VK_PIPELINE_STAGE_COLOR_ATTACHMENT_OUTPUT_BIT
        // since we're waiting for that with our semaphore, so we'll use that
        VkSubpassDependency dependency{ };
        // src/dst Subpass refers to the implicit subpass that happens before or after this render pass, respectively.
        // dstSubpass must always be a higher number than srcSubpass, unless src is defined as VK_SUBPASS_EXTERNAL, like
        // we have here. 0 refers to our subpass that we've described above, which is the first and only explicit subpass.
        dependency.srcSubpass = VK_SUBPASS_EXTERNAL;
        dependency.dstSubpass = 0;
        // wait on our colour output stage, since we know the image will be ready when we get there
        // the colour attachment stage doesn't happen in the fragment shader! it's actually later in
        // the pipeline, for what that's worth.
        dependency.srcStageMask = VK_PIPELINE_STAGE_COLOR_ATTACHMENT_OUTPUT_BIT;
        dependency.srcAccessMask = 0;
        // the operations that we should be waiting for happen in the colour attachment stage and 
        // involve writing to the colour attachment. these settings will prevent this from happening
        // until we're actually allowed to write colours.
        dependency.dstStageMask = VK_PIPELINE_STAGE_COLOR_ATTACHMENT_OUTPUT_BIT;
        dependency.dstAccessMask = VK_ACCESS_COLOR_ATTACHMENT_WRITE_BIT;

        // this is our actual render pass info that gets used through the program, and it'll keep track of 
        // all our attachments and all of our subpasses
        VkRenderPassCreateInfo renderPassInfo{ };
        renderPassInfo.sType = VK_STRUCTURE_TYPE_RENDER_PASS_CREATE_INFO;
        renderPassInfo.attachmentCount = 1;
        renderPassInfo.pAttachments = &colorAttachment;
        renderPassInfo.subpassCount = 1;
        renderPassInfo.pSubpasses = &subpass;
        renderPassInfo.dependencyCount = 1;
        renderPassInfo.pDependencies = &dependency;

        if (vkCreateRenderPass(device, &renderPassInfo, nullptr, &renderPass) != VK_SUCCESS)
        {
            throw std::runtime_error("failed to create render pass!");
        }
    }
    
    void createGraphicsPipline()
    {
        // it's possible to auto-compile source files, these should be fed as a part of project setup or smth rather than being hardcoded
        std::vector<char> vertShaderCode{ { utils::readFile("shaders/vert.spv") } };
        std::vector<char> fragShaderCode{ { utils::readFile("shaders/frag.spv") } };

        VkShaderModule vertShaderModule{ createShaderModule(device, vertShaderCode) };
        VkShaderModule fragShaderModule{ createShaderModule(device, fragShaderCode) };

        VkPipelineShaderStageCreateInfo vertShaderStageInfo{ };
        vertShaderStageInfo.sType = VK_STRUCTURE_TYPE_PIPELINE_SHADER_STAGE_CREATE_INFO;
        vertShaderStageInfo.stage = VK_SHADER_STAGE_VERTEX_BIT;
        vertShaderStageInfo.module = vertShaderModule;
        vertShaderStageInfo.pName = "main";
        vertShaderStageInfo.pSpecializationInfo = nullptr;

        VkPipelineShaderStageCreateInfo fragShaderStageInfo{ };
        fragShaderStageInfo.sType = VK_STRUCTURE_TYPE_PIPELINE_SHADER_STAGE_CREATE_INFO;
        fragShaderStageInfo.stage = VK_SHADER_STAGE_FRAGMENT_BIT;
        fragShaderStageInfo.module = fragShaderModule;
        fragShaderStageInfo.pName = "main";
        fragShaderStageInfo.pSpecializationInfo = nullptr;

        // set up our programmable shader stages and jam them in here
        VkPipelineShaderStageCreateInfo shaderStages[]{ 
            vertShaderStageInfo, 
            fragShaderStageInfo 
        };

        // get our vertex descriptions
        auto bindingDescription{ Vertex::getBindingDescription() };
        auto attributeDescriptions{ Vertex::getAttributeDescriptions() };

        // describes how our vertices our to our pipeline. doesn't actually contain vertex data
        VkPipelineVertexInputStateCreateInfo vertexInputInfo{ };
        vertexInputInfo.sType = VK_STRUCTURE_TYPE_PIPELINE_VERTEX_INPUT_STATE_CREATE_INFO;
        vertexInputInfo.vertexBindingDescriptionCount = 1;
        vertexInputInfo.pVertexBindingDescriptions = &bindingDescription;
        vertexInputInfo.vertexAttributeDescriptionCount = static_cast<uint32_t>(attributeDescriptions.size());
        vertexInputInfo.pVertexAttributeDescriptions = attributeDescriptions.data();

        // inputAssembly.topology is how we describe every single bit of geometry our program will take in.
        // to actually know what we should be using, we'll have to look at our 3D models and see what system they use.
        // some systems will have advantages over another (like, TRIANGLE_LIST is just a bunch of triangles but
        // TRIANGLE_STRIP will construct an object while reusing vertices from the previous triangle) 
        VkPipelineInputAssemblyStateCreateInfo inputAssembly{ };
        inputAssembly.sType = VK_STRUCTURE_TYPE_PIPELINE_INPUT_ASSEMBLY_STATE_CREATE_INFO;
        inputAssembly.topology = VK_PRIMITIVE_TOPOLOGY_TRIANGLE_LIST;
        inputAssembly.primitiveRestartEnable = VK_FALSE;

        // vulkan allows us to access some of the pipeline state at runtime instead of setting it up beforehand
        // we'll need to initialize this somewhere, I'm not sure where that happens right now
        std::vector<VkDynamicState> dynamicStates{ {
            VK_DYNAMIC_STATE_VIEWPORT,
            VK_DYNAMIC_STATE_SCISSOR
        } };

        VkPipelineDynamicStateCreateInfo dynamicState{ };
        dynamicState.sType = VK_STRUCTURE_TYPE_PIPELINE_DYNAMIC_STATE_CREATE_INFO;
        dynamicState.dynamicStateCount = static_cast<uint32_t>(dynamicStates.size());
        dynamicState.pDynamicStates = dynamicStates.data();

        // if we have a dyaminc viewport and scissor, we don't need to specify what they are
        // within this struct, just how many we have
        VkPipelineViewportStateCreateInfo viewportState{ };
        viewportState.sType = VK_STRUCTURE_TYPE_PIPELINE_VIEWPORT_STATE_CREATE_INFO;
        viewportState.viewportCount = 1;
        viewportState.scissorCount = 1;

        // this is how we define what we're actaully drawing to the screen. it'll perform
        // depth testing, face culling, and a scissor test. It can also describe how we render 
        // fragments, like if we just want a wireframe or the whole shape
        VkPipelineRasterizationStateCreateInfo rasterizer{ };
        rasterizer.sType = VK_STRUCTURE_TYPE_PIPELINE_RASTERIZATION_STATE_CREATE_INFO;
        // if this is set to VK_TRUE, fragments beyond the near and far plane become clamped instead.
        // we don't really want this for normal geometry, but it might be useful for shadow maps
        // using this requires enabling a GPU feature
        rasterizer.depthClampEnable = VK_FALSE;
        // the exciting bit! this is how we describe how we want our triangles to be shown on the screen
        // the default for most things will be the default below, but there's some other options too, such as
        // VK_POLYGON_MODE_LINE and VK_POLYGON_MODE_POINT which will draw all the lines or all the points as expected
        // using anything except for VK_POLYGON_MODE_FILL requires enabling a GPU feature
        rasterizer.polygonMode = VK_POLYGON_MODE_FILL;
        // describes how thick our fragment lines are. anything above 1.0f requires the GPU to have wideLines on
        rasterizer.lineWidth = 1.f;
        // back face culling. we could cull nothing, cull front faces, or cull everything. this makes the most 
        // sense for most applications
        rasterizer.cullMode = VK_CULL_MODE_BACK_BIT;
        // this will describe what triangles are considered the front face by thier winding order. VK_FRONT_FACE_CLOCKWISE
        // is the default for vulkan, but since we're inverting the projection matrix's Y coordinate, it needs to be COUNTER_CLOCKWISE
        rasterizer.frontFace = VK_FRONT_FACE_COUNTER_CLOCKWISE;
        // this is adjusting the depth of a vertex by biasing them by some constant, or adjusting them based on their slope.
        // it's not that useful for showing geometry, but it could help for displaying shadows
        rasterizer.depthBiasEnable = VK_FALSE;
        rasterizer.depthBiasConstantFactor = 0.f; // optional
        rasterizer.depthBiasClamp = 0.f; // optional
        rasterizer.depthBiasSlopeFactor = 0.f; // optional

        // MSAA, with a sample size of 1 bit. I think that means it's not doing anything but we may as well have 
        // some kind of antialiasing set up. this will require enabling a GPU feature
        VkPipelineMultisampleStateCreateInfo multisampling{ };
        multisampling.sType = VK_STRUCTURE_TYPE_PIPELINE_MULTISAMPLE_STATE_CREATE_INFO;
        multisampling.sampleShadingEnable = VK_FALSE;
        multisampling.rasterizationSamples = VK_SAMPLE_COUNT_1_BIT;
        multisampling.minSampleShading = 1.f; // optional
        multisampling.pSampleMask = nullptr; // optional
        multisampling.alphaToCoverageEnable = VK_FALSE; // optional
        multisampling.alphaToOneEnable = VK_FALSE; // optional

        // this describes the colour blending rules for a framebuffer. we only have one framebuffer so we 
        // only need to describe this once
        VkPipelineColorBlendAttachmentState colorBlendAttachment{ };
        colorBlendAttachment.colorWriteMask = VK_COLOR_COMPONENT_R_BIT | VK_COLOR_COMPONENT_G_BIT | VK_COLOR_COMPONENT_B_BIT | VK_COLOR_COMPONENT_A_BIT;
        colorBlendAttachment.blendEnable = VK_TRUE;
        // none of the following settings will do anything if blendEnable is set to VK_FALSE
        // if we set src and dst ColorBlendFactors to these, we will have some normal alpha blending.
        // VK_BLEND_FACTOR_SRC_ALPHA says that we're going to multiply our colour by our alpha value, then
        // VK_BLEND_FACTOR_ONE_MINUS_SRC_ALPHA will do the same operaton (but 1 - a) on the destination colour
        colorBlendAttachment.srcColorBlendFactor = VK_BLEND_FACTOR_SRC_ALPHA; // optional, different than default
        colorBlendAttachment.dstColorBlendFactor = VK_BLEND_FACTOR_ONE_MINUS_SRC_ALPHA; // optional, different than default
        colorBlendAttachment.colorBlendOp = VK_BLEND_OP_ADD; // optional
        colorBlendAttachment.srcAlphaBlendFactor = VK_BLEND_FACTOR_ONE; // optional
        colorBlendAttachment.dstAlphaBlendFactor = VK_BLEND_FACTOR_ZERO; // optional
        colorBlendAttachment.alphaBlendOp = VK_BLEND_OP_ADD; // optional

        // this describes an array of structs for all the framebuffers and allows blend constants to apply to
        // all the ColorBlendAttachmentStates. 
        VkPipelineColorBlendStateCreateInfo colorBlending{ };
        colorBlending.sType = VK_STRUCTURE_TYPE_PIPELINE_COLOR_BLEND_STATE_CREATE_INFO;
        // if this is set to VK_TRUE, we'll use the logicOp for all blending instead of the ColorBlendAttachment definition
        // for blending. ColorBlendAttachment will effectively be set to VK_FALSE
        colorBlending.logicOpEnable = VK_FALSE;
        colorBlending.logicOp = VK_LOGIC_OP_COPY; // optional
        colorBlending.attachmentCount = 1;
        colorBlending.pAttachments = &colorBlendAttachment;
        colorBlending.blendConstants[0] = 0.f; // optional
        colorBlending.blendConstants[1] = 0.f; // optional
        colorBlending.blendConstants[2] = 0.f; // optional
        colorBlending.blendConstants[3] = 0.f; // optional

        // describes our pipeline layout so we can set uniform values within shaders. We're not using any right now,
        // so our setLayoutCount is 0 and we do not have SetLayouts. this also describes push constants, which
        // are a different way of passing dymanic values to shaders
        VkPipelineLayoutCreateInfo pipelineLayoutInfo{ };
        pipelineLayoutInfo.sType = VK_STRUCTURE_TYPE_PIPELINE_LAYOUT_CREATE_INFO;
        pipelineLayoutInfo.setLayoutCount = 1;
        // I didn't think about this, but apparently this being multiple descriptor sets is interesting.
        // we can use descriptor pools to define a bunch of descriptor sets to do something, maybe
        // with multiple shaders or if it makes more sense to have different uniform buffers on a single shader
        pipelineLayoutInfo.pSetLayouts = &descriptorSetLayout;
        pipelineLayoutInfo.pushConstantRangeCount = 0;
        pipelineLayoutInfo.pPushConstantRanges = nullptr;

        if (vkCreatePipelineLayout(device, &pipelineLayoutInfo, nullptr, &pipelineLayout) != VK_SUCCESS)
        {
            throw std::runtime_error("failed to create pipeline layout!");
        }

        // our pipeline info!! this'll combine all the previous descriptions we set up and allow vulkan
        // to use those as instructions for how it'll handle drawing to the screen
        VkGraphicsPipelineCreateInfo pipelineInfo{ };
        pipelineInfo.sType = VK_STRUCTURE_TYPE_GRAPHICS_PIPELINE_CREATE_INFO;
        // set up our shader stage creation info
        pipelineInfo.stageCount = 2;
        pipelineInfo.pStages = shaderStages;
        // set up our fixed function states (and dynamic state, which ig is actually fixed and provides a reference to dynamic states)
        pipelineInfo.pVertexInputState = &vertexInputInfo;
        pipelineInfo.pInputAssemblyState = &inputAssembly;
        pipelineInfo.pViewportState = &viewportState;
        pipelineInfo.pRasterizationState = &rasterizer;
        pipelineInfo.pMultisampleState = &multisampling;
        pipelineInfo.pDepthStencilState = nullptr;
        pipelineInfo.pColorBlendState = &colorBlending;
        pipelineInfo.pDynamicState = &dynamicState;
        // then our pipeline layout. this is a vulkan handle rather than a struct pointer since we
        // need to instantiate our pipeline layout ourselves anyways.
        pipelineInfo.layout = pipelineLayout;
        // then tell it about our render pass and which subpass index this pipeline info is going to refer to
        // we can also use other render passes in this pipeline if they're compatible with the current renderPass.
        // more info on compatibility at https://registry.khronos.org/vulkan/specs/1.3-extensions/html/chap8.html#renderpass-compatibility
        pipelineInfo.renderPass = renderPass;
        pipelineInfo.subpass = 0;
        // our pipelineinfo can also just inherit values from another pipeline. interesting and I'm sure that's used 
        // a lot elsewhere, it just doesn't matter for this program. The idea is that it's less expensive to swap
        // pipelines if you can share similar values between them
        pipelineInfo.basePipelineHandle = VK_NULL_HANDLE;
        pipelineInfo.basePipelineIndex = -1;

        if (vkCreateGraphicsPipelines(device, VK_NULL_HANDLE, 1, &pipelineInfo, nullptr, &graphicsPipeline) != VK_SUCCESS)
        {
            throw std::runtime_error("failed to create the graphics pipeline!");
        }
 
        // destroy the shader modules we made as we don't need them past this setup
        vkDestroyShaderModule(device, vertShaderModule, nullptr);
        vkDestroyShaderModule(device, fragShaderModule, nullptr);

        // we could also describe a depth / stencil buffer, and we'd configure those with 
        // VkPipelineDepthStencilCreateInfo. we're not using that for right now so we'll ignore it
    }

    void createDescriptorSets()
    {
        std::vector<VkDescriptorSetLayout> layouts{ MAX_FRAMES_IN_FLIGHT, descriptorSetLayout };

        VkDescriptorSetAllocateInfo allocInfo{
            .sType = VK_STRUCTURE_TYPE_DESCRIPTOR_SET_ALLOCATE_INFO,
            .descriptorPool = descriptorPool,
            .descriptorSetCount = static_cast<uint32_t>(MAX_FRAMES_IN_FLIGHT),
            .pSetLayouts = layouts.data()
        };

        descriptorSets.resize(MAX_FRAMES_IN_FLIGHT);

        if (vkAllocateDescriptorSets(device, &allocInfo, descriptorSets.data()) != VK_SUCCESS)
        {
            throw std::runtime_error("failed to allocate descriptor sets!");
        }

        for (size_t i{ 0 }; i < MAX_FRAMES_IN_FLIGHT; ++i)
        {
            VkDescriptorBufferInfo bufferInfo{
                .buffer = uniformBuffers[i],
                .offset = 0,
                .range = sizeof(UniformBufferObject)
            };

            VkWriteDescriptorSet descriptorWrite{
                .sType = VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET,
                .dstSet = descriptorSets[i],
                // this is the binding index, which we assigned to 0 in the shader
                .dstBinding = 0,
                // the uniform buffer we're writing to isn't an array, so this should also be 0
                .dstArrayElement = 0,
                // this is how many descriptors we want to update at once, in our case it's just one since we only have one element
                .descriptorCount = 1,
                .descriptorType = VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER,
                // we should only be using one of these at a time, in this case we're updating uniform buffers so we'll populate pBufferInfo
                // pImageInfo is used for descriptors that refer to image data,
                // pBufferInfo is used for descriptors that refer to buffer data (we're using this)
                // pTexelBufferView is used for descriptors that refer to buffer views
                .pImageInfo = nullptr, // optional
                .pBufferInfo = &bufferInfo,
                .pTexelBufferView = nullptr, // optional
            };

            // the last parameter is used for copying descriptors to each other, which we aren't doing so it doesn't matter
            // we just want to update our one descriptor set using the parameters we described above
            vkUpdateDescriptorSets(device, 1, &descriptorWrite, 0, nullptr);
        }
    }

    void createDescriptorPool()
    {
        VkDescriptorPoolSize poolSize{ 
            .type = VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER,
            .descriptorCount = static_cast<uint32_t>(MAX_FRAMES_IN_FLIGHT)
        };

        VkDescriptorPoolCreateInfo poolInfo{
            .sType = VK_STRUCTURE_TYPE_DESCRIPTOR_POOL_CREATE_INFO,
            // describes the maximum number of descriptor sets we'll be allowed to use
            .maxSets = static_cast<uint32_t>(MAX_FRAMES_IN_FLIGHT),
            .poolSizeCount = 1,
            .pPoolSizes = &poolSize
            // there's an optional flag similar to command pools that says if individual descriptor sets are allowed to
            // be freed or not; VK_DESCRIPTOR_POOL_CREATE_FREE_DESCRIPTOR_SET_BIT. we won't be touching 
            // the descriptor sets after we create them, so we can leave the flags field to its default of 0
        };

        if (vkCreateDescriptorPool(device, &poolInfo, nullptr, &descriptorPool) != VK_SUCCESS)
        {
            throw std::runtime_error("failed to create descriptor pool!");
        }
    }

    void createUniformBuffers()
    {
        VkDeviceSize bufferSize{ sizeof(UniformBufferObject) };

        uniformBuffers.resize(MAX_FRAMES_IN_FLIGHT);
        uniformBuffersMemory.resize(MAX_FRAMES_IN_FLIGHT);
        uniformBuffersMapped.resize(MAX_FRAMES_IN_FLIGHT);

        for (size_t i{ 0 }; i < MAX_FRAMES_IN_FLIGHT; ++i)
        {
            createBuffer(bufferSize, VK_BUFFER_USAGE_UNIFORM_BUFFER_BIT, VK_MEMORY_PROPERTY_HOST_COHERENT_BIT | VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT, uniformBuffers[i], uniformBuffersMemory[i]);
            vkMapMemory(device, uniformBuffersMemory[i], 0, bufferSize, 0, &uniformBuffersMapped[i]);
        }
    }

    void createDescriptorSetLayout()
    {
        VkDescriptorSetLayoutBinding uboLayoutBinding{
            .binding = 0,
            .descriptorType = VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER,
            // our shader variable could be an array of objects, so we'd need to specify that there's more
            // than one descriptor here. this descriptor wouldn't contain any of that data though
            .descriptorCount = 1,
            // this descriptor's going to apply to the vertex shader stage, so we have to say that
            .stageFlags = VK_SHADER_STAGE_VERTEX_BIT,
            // relevant for image sampling related descriptors. not sure what that means but we'll revisit it later
            .pImmutableSamplers = nullptr // optional
        };

        VkDescriptorSetLayoutCreateInfo layoutInfo{ 
            .sType = VK_STRUCTURE_TYPE_DESCRIPTOR_SET_LAYOUT_CREATE_INFO,
            .bindingCount = 1,
            .pBindings = &uboLayoutBinding
        };

        if (vkCreateDescriptorSetLayout(device, &layoutInfo, nullptr, &descriptorSetLayout) != VK_SUCCESS)
        {
            throw std::runtime_error("failed to create descriptor set layout!");
        }
    }

    VkShaderModule createShaderModule(VkDevice device, const std::vector<char>& code)
    {
        VkShaderModuleCreateInfo createInfo{ };
        createInfo.sType = VK_STRUCTURE_TYPE_SHADER_MODULE_CREATE_INFO;
        createInfo.codeSize = code.size();
        createInfo.pCode = reinterpret_cast<const uint32_t*>(code.data());

        VkShaderModule shaderModule;
        if (vkCreateShaderModule(device, &createInfo, nullptr, &shaderModule) != VK_SUCCESS)
        {
            throw std::runtime_error("failed to create shader module!");
        }

        return shaderModule;
    }

    void createImageViews()
    {
        // set up our image views to match the swap chain image size
        swapChainImageViews.resize(swapChainImages.size());

        for (size_t i = 0; i < swapChainImages.size(); ++i)
        {
            swapChainImageViews[i] = createImageView(swapChainImages[i], swapChainImageFormat);
        }
    }

    void createSwapChain()
    {
        // figure out our swap chain support details
        SwapChainSupportDetails swapChainSupport{ querySwapChainSupport(physicalDevice) };

        // set up the related objects, auto choosing the "best" implementation
        VkSurfaceFormatKHR surfaceFormat{ chooseSwapSurfaceFormat(swapChainSupport.formats) };
        VkPresentModeKHR presentMode{ chooseSwapPresentMode(swapChainSupport.presentModes) };
        VkExtent2D extent{ chooseSwapExtent(swapChainSupport.capabilities) };

        // set the number of images we would like to have in the swap chain.
        // the minimum image count means we're probably waiting a long time for the driver to finish the operation,
        // which will lead to stutter. the higher this value is, the worse latency will be
        uint32_t imageCount{ swapChainSupport.capabilities.minImageCount + 1 };

        // make sure we don't accidentally set our image count too high. 0 means there is no max.
        if (swapChainSupport.capabilities.maxImageCount > 0 && imageCount > swapChainSupport.capabilities.maxImageCount)
        {
            imageCount = swapChainSupport.capabilities.maxImageCount;
        }

        VkSwapchainCreateInfoKHR createInfo{ };
        createInfo.sType = VK_STRUCTURE_TYPE_SWAPCHAIN_CREATE_INFO_KHR;
        createInfo.surface = surface;
        createInfo.minImageCount = imageCount;
        createInfo.imageFormat = surfaceFormat.format;
        createInfo.imageColorSpace = surfaceFormat.colorSpace;
        createInfo.imageExtent = extent;
        // how many layers each image has. this is only ever 1 unless we're creating a stereoscopic image (VR)
        createInfo.imageArrayLayers = 1;
        // this means we will directly render the images we are providing.
        // we could use a value like VK_IMAGE_USAGE_TRANSFER_DST_BIT to post-process an image before 
        // using a memory operation to swap it back in
        createInfo.imageUsage = VK_IMAGE_USAGE_COLOR_ATTACHMENT_BIT;

        QueueFamiyIndices indices{ findQueueFamilies(physicalDevice) };
        uint32_t queueFamilyIndices[] {
            indices.graphicsFamily.value(),
            indices.presentFamily.value()
        };

        // if our graphics and present queueFamilies are not in the same queue, we have some different logic
        if (indices.graphicsFamily.value() != indices.presentFamily.value())
        {
            // this says that we can share an image across multiple families and we don't need to transfer ownership
            // we can get around this by explicitly transferring ownership, we might be coming back to this later.
            createInfo.imageSharingMode = VK_SHARING_MODE_CONCURRENT;
            createInfo.queueFamilyIndexCount = 2;
            createInfo.pQueueFamilyIndices = queueFamilyIndices;
        }
        else
        {
            // this says that only one queue family can own an image at a time. this is the best performance so we'll try to prefer it.
            createInfo.imageSharingMode = VK_SHARING_MODE_EXCLUSIVE;
            createInfo.queueFamilyIndexCount = 0; // optional, probably defaults to this
            createInfo.pQueueFamilyIndices = nullptr; // optional, probably defaults to this
        }

        // we can actually pre transform any image going into the swap chain. use currentTransform to not do that.
        createInfo.preTransform = swapChainSupport.capabilities.currentTransform;
        // specifies what the alpha channel is doing when blending with other windows. VK_COMPOSITE_ALPHA_OPAQUE_BIT_KHR means it'll ignore this.
        // maybe we could make a window stencil by saying it'll blend, then we could display whatever the hell we want for a window.
        createInfo.compositeAlpha = VK_COMPOSITE_ALPHA_OPAQUE_BIT_KHR;
        createInfo.presentMode = presentMode;
        // clipped = true means that we do not care about pixels that are covered by another window
        createInfo.clipped = VK_TRUE;
        // the swap chain can become invalid, possibly because the window was resized. when that happens, we need to make a new swap chain
        // and provide a reference to the old one here. we're ignoring this for now.
        createInfo.oldSwapchain = VK_NULL_HANDLE;

        if (vkCreateSwapchainKHR(device, &createInfo, nullptr, &swapChain) != VK_SUCCESS)
        {
            throw std::runtime_error("could not create a valid swap chain!");
        }

        // we need some way to references the images created by the swap chain, so we'll set that up here
        vkGetSwapchainImagesKHR(device, swapChain, &imageCount, nullptr);
        swapChainImages.resize(imageCount);
        vkGetSwapchainImagesKHR(device, swapChain, &imageCount, swapChainImages.data());

        // swapChainImageFormat is to set up the ImageViews properly
        swapChainImageFormat = surfaceFormat.format;
        swapChainExtent = extent;
    }

    // platform-agnostic glfw library to create a window surface and store it in the member variable
    void createSurface()
    {
        if (glfwCreateWindowSurface(instance, window, nullptr, &surface) != VK_SUCCESS)
        {
            throw std::runtime_error("could not create glfw window surface!");
        }
    }

    // create a logical device that we're going to send commands to
    void createLogicalDevice()
    {
        // find the queue families and populate a struct with that info
        QueueFamiyIndices indices{ findQueueFamilies(physicalDevice) };

        std::vector<VkDeviceQueueCreateInfo> queueCreateInfos;
        std::set<uint32_t> uniqueQueueFamilies{
            indices.graphicsFamily.value(),
            indices.presentFamily.value()
        };

        // this is a value between 0.f and 1.f, where 1 is the highest priority and 0 is the lowest. 1 is probably fine for everything until it becomes a performance issue
        float queuePriority{ 1.0f };

        // initialize each queueCreateInfo with its own queueFamily
        for (uint32_t queueFamiliy : uniqueQueueFamilies)
        {
            VkDeviceQueueCreateInfo queueCreateInfo{ };
            queueCreateInfo.sType = VK_STRUCTURE_TYPE_DEVICE_QUEUE_CREATE_INFO;
            queueCreateInfo.queueFamilyIndex = queueFamiliy;
            // we only have 1 queue in each queueCreateInfo, so keep this at 1
            queueCreateInfo.queueCount = 1;
            queueCreateInfo.pQueuePriorities = &queuePriority;
            queueCreateInfos.push_back(queueCreateInfo);
        }

        // create our logical device info
        VkPhysicalDeviceFeatures deviceFeatures{ };
        VkDeviceCreateInfo createInfo{ };
        createInfo.sType = VK_STRUCTURE_TYPE_DEVICE_CREATE_INFO;
        createInfo.pQueueCreateInfos = queueCreateInfos.data();
        createInfo.queueCreateInfoCount = static_cast<uint32_t>(queueCreateInfos.size());
        createInfo.pEnabledFeatures = &deviceFeatures;
        createInfo.enabledExtensionCount = static_cast<uint32_t>(deviceExtensions.size());
        createInfo.ppEnabledExtensionNames = deviceExtensions.data();

        // debug, if we have validation layers on then add those to our createInfo
        if (enableValidationLayers)
        {
            createInfo.enabledLayerCount = static_cast<uint32_t>(validationLayers.size());
            createInfo.ppEnabledLayerNames = validationLayers.data();
        }
        else
        {
            createInfo.enabledLayerCount = 0;
        }

        // actually create the device and store it in our member variable device
        if (vkCreateDevice(physicalDevice, &createInfo, nullptr, &device) != VK_SUCCESS)
        {
            throw std::runtime_error("failed to create a logical device!");
        }

        // get the device queues for all the queues we're planning to use and store them in our member variables
        vkGetDeviceQueue(device, indices.graphicsFamily.value(), 0, &graphicsQueue);
        vkGetDeviceQueue(device, indices.presentFamily.value(), 0, &presentQueue);
    }

    // pick which physical device we want to use
    void pickPhysicalDevice()
    {
        // this is a standard way to optimize setup, call our vkEnumerate function with no data object to get the deviceCount first
        uint32_t deviceCount{ 0 };
        vkEnumeratePhysicalDevices(instance, &deviceCount, nullptr);

        if (deviceCount == 0)
        {
            throw std::runtime_error("failed to find GPUs wit Vulkan support!");
        }

        // which will allow us to have a vector that's exactly the right size
        std::vector<VkPhysicalDevice> devices( deviceCount );
        vkEnumeratePhysicalDevices(instance, &deviceCount, devices.data());

        for (const auto& device : devices)
        {
            // take the first suitable device, we could rank each device by a score and return the best one,
            // but we probably don't need to 99% of the time anyways.
            if (isDeviceSuitable(device))
            {
                physicalDevice = device;
                break;
            }
        }

        if (physicalDevice == VK_NULL_HANDLE)
        {
            throw std::runtime_error("failed to find a suitable GPU");
        }
    }

    // the device is only suitable if it supports the minimum requirement for what we want to do.
    // if anything fails in queue setup for some reason, the device will also fail to be suitable
    bool isDeviceSuitable(VkPhysicalDevice device)
    {
        QueueFamiyIndices indices{ findQueueFamilies(device) };

        bool extensionsSupported{ checkDeviceExtensionSupport(device) };

        VkPhysicalDeviceFeatures supportedFeatures;
        vkGetPhysicalDeviceFeatures(device, &supportedFeatures);

        bool swapChainAdequate{ false };
        if (extensionsSupported)
        {
            SwapChainSupportDetails swapChainSupport{ querySwapChainSupport(device) };
            swapChainAdequate = !swapChainSupport.formats.empty() && !swapChainSupport.presentModes.empty();
        }

        return indices.isComplete() && extensionsSupported && swapChainAdequate && supportedFeatures.samplerAnisotropy;

        // could add more info here, like iterating through all the devices and getting the best, or displaying them.
        // all that matters right now is that we have a device that can run vulkan
    }

    // figure out if our device can support a swap chain. this will fail on devices that do not 
    // have a display output. also returns the swapchain support details with related info
    SwapChainSupportDetails querySwapChainSupport(VkPhysicalDevice device) const
    {
        SwapChainSupportDetails details;

        vkGetPhysicalDeviceSurfaceCapabilitiesKHR(device, surface, &details.capabilities);

        uint32_t formatCount;
        vkGetPhysicalDeviceSurfaceFormatsKHR(device, surface, &formatCount, nullptr);

        if (formatCount != 0)
        {
            details.formats.resize(formatCount);
            vkGetPhysicalDeviceSurfaceFormatsKHR(device, surface, &formatCount, details.formats.data());
        }

        uint32_t presentModeCount;
        vkGetPhysicalDeviceSurfacePresentModesKHR(device, surface, &presentModeCount, nullptr);

        if (presentModeCount != 0)
        {
            details.presentModes.resize(presentModeCount);
            vkGetPhysicalDeviceSurfacePresentModesKHR(device, surface, &presentModeCount, details.presentModes.data());
        }

        return details;
    }

    // choose which swap surface we want to be using
    VkSurfaceFormatKHR chooseSwapSurfaceFormat(const std::vector<VkSurfaceFormatKHR>& availableFormats)
    {
        for (const auto& availableFormat : availableFormats)
        {
            if (availableFormat.format == VK_FORMAT_B8G8R8A8_SRGB && availableFormat.colorSpace == VK_COLORSPACE_SRGB_NONLINEAR_KHR)
            {
                return availableFormat;
            }

            // could rank how good formats are here
        }

        // but the first one is probably fine if we don't support srgb
        return availableFormats[0];
    }

    VkPresentModeKHR chooseSwapPresentMode(const std::vector<VkPresentModeKHR>& availablePresentModes)
    {
        for (const auto& availablePresentMode : availablePresentModes)
        {
            // triple buffering, where the program is allowed to present new frames at any speed but the frames are only
            // shown when a display refreshes
            if (availablePresentMode == VK_PRESENT_MODE_MAILBOX_KHR)
            {
                return availablePresentMode;
            }
        }

        // standard v-sync
        return VK_PRESENT_MODE_FIFO_KHR;

        // there are two other modes, 
        // VK_PRESENT_MODE_IMMEDIATE_KHR
        // VK_PRESENT_MODE_FIFO_RELAXED_KHR
        // immediate is no v-sync, display the frame immediately
        // fifo relaxed is standard v-sync, but if the program is rendering too slowly it will be allowed to display
        // right to the screen instead, kinda the worst of both immediate and v-sync
    }

    // determine how big our screen is
    VkExtent2D chooseSwapExtent(const VkSurfaceCapabilitiesKHR& capabilities)
    {
        // we have special handling if the width or height are set to the max value
        // but if it's not, just use the current capabilites extent
        // width and height should be maxed out if the screen coordinates don't match up with the available pixels
        if (capabilities.currentExtent.width != std::numeric_limits<uint32_t>::max())
        {
            return capabilities.currentExtent;
        }

        int width;
        int height;

        // gets the frame buffer size in pixels
        glfwGetFramebufferSize(window, &width, &height);

        VkExtent2D actualExtent {
            static_cast<uint32_t>(width),
            static_cast<uint32_t>(height)
        };

        // clamp the values so they can't be bigger than the screen capabilities
        actualExtent.width = std::clamp(actualExtent.width, capabilities.minImageExtent.width, capabilities.maxImageExtent.width);
        actualExtent.height = std::clamp(actualExtent.height, capabilities.minImageExtent.height, capabilities.maxImageExtent.height);

        // return the size of the screen in pixels
        return actualExtent;
    }

    bool checkDeviceExtensionSupport(VkPhysicalDevice device)
    {
        uint32_t extensionCount;
        vkEnumerateDeviceExtensionProperties(device, nullptr, &extensionCount, nullptr);

        std::vector<VkExtensionProperties> availableExtensions( extensionCount );
        vkEnumerateDeviceExtensionProperties(device, nullptr, &extensionCount, availableExtensions.data());

        // make an unordered set of our required extensions using the list we made at the top of this file
        std::set<std::string> requiredExtensions{ deviceExtensions.begin(), deviceExtensions.end() };

        for (const auto& extension : availableExtensions)
        {
            requiredExtensions.erase(extension.extensionName);
        }

        // our device supports all the extensions if we successfully erased everything
        return requiredExtensions.empty();
    }

    QueueFamiyIndices findQueueFamilies(VkPhysicalDevice device) const
    {
        QueueFamiyIndices indices{ };

        uint32_t queueFamilyCount{ 0 };
        vkGetPhysicalDeviceQueueFamilyProperties(device, &queueFamilyCount, nullptr);

        std::vector<VkQueueFamilyProperties> queueFamilies( queueFamilyCount );
        vkGetPhysicalDeviceQueueFamilyProperties(device, &queueFamilyCount, queueFamilies.data());

        // go through all our queueFamilies
        for (uint32_t i{ 0 }; i < queueFamilyCount; ++i)
        {
            // if the queueFamily has a bit flag that lines up with VK_QUEUE_GRAPHICS_BIT, it is our graphics family
            if (queueFamilies[i].queueFlags & VK_QUEUE_GRAPHICS_BIT)
            {
                indices.graphicsFamily = i;
            }

            // if the queueFamily passes vkGetPhysicalDeviceSurfaceSupportKHR, it is our present family
            VkBool32 presentSupport{ false };
            vkGetPhysicalDeviceSurfaceSupportKHR(device, i, surface, &presentSupport);

            if (presentSupport)
            { 
                indices.presentFamily = i;
            }

            // both of these families could have the same index, so we need to check both on every loop
            // this could introduce a problem where we've already set one of them, and then for some reason it
            // will get set again later. I don't know the consequences of this, but it's probably performance.
            // after reading a bit more, it looks like it's good if these queues share an index, so we actually
            // would like it to try and reassign if it can
            if (indices.isComplete())
            {
                break;
            }
        }

        return indices;
    }

    // boilerplate to populate a debugInfo create struct
    void populateDebugMessengerCreateInfo(VkDebugUtilsMessengerCreateInfoEXT& createInfo)
    {
        createInfo = { };
        createInfo.sType = VK_STRUCTURE_TYPE_DEBUG_UTILS_MESSENGER_CREATE_INFO_EXT;
        createInfo.messageSeverity = VK_DEBUG_UTILS_MESSAGE_SEVERITY_VERBOSE_BIT_EXT | VK_DEBUG_UTILS_MESSAGE_SEVERITY_WARNING_BIT_EXT | VK_DEBUG_UTILS_MESSAGE_SEVERITY_ERROR_BIT_EXT;
        createInfo.messageType = VK_DEBUG_UTILS_MESSAGE_TYPE_GENERAL_BIT_EXT | VK_DEBUG_UTILS_MESSAGE_TYPE_VALIDATION_BIT_EXT | VK_DEBUG_UTILS_MESSAGE_TYPE_PERFORMANCE_BIT_EXT;
        createInfo.pfnUserCallback = debugCallback;
    }

    void setupDebugMessenger()
    {
        // if we aren't debugging, we do not want the debug messenger to start up
        if (!enableValidationLayers)
        {
            return;
        }

        VkDebugUtilsMessengerCreateInfoEXT createInfo{ };
        populateDebugMessengerCreateInfo(createInfo);
        createInfo.pUserData = nullptr;

        if (CreateDebugUtilsMessengerEXT(instance, &createInfo, nullptr, &debugMessenger) != VK_SUCCESS)
        {
            throw std::runtime_error("failed to setup debug messenger!");
        }
    }

    void createInstance()
    {
        if (enableValidationLayers && !checkValidationLayerSupport())
        {
            throw std::runtime_error("validation layers requested, but not available!");
        }

        // appinfo boilerplate for the OS and graphics driver to know what's going on
        VkApplicationInfo appInfo{ };
        appInfo.sType = VK_STRUCTURE_TYPE_APPLICATION_INFO;
        appInfo.pApplicationName = "Hello Triangle";
        appInfo.applicationVersion = VK_MAKE_API_VERSION(0, 1, 0, 0);
        appInfo.pEngineName = "No Engine";
        appInfo.engineVersion = VK_MAKE_API_VERSION(0, 1, 0, 0);
        appInfo.apiVersion = VK_API_VERSION_1_3;

        // our actual instance createInfo
        VkInstanceCreateInfo createInfo{ };
        // we make this here so it doesn't go out of scope before we set up our instance
        VkDebugUtilsMessengerCreateInfoEXT debugCreateInfo{ };
        createInfo.sType = VK_STRUCTURE_TYPE_INSTANCE_CREATE_INFO;
        createInfo.pApplicationInfo = &appInfo;
        if (enableValidationLayers)
        {
            createInfo.enabledLayerCount = static_cast<uint32_t>(validationLayers.size());
            createInfo.ppEnabledLayerNames = validationLayers.data();

            // a temporary debug messenger so we can grab some info about the program startup. our main debug 
            // messenger will take over after we set it up
            populateDebugMessengerCreateInfo(debugCreateInfo);
            createInfo.pNext = (VkDebugUtilsMessengerCreateInfoEXT*)&debugCreateInfo;
        }
        else
        {
            createInfo.enabledLayerCount = 0;
            createInfo.pNext = nullptr;
        }
        std::vector<const char*> extensions{ { getRequiredExtensions() } };
        createInfo.enabledExtensionCount = static_cast<uint32_t>(extensions.size());
        createInfo.ppEnabledExtensionNames = extensions.data();

        uint32_t availableExtensionCount{ 0 };
        std::vector<VkExtensionProperties> availableExtensions{ { showAvailableExtensions() } };

        showIfRequiredExtensionsAreAvailable(extensions, availableExtensions);

        // create the instance. let's go!!!!!
        if (vkCreateInstance(&createInfo, nullptr, &instance) != VK_SUCCESS)
        {
            throw std::runtime_error("failed to create instance!");
        }
    }

    // finds every available extension for our system
    std::vector<VkExtensionProperties> showAvailableExtensions()
    {
        uint32_t extensionCount = 0;
        vkEnumerateInstanceExtensionProperties(nullptr, &extensionCount, nullptr);

        std::vector<VkExtensionProperties> extensions( extensionCount );
        vkEnumerateInstanceExtensionProperties(nullptr, &extensionCount, extensions.data());

        return extensions;
    }

    // shows if requiredExtensions are contained within availableExtensions and displays that info to the standard output stream
    int showIfRequiredExtensionsAreAvailable(std::vector<const char*> requiredExtensions, std::vector<VkExtensionProperties> availableExtensions)
    {
        int numSupportedRequiredExtensions{ 0 };
        std::cout << "checking if required extensions are supported by the device:\n";

        for (const auto& extension : requiredExtensions)
        {
            bool isAvailable{ false };
            for (const auto& available : availableExtensions)
            {
                if (isAvailable = (strcmp(available.extensionName, extension) == 0))
                {
                    std::cout << "\t" << "supported: " << available.extensionName << "\n";
                    ++numSupportedRequiredExtensions;
                    break;
                }
            }

            if (!isAvailable)
            {
                std::cerr << "\t" << "unsupported: " << extension << "\n";
            }
        }

        std::cout << std::endl;

        return numSupportedRequiredExtensions;
    }

    // check to make sure we support all our debug validation layers
    bool checkValidationLayerSupport()
    {
        uint32_t layerCount;
        vkEnumerateInstanceLayerProperties(&layerCount, nullptr);

        std::vector<VkLayerProperties> availableLayers( layerCount );
        vkEnumerateInstanceLayerProperties(&layerCount, availableLayers.data());

        std::cout << "checking if required layers are supported by the device:\n";

        for (const char* layerName : validationLayers)
        {
            bool layerFound{ false };

            for (const auto& layerProperties : availableLayers)
            {
                if (strcmp(layerName, layerProperties.layerName) == 0)
                {
                    std::cout << "\t" << "supported: " << layerProperties.layerName << "\n";
                    layerFound = true;
                    break;
                }
            }

            if (!layerFound)
            {
                std::cerr << "\t" << "unsupported: " << layerName << std::endl;
                return false;
            }
        }

        std::cout << std::endl;

        return true;
    }

    // gets all the required extensions for glfw
    std::vector<const char*> getRequiredExtensions()
    {
        uint32_t glfwExtensionCount{ 0 };
        const char** glfwExtensions{ glfwGetRequiredInstanceExtensions(&glfwExtensionCount) };

        std::vector<const char*> extensions( glfwExtensions, glfwExtensions + glfwExtensionCount );

        // if we're in debug, this is also a required extension
        if (enableValidationLayers)
        {
            extensions.push_back(VK_EXT_DEBUG_UTILS_EXTENSION_NAME);
        }

        return extensions;
    }
};

int main() 
{
    HelloTriangleApplication app;

    try 
    {
        app.run();
    }
    catch (const std::exception& e) 
    {
        std::cerr << e.what() << std::endl;
        return EXIT_FAILURE;
    }

    return EXIT_SUCCESS;
}