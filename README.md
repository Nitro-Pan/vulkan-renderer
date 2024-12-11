# A Vulkan Rendering Backend / Game Engine
This is a project to see how deep into engine development I can get.

## How to run this program
I've built this in Visual Studio 2022, so you'll probably want to open it in that.
Adjust the following in your `C/C++ > General > Additional Include Directories`:
- [tiny_obj_loader](https://github.com/tinyobjloader/tinyobjloader/blob/release/tiny_obj_loader.h)
- [Vulkan SDK](https://vulkan.lunarg.com/)
- [glfw](https://www.glfw.org/)
- [stb_image](https://github.com/nothings/stb/blob/master/stb_image.h)
They should have paths defined to help you out, but point them to the correct place on your machine.

Make sure you have the Vulkan and glfw binaries in `Linker > General > Additional Library Directories` or the linker will complain.