#version 450

layout(binding = 0) uniform UniformBufferObject
{
	mat4 model;
	mat4 view;
	mat4 proj;
} ubo;


// both of these variables are 32 bits, so they'll use one slot.
// something like dvec2 would be 64 bits, which would use 2 slots
// and leave inColour at location 2.
layout(location = 0) in vec2 inPosition;
layout(location = 1) in vec3 inColour;

layout(location = 0) out vec3 fragColour;

void main()
{
	gl_Position = ubo.proj * ubo.view * ubo.model * vec4(inPosition, 0.0, 1.0);
	fragColour = inColour;
}