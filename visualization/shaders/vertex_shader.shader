#version 460 core

layout(location=0) in vec4 vertex;
layout(location=1) in vec4 color;

out vec4 rgb;

uniform mat4 view;
uniform mat4 projection;
uniform mat4 model;

void main() {
    gl_Position = projection*view*model*vec4(vertex.xyz, 1.0);
    float a= color.x;
    float b= color.y;
    float c= color.z;
    float d= color.w;
    float e= 1.0 - a - b - c - d;
    rgb = vec4(a+b, b+c+d, d+e, 1.0);
}
