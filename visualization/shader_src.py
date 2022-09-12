vertex_src = """
# version 330 core
layout(location=0) in vec3 vertex_position;
layout(location=1) in vec3 vertex_color;
uniform mat4 view;
uniform mat4 projection;
uniform mat4 model;
out vec3 f_color;
void main(){
    gl_Position=projection*view*model*vec4(vertex_position.x*2, vertex_position.y*2, vertex_position.z*2, 1.0);
    f_color = vertex_color;
}
"""

fragment_src = """
# version 330 core
in vec3 f_color;
uniform int color_flag;
uniform vec3 color;
out vec4 FragColor;

void main(){
    switch(color_flag){
        case 0:
            FragColor=vec4(cos(f_color.xyz + 0.5), 1.0);
            break;
        case 1:
            FragColor=vec4(f_color.xyz, 1.0);
            break;
        case 2:
            FragColor=vec4(0.4, 0.8, 0.5+f_color.z/80, 1.0);
            break;
            };
}
"""
