#version 460 compatibility


//  (101629, 100, 4)
layout(std430, binding=1) buffer Data{
    vec4 Vertices[];  // ~305mb

};
layout(std140, binding=2) buffer Tensor{
    mat4x4 Buffer[];  // ~1220mb, x, x', x'', x''', x'''', y, y', ..., z'''', 0
};
layout(local_size_x=1, local_size_y=1, local_size_z=1) in;


uint gid = gl_GlobalInvocationID.x;
int index = int(gid);

void derivative(int vertex_id){
    float x_0;
    float x_1;
    float x_2;
    float x_3;
    float x_4;
    float y_0;
    float y_1;
    float y_2;
    float y_3;
    float y_4;
    float z_0;
    float z_1;
    float z_2;
    float z_3;
    float z_4;
    for(int t=4; t<100; ++t){
        x_0 = Vertices[vertex_id*100+t-0].y;
        x_1 = Vertices[vertex_id*100+t-1].y;
        x_2 = Vertices[vertex_id*100+t-2].y;
        x_3 = Vertices[vertex_id*100+t-3].y;
        x_4 = Vertices[vertex_id*100+t-4].y;

        y_0 = Vertices[vertex_id*100+t-0].z;
        y_1 = Vertices[vertex_id*100+t-1].z;
        y_2 = Vertices[vertex_id*100+t-2].z;
        y_3 = Vertices[vertex_id*100+t-3].z;
        y_4 = Vertices[vertex_id*100+t-4].z;

        z_0 = Vertices[vertex_id*100+t-0].w;
        z_1 = Vertices[vertex_id*100+t-1].w;
        z_2 = Vertices[vertex_id*100+t-2].w;
        z_3 = Vertices[vertex_id*100+t-3].w;
        z_4 = Vertices[vertex_id*100+t-4].w;


        Buffer[vertex_id*100+t][0].x = x_0;  // x
        Buffer[vertex_id*100+t][0].y = (x_0-x_1)/0.01;  // x'
        Buffer[vertex_id*100+t][0].z = (x_0-2*x_1+x_2)/0.01/0.01;  // x''
        Buffer[vertex_id*100+t][0].w = (x_0-3*x_1+3*x_2-x_3)/0.01/0.01/0.01;  // x'''
        Buffer[vertex_id*100+t][1].x = (x_0-4*x_1+6*x_2-4*x_3+x_4)/0.01/0.01/0.01/0.01;  // x''''

        Buffer[vertex_id*100+t][1].y = y_0;  // y
        Buffer[vertex_id*100+t][1].z = (y_0-y_1)/0.01;  // y'
        Buffer[vertex_id*100+t][1].w = (y_0-2*y_1+y_2)/0.01/0.01;  // y''
        Buffer[vertex_id*100+t][2].x = (y_0-3*y_1+3*y_2-y_3)/0.01/0.01/0.01;  // y'''
        Buffer[vertex_id*100+t][2].y = (y_0-4*y_1+6*y_2-4*y_3+y_4)/0.01/0.01/0.01/0.01;  // y''''

        Buffer[vertex_id*100+t][2].z = z_0;  // z
        Buffer[vertex_id*100+t][2].w = (z_0-z_1)/0.01;  // z'
        Buffer[vertex_id*100+t][3].x = (z_0-2*z_1+z_2)/0.01/0.01;  // z''
        Buffer[vertex_id*100+t][3].y = (z_0-3*z_1+3*z_2-z_3)/0.01/0.01/0.01;  // z'''
        Buffer[vertex_id*100+t][3].z = (z_0-4*z_1+6*z_2-4*z_3+z_4)/0.01/0.01/0.01/0.01;  // z''''
    }

    // t = 3
    x_0 = Vertices[vertex_id*100+3-0].y;
    x_1 = Vertices[vertex_id*100+3-1].y;
    x_2 = Vertices[vertex_id*100+3-2].y;
    x_3 = Vertices[vertex_id*100+3-3].y;

    y_0 = Vertices[vertex_id*100+3-0].z;
    y_1 = Vertices[vertex_id*100+3-1].z;
    y_2 = Vertices[vertex_id*100+3-2].z;
    y_3 = Vertices[vertex_id*100+3-3].z;

    z_0 = Vertices[vertex_id*100+3-0].w;
    z_1 = Vertices[vertex_id*100+3-1].w;
    z_2 = Vertices[vertex_id*100+3-2].w;
    z_3 = Vertices[vertex_id*100+3-3].w;

    Buffer[vertex_id*100+3][0].x = x_0;  // x
    Buffer[vertex_id*100+3][0].y = (x_0-x_1)/0.01;  // x'
    Buffer[vertex_id*100+3][0].z = (x_0-2*x_1+x_2)/0.01/0.01;  // x''
    Buffer[vertex_id*100+3][0].w = (x_0-3*x_1+3*x_2-x_3)/0.01/0.01/0.01;  // x'''
    Buffer[vertex_id*100+3][1].x = Buffer[vertex_id*100+4][1].x/4*3;  // x''''

    Buffer[vertex_id*100+3][1].y = y_0;  // y
    Buffer[vertex_id*100+3][1].z = (y_0-y_1)/0.01;  // y'
    Buffer[vertex_id*100+3][1].w = (y_0-2*y_1+y_2)/0.01/0.01;  // y''
    Buffer[vertex_id*100+3][2].x = (y_0-3*y_1+3*y_2-y_3)/0.01/0.01/0.01;  // y'''
    Buffer[vertex_id*100+3][2].y = Buffer[vertex_id*100+4][2].y/4*3;  // y''''

    Buffer[vertex_id*100+3][2].z = z_0;  // z
    Buffer[vertex_id*100+3][2].w = (z_0-z_1)/0.01;  // z'
    Buffer[vertex_id*100+3][3].x = (z_0-2*z_1+z_2)/0.01/0.01;  // z''
    Buffer[vertex_id*100+3][3].y = (z_0-3*z_1+3*z_2-z_3)/0.01/0.01/0.01;  // z'''
    Buffer[vertex_id*100+3][3].z = Buffer[vertex_id*100+4][3].z/4*3;  // z''''

    // t = 2
    x_0 = Vertices[vertex_id*100+2-0].y;
    x_1 = Vertices[vertex_id*100+2-1].y;
    x_2 = Vertices[vertex_id*100+2-2].y;

    y_0 = Vertices[vertex_id*100+2-0].z;
    y_1 = Vertices[vertex_id*100+2-1].z;
    y_2 = Vertices[vertex_id*100+2-2].z;

    z_0 = Vertices[vertex_id*100+2-0].w;
    z_1 = Vertices[vertex_id*100+2-1].w;
    z_2 = Vertices[vertex_id*100+2-2].w;

    Buffer[vertex_id*100+2][0].x = x_0;  // x
    Buffer[vertex_id*100+2][0].y = (x_0-x_1)/0.01;  // x'
    Buffer[vertex_id*100+2][0].z = (x_0-2*x_1+x_2)/0.01/0.01;  // x''
    Buffer[vertex_id*100+2][0].w = Buffer[vertex_id*100+3][0].w/3*2;  // x'''
    Buffer[vertex_id*100+2][1].x = Buffer[vertex_id*100+4][1].x/4*2;  // x''''

    Buffer[vertex_id*100+2][1].y = y_0;  // y
    Buffer[vertex_id*100+2][1].z = (y_0-y_1)/0.01;  // y'
    Buffer[vertex_id*100+2][1].w = (y_0-2*y_1+y_2)/0.01/0.01;  // y''
    Buffer[vertex_id*100+2][2].x = Buffer[vertex_id*100+3][2].x/3*2;  // y'''
    Buffer[vertex_id*100+2][2].y = Buffer[vertex_id*100+4][2].y/4*2;  // y''''

    Buffer[vertex_id*100+2][2].z = z_0;  // z
    Buffer[vertex_id*100+2][2].w = (z_0-z_1)/0.01;  // z'
    Buffer[vertex_id*100+2][3].x = (z_0-2*z_1+z_2)/0.01/0.01;  // z''
    Buffer[vertex_id*100+2][3].y = Buffer[vertex_id*100+3][3].y/3*2;  // z'''
    Buffer[vertex_id*100+2][3].z = Buffer[vertex_id*100+4][3].z/4*2;  // z''''

    // t = 1
    x_0 = Vertices[vertex_id*100+1-0].y;
    x_1 = Vertices[vertex_id*100+1-1].y;

    y_0 = Vertices[vertex_id*100+1-0].z;
    y_1 = Vertices[vertex_id*100+1-1].z;

    z_0 = Vertices[vertex_id*100+1-0].w;
    z_1 = Vertices[vertex_id*100+1-1].w;

    Buffer[vertex_id*100+1][0].x = x_0;  // x
    Buffer[vertex_id*100+1][0].y = (x_0-x_1)/0.01;  // x'
    Buffer[vertex_id*100+1][0].z = Buffer[vertex_id*100+2][0].z/2;  // x''
    Buffer[vertex_id*100+1][0].w = Buffer[vertex_id*100+3][0].w/3*1;  // x'''
    Buffer[vertex_id*100+1][1].x = Buffer[vertex_id*100+4][1].x/4*1;  // x''''

    Buffer[vertex_id*100+1][1].y = y_0;  // y
    Buffer[vertex_id*100+1][1].z = (y_0-y_1)/0.01;  // y'
    Buffer[vertex_id*100+1][1].w = Buffer[vertex_id*100+2][1].w/2;  // y''
    Buffer[vertex_id*100+1][2].x = Buffer[vertex_id*100+3][2].x/3*1;  // y'''
    Buffer[vertex_id*100+1][2].y = Buffer[vertex_id*100+4][2].y/4*1;  // y''''

    Buffer[vertex_id*100+1][2].z = z_0;  // z
    Buffer[vertex_id*100+1][2].w = (z_0-z_1)/0.01;  // z'
    Buffer[vertex_id*100+1][3].x = Buffer[vertex_id*100+2][3].x/2;  // z''
    Buffer[vertex_id*100+1][3].y = Buffer[vertex_id*100+3][3].y/3*1;  // z'''
    Buffer[vertex_id*100+1][3].z = Buffer[vertex_id*100+4][3].z/4*1;  // z''''

    // t = 0
    x_0 = Vertices[vertex_id*100+0-0].y;

    y_0 = Vertices[vertex_id*100+0-0].z;

    z_0 = Vertices[vertex_id*100+0-0].w;

    Buffer[vertex_id*100+0][0].x = x_0;  // x
    Buffer[vertex_id*100+0][0].y = 0.0;  // x'
    Buffer[vertex_id*100+0][0].z = 0.0;  // x''
    Buffer[vertex_id*100+0][0].w = 0.0;  // x'''
    Buffer[vertex_id*100+0][1].x = 0.0;  // x''''

    Buffer[vertex_id*100+0][1].y = y_0;  // y
    Buffer[vertex_id*100+0][1].z = 0.0;  // y'
    Buffer[vertex_id*100+0][1].w = 0.0;  // y''
    Buffer[vertex_id*100+0][2].x = 0.0;  // y'''
    Buffer[vertex_id*100+0][2].y = 0.0;  // y''''

    Buffer[vertex_id*100+0][2].z = z_0;  // z
    Buffer[vertex_id*100+0][2].w = 0.0;  // z'
    Buffer[vertex_id*100+0][3].x = 0.0;  // z''
    Buffer[vertex_id*100+0][3].y = 0.0;  // z'''
    Buffer[vertex_id*100+0][3].z = 0.0;  // z''''


}

void main() {
    derivative(index);

}
