import pyrr
import numpy as np


class Camera:
    def __init__(self):
        self.position = pyrr.Vector4([0.0, 0.0, 30.0, 1.0])
        self.front = pyrr.Vector4([0.0, 0.0, -1.0, 1.0])
        self.up = pyrr.Vector4([0.0, 1.0, 0.0, 1.0])

        self.projection = pyrr.matrix44.create_perspective_projection_matrix(45, 1.0, 0.001, 1000)
        self.view = pyrr.matrix44.create_look_at(self.position.xyz, self.front.xyz, self.up.xyz)
        self.translate = pyrr.matrix44.create_identity()
        self.rotate = pyrr.matrix44.create_identity()
        self.scale = pyrr.matrix44.create_identity()

        self.mouse_left = False
        self.mouse_middle = False
        self.mouse_pos = pyrr.Vector3([0.0, 0.0, 0.0])

    def __call__(self, delta=pyrr.Vector3([0.0, 0.0, 0.0]), flag=None):
        # left: rotate/spin
        if flag == "left" and self.mouse_left:
            # spin
            if abs(self.mouse_pos.x) >= 800 or abs(self.mouse_pos.y) >= 800:
                rotation_matrix = pyrr.matrix44.create_from_axis_rotation(-np.sign(
                    (pyrr.vector3.cross(pyrr.Vector3([self.mouse_pos.x + delta.x, self.mouse_pos.y + delta.y, 0.0]),
                                        pyrr.Vector3([self.mouse_pos.x, self.mouse_pos.y, 0.0])))[2]) * self.front.xyz,
                                                                          delta.z)
                self.up = rotation_matrix @ self.up
            # rotate
            else:
                rotation_matrix = pyrr.matrix44.create_from_axis_rotation(self.up.xyz, -delta.x / 100) @ \
                                  pyrr.matrix44.create_from_axis_rotation(
                                      pyrr.vector3.cross(self.front.xyz, self.up.xyz),
                                      delta.y / 100)
                self.position = rotation_matrix @ self.position
                self.front = rotation_matrix @ self.front
                self.up = rotation_matrix @ self.up
        # middle: move
        if flag == "middle" and self.mouse_middle:
            right = pyrr.vector3.normalize(pyrr.vector3.cross(self.front.xyz, self.up.xyz))
            self.position += pyrr.Vector4([*right, 0.0]) * delta.x/10
            self.position += pyrr.Vector4([*self.up.xyz, 0.0]) * delta.y/10


        # TODO: apply translation


        return pyrr.matrix44.create_look_at(self.position.xyz, (self.position+self.front).xyz, self.up.xyz)

    def x_view(self):
        self.position = pyrr.Vector4([-np.linalg.norm(self.position.xyz), 0.0, 0.0, 1.0])
        self.front = pyrr.Vector4([1.0, 0.0, 0.0, 1.0])
        self.up = pyrr.Vector4([0.0, 1.0, 0.0, 1.0])
        return pyrr.matrix44.create_look_at(self.position.xyz, (self.position + self.front).xyz, self.up.xyz)

    def y_view(self):
        self.position = pyrr.Vector4([0.0, -np.linalg.norm(self.position.xyz), 0.0, 1.0])
        self.front = pyrr.Vector4([0.0, 1.0, 0.0, 1.0])
        self.up = pyrr.Vector4([0.0, 0.0, 1.0, 1.0])
        return pyrr.matrix44.create_look_at(self.position.xyz, (self.position + self.front).xyz, self.up.xyz)

    def z_view(self):
        self.position = pyrr.Vector4([0.0, 0.0, -np.linalg.norm(self.position.xyz), 1.0])
        self.front = pyrr.Vector4([0.0, 0.0, 1.0, 1.0])
        self.up = pyrr.Vector4([0.0, 1.0, 0.0, 1.0])
        return pyrr.matrix44.create_look_at(self.position.xyz, (self.position + self.front).xyz, self.up.xyz)

    def set_view(self, import_str: str):
        try:
            self.indice = [item.split(",") for item in import_str.split("\n")]
            self.position = pyrr.Vector4([*[float(item) for item in self.indice[0]], 1.0])
            self.front = pyrr.Vector4([*[float(item) for item in self.indice[1]], 1.0])
            self.up = pyrr.Vector4([*[float(item) for item in self.indice[2]], 1.0])
        except AttributeError:
            self.position = pyrr.Vector4([0.0, 0.0, 30.0, 1.0])
            self.front = pyrr.Vector4([0.0, 0.0, -1.0, 1.0])
            self.up = pyrr.Vector4([0.0, 1.0, 0.0, 1.0])
        return pyrr.matrix44.create_look_at(self.position.xyz, (self.position + self.front).xyz, self.up.xyz), \
               "{:.2f},{:.2f},{:.2f}\n{:.2f},{:.2f},{:.2f}\n{:.2f},{:.2f},{:.2f}".format(*self.position.xyz, *self.front.xyz, *self.up.xyz)
