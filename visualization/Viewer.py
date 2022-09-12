import datetime
import sys
from PyQt5.QtCore import *
from PyQt5.QtWidgets import *
from PyQt5.QtGui import *
from mainwindow import Ui_MainWindow
import numpy as np
import time
from OpenGL.GL import *
from OpenGL.GL.shaders import compileProgram, compileShader
import pyrr
from threading import Thread
from loader import Classifier


def console_out(target, text):
    target.moveCursor(QTextCursor.End)
    target.insertHtml(f'<span style="color:#000000">{text}</span>')
    target.insertPlainText("\n")
    return sys.stdout.write("stdout:" + text)


def console_err(target, text):
    target.moveCursor(QTextCursor.End)
    target.insertHtml(f'<span style="color:#FF6600">{text}</span>')
    target.insertPlainText("\n")
    return sys.stderr.write("stderr:" + text)


class Viewer(QMainWindow, Ui_MainWindow):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        main_window = self.setupUi(self)
        self.openGLWidget.setMinimumSize(960, 540)
        # self.openGLWidget.setMaximumSize(960, 540)
        self.glw = MyGLWidget(self.openGLWidget)
        self.glw.resize(960, 540)
        self.pushButton_switch.clicked.connect(lambda event: self.glw.__setattr__("freeze", not self.glw.freeze))
        self.glw.makeCurrent()
        self.openGLWidget.resized.connect(lambda: self.glw.resize(self.openGLWidget.size()))
        # glw colormap index
        self.column = 6

        def excel_column_number(name):
            """Excel-style column name to number, e.g., A = 1, Z = 26, AA = 27, AAA = 703."""
            n = 0
            for c in name:
                n = n * 26 + 1 + ord(c.upper()) - ord('A')
            if n - 1 == -1:
                return 6
            else:
                return n - 1

        self.lineEdit_colormap_column.textChanged.connect(
            lambda *args: self.__setattr__("column", excel_column_number(self.lineEdit_colormap_column.text())))
        self.lineEdit_colormap_column.textChanged.connect(lambda *args: print(self.column))
        # self.setAcceptDrops(True)
        self.glw.setAcceptDrops(True)
        self.listWidget_object_list.setAcceptDrops(True)
        # self.listWidget_object_list.setDragDropMode(QAbstractItemView.DragDrop)
        self.listWidget_object_list.addItems(["Axis"])
        self.listWidget_object_list.item(0).setBackground(QColor(220, 120, 20))
        self.listWidget_object_list.dropEvent = self.dropEvent
        self.listWidget_object_list.dragEnterEvent = self.dragEnterEvent
        self.listWidget_object_list.dragMoveEvent = self.dragMoveEvent
        self.listWidget_object_list.setDragDropMode(QAbstractItemView.DragDrop)
        # list object edit
        self.listWidget_object_list.clicked.connect(self.list_widget_clicked_clb)
        # self.listWidget_object_list.setEditTriggers(QAbstractItemView.DoubleClicked)
        # self.listWidget_object_list.doubleClicked.connect(lambda item:
        # self.__setattr__("current_tag", self.listWidget_object_list.item(item.row()).text()))
        # self.current_tag = None
        # console
        self.textEdit_console_log.setText(f"Version: 0.0.0.3\nStart time: {datetime.datetime.now()}\n")

    def list_widget_clicked_clb(self, item):
        # enable text editable
        # self.listWidget_object_list.item(item.row()).setFlags(
        #     self.listWidget_object_list.item(item.row()).flags() | Qt.ItemIsEditable)
        tag = self.listWidget_object_list.item(item.row()).text()
        sys.stdout.write(f"Current selection: {tag}.\n")
        self.glw.vao[tag][2] = not self.glw.vao[tag][2]
        if self.glw.vao[tag][2]:
            self.listWidget_object_list.item(item.row()).setBackground(QColor(220, 120, 20))
        else:
            self.listWidget_object_list.item(item.row()).setBackground(QColor(255, 255, 255))

    def dragEnterEvent(self, event):
        event.acceptProposedAction()

    def dragMoveEvent(self, event):
        event.acceptProposedAction()

    def dropEvent(self, event):
        # avoid list index error if text is ""
        if event.mimeData().text() == "":
            return
        # print(event.mimeData().text()) #"file:///C:/Users/情報基礎数学専攻/Documents/k01_blender_stl_100.stl"
        import re
        print(event.mimeData().text())
        console_out(self.textEdit_console_log, event.mimeData().text() + "\n")
        print(re.findall(re.compile(r"file:///(.*)", re.S), event.mimeData().text()))
        print(re.findall(re.compile(r"file:///.+/(.*)", re.S), event.mimeData().text()))
        # TODO: run Classifier() in another thread.
        # return buffered data or None(unidentified file-format).
        buffer = Classifier(self.column)(re.findall(re.compile(r"file:///(.*)", re.S), event.mimeData().text())[0])
        # validation check
        if buffer is not None:
            # "file:///C:/.../(selected-part)"
            tag = str(event.mimeData().text()).split("/")[-1]
            if type(buffer) == tuple:
                buffer, pointer = buffer
                # move to (0, 0, 0)
                buffer[:, 0] -= np.average(buffer[:, 0])
                buffer[:, 1] -= np.average(buffer[:, 1])
                buffer[:, 2] -= np.average(buffer[:, 2])
            else:  # type(buffer) == np.ndarray
                buffer = buffer
                pointer = [[0, buffer.shape[0]]]
                # move to (0, 0, 0)
                buffer[:, 0] -= np.average(buffer[:, 0])
                buffer[:, 1] -= np.average(buffer[:, 1])
                buffer[:, 2] -= np.average(buffer[:, 2])
            try:
                assert tag not in [self.listWidget_object_list.item(x).text() for x in
                                   range(self.listWidget_object_list.count())]
                self.listWidget_object_list.addItem(tag)
                self.listWidget_object_list.item(self.listWidget_object_list.count() - 1).setBackground(
                    QColor(220, 120, 20))
                self.glw.load_buffer(tag, buffer, pointer)
            except AssertionError:
                tail = 1
                _ = f"{tag}_{tail}"
                while _ in [self.listWidget_object_list.item(x).text() for x in
                            range(self.listWidget_object_list.count())]:
                    tail += 1
                    _ = f"{tag}_{tail}"
                self.listWidget_object_list.addItem(_)
                self.listWidget_object_list.item(self.listWidget_object_list.count() - 1).setBackground(
                    QColor(220, 120, 20))
                self.glw.load_buffer(_, buffer, pointer)
            except IndexError:
                pass
        else:
            sys.stderr.write("Empty buffer! Ignored!\n")
            console_err(self.textEdit_console_log, "Empty buffer! Ignored!\n")
            pass


class MyGLWidget(QOpenGLWidget):
    def __init__(self, *args):
        super().__init__(*args)
        self.bg = [1.0, 1.0, 1.0, 1.0]
        # refresh rate
        self.timer = QTimer(self)
        self.timer.setInterval(16)
        self.timer.timeout.connect(self.update)
        self.timer.start()
        self.start_time = time.time()
        self.setMouseTracking(True)
        # self.grabKeyboard()
        self.freeze = False
        # VAO and Draw-list
        self.vao = {}
        self.display_list = []

    def load_buffer(self, tag: str, buffer: np.ndarray, pointer: list):
        self.makeCurrent()
        vao = glGenVertexArrays(1)
        glBindVertexArray(vao)
        vbo = glGenBuffers(1)
        glBindBuffer(GL_ARRAY_BUFFER, vbo)
        glBufferData(GL_ARRAY_BUFFER, buffer.nbytes, buffer, GL_STATIC_DRAW)
        # vertex shape: [x, y, z, r, g, b]
        if buffer.shape[1] == 6:
            glEnableVertexAttribArray(0)
            glVertexAttribPointer(0, 3, GL_FLOAT, GL_FALSE, 24, ctypes.c_void_p(0))
            glEnableVertexAttribArray(1)
            glVertexAttribPointer(1, 3, GL_FLOAT, GL_FALSE, 24, ctypes.c_void_p(12))
        # TODO: this will be removed or improved.
        elif buffer.shape[1] == 3:
            glEnableVertexAttribArray(0)
            glVertexAttribPointer(0, 3, GL_FLOAT, GL_FALSE, 12, ctypes.c_void_p(0))
            glEnableVertexAttribArray(1)
            glVertexAttribPointer(1, 3, GL_FLOAT, GL_FALSE, 12, ctypes.c_void_p(0))
        elif buffer.shape[1] == 8:
            glEnableVertexAttribArray(0)
            glVertexAttribPointer(0, 4, GL_FLOAT, GL_FALSE, 32, ctypes.c_void_p(0))
            glEnableVertexAttribArray(1)
            glVertexAttribPointer(1, 4, GL_FLOAT, GL_FALSE, 32, ctypes.c_void_p(12))
        self.vao[tag] = [vao, pointer, True]

    def load(self, file: str, tag: str):
        self.makeCurrent()
        vertices = np.load(file)
        print(vertices.shape)
        self.vao[tag] = glGenVertexArrays(1)
        glBindVertexArray(self.vao[tag])
        vbo = glGenBuffers(1)
        glBindBuffer(GL_ARRAY_BUFFER, vbo)
        glBufferData(GL_ARRAY_BUFFER, vertices.nbytes, vertices, GL_STATIC_DRAW)
        
        if vertices.shape[1] == 8:
            glEnableVertexAttribArray(0)
            glVertexAttribPointer(0, 4, GL_FLOAT, GL_FALSE, 32, ctypes.c_void_p(0))
            glEnableVertexAttribArray(1)
            glVertexAttribPointer(1, 4, GL_FLOAT, GL_FALSE, 32, ctypes.c_void_p(12))
        else:
            glEnableVertexAttribArray(0)
            glVertexAttribPointer(0, 3, GL_FLOAT, GL_FALSE, 12, ctypes.c_void_p(0))
            glEnableVertexAttribArray(1)
            glVertexAttribPointer(1, 3, GL_FLOAT, GL_FALSE, 12, ctypes.c_void_p(0))
        self.display_list.append(tag)

    @staticmethod
    def load_series(directory: str, file: list):
        vao = {i: glGenVertexArrays(1) for i in range(20)}
        for i in vao.keys():
            buffer = np.load(directory + "/" + file[i])
            print(i)
            glBindVertexArray(vao[i])
            vertices = buffer
            vbo = glGenBuffers(1)
            glBindBuffer(GL_ARRAY_BUFFER, vbo)
            glBufferData(GL_ARRAY_BUFFER, vertices.nbytes, vertices, GL_STATIC_DRAW)
            glEnableVertexAttribArray(0)
            glVertexAttribPointer(0, 3, GL_FLOAT, GL_FALSE, 12, ctypes.c_void_p(0))
            glEnableVertexAttribArray(1)
            glVertexAttribPointer(1, 3, GL_FLOAT, GL_FALSE, 12, ctypes.c_void_p(0))
        return vao

    def initializeGL(self):
        import shader_src
        self.shader = compileProgram(compileShader(shader_src.vertex_src, GL_VERTEX_SHADER),
                                     compileShader(shader_src.fragment_src, GL_FRAGMENT_SHADER))
        glUseProgram(self.shader)
        glClearColor(*self.bg)
        glEnable(GL_DEPTH_TEST)

        self.projection = pyrr.matrix44.create_perspective_projection(45, 960 / 540, 0.001, 1000)
        self.view = pyrr.matrix44.create_look_at(pyrr.Vector3([0.0, 0.0, 30.0]), pyrr.Vector3([0.0, 0.0, -1.0]),
                                                 pyrr.Vector3([0, 1, 0]))
        model = pyrr.matrix44.create_from_z_rotation(0.0)

        self.proj_loc = glGetUniformLocation(self.shader, "projection")
        self.view_loc = glGetUniformLocation(self.shader, "view")
        model_loc = glGetUniformLocation(self.shader, "model")
        self.color_loc = glGetUniformLocation(self.shader, "color")

        glUniformMatrix4fv(self.proj_loc, 1, GL_FALSE, self.projection)
        glUniformMatrix4fv(self.view_loc, 1, GL_FALSE, self.view)
        glUniformMatrix4fv(model_loc, 1, GL_FALSE, model)

        import camera
        self.camera = camera.Camera()

        glPointSize(4)

        self.current_frame = 0
        self.color_flag = 0
        self.color_pos = glGetUniformLocation(self.shader, "color_flag")
        # load vao
        import Coordinates
        axis = Coordinates.Coord()
        self.vao["Axis"] = [axis.vao, [[0, axis.indices.nbytes // 4]], True]

        self.color_loc = glGetUniformLocation(self.shader, "color")

        vertex_src = """"""
        fragment_src = """"""
        with open(r"./shaders/vertex_shader.shader", "r") as f:
            for row in f:
                vertex_src += row
            f.close()
        with open(r"./shaders/fragment_shader.shader", "r") as f:
            for row in f:
                fragment_src += row
            f.close()
        self.ai_shader = compileProgram(compileShader(vertex_src, GL_VERTEX_SHADER),
                                        compileShader(fragment_src, GL_FRAGMENT_SHADER))
        glUseProgram(self.ai_shader)
        self.ai_proj_loc = glGetUniformLocation(self.ai_shader, "projection")
        self.ai_view_loc = glGetUniformLocation(self.ai_shader, "view")
        ai_model_loc = glGetUniformLocation(self.ai_shader, "model")
        glUniformMatrix4fv(self.ai_proj_loc, 1, GL_FALSE, self.projection)
        glUniformMatrix4fv(self.ai_view_loc, 1, GL_FALSE, self.view)
        glUniformMatrix4fv(ai_model_loc, 1, GL_FALSE, model)
        glUseProgram(self.shader)

    def resizeGL(self, w, h):
        print(w, h)
        glViewport(0, 0, w, h)
        self.projection = pyrr.matrix44.create_perspective_projection(45, w / h, 0.001, 1000)
        glUniformMatrix4fv(self.proj_loc, 1, GL_FALSE, self.projection)

    def paintGL(self):
        glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT)
        glUniformMatrix4fv(self.view_loc, 1, GL_FALSE, self.view)
        for key in self.vao.keys():
            vao, pointer, draw = self.vao[key]
            if draw and key != "Axis" and key[:4] != "ai__":
                glUniform1i(self.color_pos, 1)
                glBindVertexArray(vao)
                # frame by frame
                glDrawArrays(GL_POINTS,
                             pointer[self.current_frame % len(pointer)][0],
                             pointer[self.current_frame % len(pointer)][1] - pointer[self.current_frame % len(pointer)][
                                 0])
            elif draw and key == "Axis":
                glUniform1i(self.color_pos, 0)
                glBindVertexArray(vao)
                glDrawElements(GL_TRIANGLES, pointer[0][1], GL_UNSIGNED_INT, None)
            elif draw and key[:4] == "ai__":
                glUseProgram(self.ai_shader)
                glUniformMatrix4fv(self.ai_proj_loc, 1, GL_FALSE, self.projection)
                glUniformMatrix4fv(self.ai_view_loc, 1, GL_FALSE, self.view)
                glBindVertexArray(vao)
                # frame by frame
                glDrawArrays(GL_POINTS,
                             pointer[self.current_frame % len(pointer)][0],
                             pointer[self.current_frame % len(pointer)][1] - pointer[self.current_frame % len(pointer)][
                                 0])
                glUseProgram(self.shader)

        if not self.freeze:
            self.current_frame += 1
        self.current_frame %= 900

    def mouseMoveEvent(self, a0: QMouseEvent):
        x_pos = a0.pos().x()
        y_pos = a0.pos().y()
        if abs(x_pos) < 400 and abs(y_pos) < 400:
            self.grabKeyboard()
        else:
            self.releaseKeyboard()
        x_pos -= self.size().width() / 2
        y_pos *= -1
        y_pos += self.size().height() / 2
        if self.camera.mouse_left:
            dx = self.camera.mouse_pos.x - x_pos
            dy = self.camera.mouse_pos.y - y_pos
            self.camera.mouse_pos = pyrr.Vector3([x_pos, y_pos, 0.0])
            delta = pyrr.Vector3([dx, dy, np.sqrt(dx ** 2 + dy ** 2) / 100])
            self.view = self.camera(delta, flag="left")
        elif self.camera.mouse_middle:
            dx = self.camera.mouse_pos.x - x_pos
            dy = self.camera.mouse_pos.y - y_pos
            self.camera.mouse_pos = pyrr.Vector3([x_pos, y_pos, 0.0])
            delta = pyrr.Vector3([dx, dy, 0])
            self.view = self.camera(delta / 5, flag="middle")
        else:
            self.camera.mouse_pos = pyrr.Vector3([x_pos, y_pos, 0.0])

    def mousePressEvent(self, a0: QMouseEvent):
        if a0.button() == 1:  # 1==left, 4=middle
            self.camera.mouse_left = True
        if a0.button() == 4:  # 1==left, 4=middle
            self.camera.mouse_middle = True

    def mouseReleaseEvent(self, a0: QMouseEvent):
        if a0.button() == 1:  # 1==left, 4=middle
            self.camera.mouse_left = False
        if a0.button() == 4:  # 1==left, 4=middle
            self.camera.mouse_middle = False

    def wheelEvent(self, a0: QWheelEvent) -> None:
        y_offset = a0.angleDelta().y() / 120

        def f():
            if sum([abs(item) for item in self.camera.position.xyz]) <= 1.01:
                if y_offset >= 0:
                    return
            for i in range(50):
                self.camera.position += self.camera.front * y_offset * 0.02
                self.camera.position = pyrr.Vector4([*self.camera.position.xyz, 1.0])
                self.view = self.camera(flag="wheel")
                time.sleep(0.005)
                if abs(sum([*self.camera.position.xyz])) <= 1.01:
                    if y_offset >= 0:
                        return

        t = Thread(target=f)
        t.start()

    def keyPressEvent(self, a0: QKeyEvent):
        print(self.size().height())
        print(a0.key(), a0.text())


if __name__ == '__main__':
    app = QApplication(sys.argv)
    window = MainWindow()
    window.show()
    app.exec()
