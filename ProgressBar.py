from PyQt5 import QtCore, QtGui, QtWidgets
from PyQt5.QtCore import Qt
from threading import Thread
import time

class _Bar(QtWidgets.QWidget):

    clickedValue = QtCore.pyqtSignal(int)

    def __init__(self, steps, *args, **kwargs):
        super().__init__(*args, **kwargs)

        self.setSizePolicy(
            QtWidgets.QSizePolicy.MinimumExpanding,
            QtWidgets.QSizePolicy.MinimumExpanding
        )

        if isinstance(steps, list):
            # list of colors.
            self.n_steps = len(steps)
            self.steps = steps

        elif isinstance(steps, int):
            # int number of bars, defaults to red.
            self.n_steps = steps
            self.steps = [(152+i/steps*(212-152), 153+i/steps*(218-153), 236-i/steps*40) for i in range(steps)]

        else:
            raise TypeError('steps must be a list or int')

        self._bar_solid_percent = 0.8
        self._background_color = QtGui.QColor(255, 255, 255, 0)
        self._padding = 4.0  # n-pixel gap around edge.

    def paintEvent(self, e):
        painter = QtGui.QPainter(self)

        brush = QtGui.QBrush()
        brush.setColor(self._background_color)
        brush.setStyle(Qt.SolidPattern)
        rect = QtCore.QRect(0, 0, painter.device().width(), painter.device().height())
        painter.fillRect(rect, brush)

        # Get current state.
        parent = self.parent()
        vmin, vmax = parent.minimum(), parent.maximum()
        value = parent.value()

        # Define our canvas.
        d_height = painter.device().height() - (self._padding * 2)
        d_width = painter.device().width() - (self._padding * 2)

        # Draw the bars.
        step_size = d_height / self.n_steps
        bar_height = step_size * self._bar_solid_percent
        bar_spacer = step_size * (1 - self._bar_solid_percent) / 2

        # Calculate the y-stop position, from the value in range.
        pc = (value - vmin) / (vmax - vmin)
        n_steps_to_draw = int(pc * self.n_steps)

        for n in range(n_steps_to_draw):
            brush.setColor(QtGui.QColor(*self.steps[n]))
            rect = QtCore.QRect(
                self._padding + n*d_width/len(self.steps),
                self._padding,
                d_width/len(self.steps)-2,
                d_height
            )
            painter.fillRect(rect, brush)

        painter.end()

    def sizeHint(self):
        return QtCore.QSize(20, 120)

    def _trigger_refresh(self):
        self.update()

    def _calculate_clicked_value(self, e):
        parent = self.parent()
        vmin, vmax = parent.minimum(), parent.maximum()

        value = vmin + e.x()//((self.size().width()) / (vmax - vmin+1))
        self.clickedValue.emit(value)

    """def mouseMoveEvent(self, e):
        self._calculate_clicked_value(e)

    def mousePressEvent(self, e):
        self._calculate_clicked_value(e)"""


class PowerBar(QtWidgets.QWidget):
    """
    Custom Qt Widget to show a power bar and dial.
    Demonstrating compound and custom-drawn widget.

    Left-clicking the button shows the color-chooser, while
    right-clicking resets the color to None (no-color).
    """

    colorChanged = QtCore.pyqtSignal()
    valueChanged = QtCore.pyqtSignal(object)

    def __init__(self, steps=50, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.__value = 20
        layout = QtWidgets.QVBoxLayout()
        self._bar = _Bar(steps)
        layout.addWidget(self._bar)

        # Create the QDial widget and set up defaults.
        # - we provide accessors on this class to override.
        self._dial = QtWidgets.QDial()
        # self._dial.setNotchesVisible(True)
        # self._dial.setWrapping(False)
        # self._dial.valueChanged.connect(self._bar._trigger_refresh)
        def setter(value):
            self.set_value = value

        # self._dial.valueChanged.connect(setter)

        # self.valueChanged.connect(lambda event: print(event))
        self.valueChanged.connect(self._bar._trigger_refresh)
        self.valueChanged.connect(self._dial.setValue)
        self._bar.clickedValue.connect(lambda event: print(event))
        self._bar.clickedValue.connect(setter)




        # Take feedback from click events on the meter.
        self._bar.clickedValue.connect(self._dial.setValue)

        # layout.addWidget(self._dial)
        self.setLayout(layout)

    @property
    def set_value(self):
        return self.__value

    @set_value.setter
    def set_value(self, value):
        self.__value = value
        self.valueChanged.emit(value)


    def __getattr__(self, name):
        if name in self.__dict__:
            return self[name]

        return getattr(self._dial, name)

    def setColor(self, color):
        self._bar.steps = [color] * self._bar.n_steps
        self._bar.update()

    def setColors(self, colors):
        self._bar.n_steps = len(colors)
        self._bar.steps = colors
        self._bar.update()

    def setBarPadding(self, i):
        self._bar._padding = int(i)
        self._bar.update()

    def setBarSolidPercent(self, f):
        self._bar._bar_solid_percent = float(f)
        self._bar.update()

    def setBackgroundColor(self, color):
        self._bar._background_color = QtGui.QColor(color)
        self._bar.update()


if __name__ == "__main__":
    app = QtWidgets.QApplication([])
    progress_bar = PowerBar()
    progress_bar.show()
    progress_bar.set_value = 0
    def set_it():
        t = 0
        while True:
            progress_bar.set_value = t
            t += 1
            time.sleep(0.2)
            t = t%100
    t1 = Thread(target=set_it)
    t1.setDaemon(True)
    t1.start()
    app.exec_()