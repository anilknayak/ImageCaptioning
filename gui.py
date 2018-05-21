#-------------------------------------------------------------------
# @author 
# @copyright (C) 2018, 
# @doc
#
# @end
# Created : 21. May 2018 3:28 PM
#-------------------------------------------------------------------

from PyQt5 import QtCore, QtGui, QtWidgets
from PyQt5.QtWidgets import QFileDialog
import pyqtgraph as pg
import text_images as caption_obj

class MainWindow(QtWidgets.QMainWindow):
    def __init__(self):
        super(MainWindow, self).__init__()
        screen = QtWidgets.QDesktopWidget().availableGeometry()
        self.screen_width = screen.width()
        self.screen_height = screen.height()
        self.setWindowTitle('Image Captioning')

        frame = QtWidgets.QFrame()
        layout = QtWidgets.QHBoxLayout()
        layout.setContentsMargins(0, 0, 0, 0)
        layout.setSpacing(0)
        frame.setLayout(layout)

        block_title = u'&Display Image'
        image_window = QtWidgets.QGroupBox(block_title)
        image_window_layout = QtWidgets.QVBoxLayout()

        self.display_image = pg.QtGui.QLabel()

        image_window_layout.addWidget(self.display_image)
        image_window.setLayout(image_window_layout)
        layout.addWidget(image_window)

        self.pushButton = QtGui.QPushButton("Select Image")
        self.pushButton.clicked.connect(self.on_pushButton_clicked)
        image_window_layout.addWidget(self.pushButton)

        self.pushButton_c = QtGui.QPushButton("Get Caption")
        self.pushButton_c.clicked.connect(self.on_pushButton_caption)
        image_window_layout.addWidget(self.pushButton_c)

        block_title = u'&Display Caption'
        caption_window = QtWidgets.QGroupBox(block_title)
        caption_window_layout = QtWidgets.QVBoxLayout()
        self.text = pg.QtGui.QLabel()
        self.text.setWordWrap(True)
        caption_window_layout.addWidget(self.text)
        caption_window.setFixedWidth(self.screen_width//4)
        caption_window.setLayout(caption_window_layout)
        layout.addWidget(caption_window)

        self.setGeometry(0, 0, self.screen_width, self.screen_height)
        self.setAutoFillBackground(True)
        self.setCentralWidget(frame)

        self.caption_model_obj = caption_obj.Sample()

    def openFileNameDialog(self):
        options = QFileDialog.Options()
        options |= QFileDialog.DontUseNativeDialog
        fileName, _ = QFileDialog.getOpenFileName(self, "Choose Image", "",
                                                  "Image Files (*.jpg)", options=options)

        self.text.setText('')
        self.filename = fileName
        self.display_image.setPixmap(QtGui.QPixmap(self.filename))

    def on_pushButton_clicked(self):
        self.openFileNameDialog()

    def on_pushButton_caption(self):
        caption = self.caption_model_obj.get_caption(self.filename)
        caption = caption.replace('startseq', '')
        caption = caption.replace('endseq', '')
        self.text.setText(caption)


if __name__ == '__main__':
    import sys
    app = QtWidgets.QApplication(sys.argv)
    mainWin = MainWindow()
    mainWin.show()
    sys.exit(app.exec_())