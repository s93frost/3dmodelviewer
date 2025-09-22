import sys
from PyQt5.QtWidgets import QApplication
from minimal_viewer import ViewerApp

if __name__ == "__main__":
    app = QApplication(sys.argv)
    window = ViewerApp()
    window.show()
    sys.exit(app.exec_())
