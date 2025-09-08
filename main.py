from PyQt5.QtWidgets import QApplication
from viewer import ViewerApp
import sys

if __name__ == "__main__":
    app = QApplication(sys.argv)
    window = ViewerApp()
    window.show()
    sys.exit(app.exec_())
