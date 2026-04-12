from PySide6.QtWidgets import QWidget
from tool.ui.controller import Controller

class SidePanel(QWidget):
    def __init__(self, controller: Controller, parent=None):
        super().__init__(parent)
        self.controller = controller
