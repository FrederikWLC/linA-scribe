from PySide6.QtCore import Qt
from PySide6.QtWidgets import QDockWidget, QMainWindow
from tool.ui.canvas import Canvas
from tool.ui.controller import Controller
from tool.ui.side_panel import SidePanel
from tool.ui.toolbar import ToolBar

class MainWindow(QMainWindow):
    def __init__(self,controller: Controller) -> None:
        super().__init__()
        self.controller = controller
        self.setWindowTitle("Scribe: a segmentation tool")
        self.resize(900, 600)
        
        self.canvas = Canvas(controller,parent=self) # breaks DIP but tradition in Qt
        self.setCentralWidget(self.canvas)
        print(self.canvas)
        print(type(self.canvas))
        print(self.canvas.acceptDrops())

        self.tool_bar = ToolBar(controller, parent=self) # breaks DIP but tradition in Qt
        self.addToolBar(Qt.TopToolBarArea, self.tool_bar)
        # self.tool_bar.setMovable(False) # prevent user from undocking the toolbar

        self.side_panel = SidePanel(controller, parent=self) # breaks DIP but tradition in Qt
        dock = QDockWidget("Tools")
        # dock.setFeatures(QDockWidget.NoDockWidgetFeatures) # prevent user from undocking the side panel
        dock.setWidget(self.side_panel)
        self.addDockWidget(Qt.LeftDockWidgetArea, dock)