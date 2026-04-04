from PySide6.QtGui import QAction,QIcon
from PySide6.QtWidgets import QToolBar
from PySide6.QtWidgets import QFileDialog
from tool.ui.controller import Controller


class ToolBar(QToolBar):
    def __init__(self, controller: Controller, parent=None):
        super().__init__(parent)
        self.controller = controller

        # Run button
        run_action = QAction("Run", self)
        run_action.triggered.connect(self.on_run)
        self.addAction(run_action)

        # Clear button
        clear_action = QAction("Clear", self)
        clear_action.triggered.connect(self.on_clear)
        self.addAction(clear_action)

        # Import button
        import_action = QAction("Import", self)
        import_action.triggered.connect(self.on_import)
        self.addAction(import_action)

        # Export button
        export_action = QAction("Export", self)
        export_action.triggered.connect(self.on_export)
        self.addAction(export_action)


    def on_run(self):
        print("Run clicked")

    def on_clear(self):
        print("Clear clicked")

    def on_import(self):
        print("Import clicked")
        path, _ = QFileDialog.getOpenFileName(
            self,
            "Open image",
            "",
            "Images (*.png *.jpg *.jpeg *.bmp)"
        )
        print(path)
        self.controller.load_image(path)

    def on_export(self):
        print("Export clicked")
        path, _ = QFileDialog.getSaveFileName(
            self,
            "Save image",
            "",
            "PNG Image (*.png);;JPEG Image (*.jpg *.jpeg);;Bitmap Image (*.bmp)"
        )
        print(path)
        self.controller.save_image(path)