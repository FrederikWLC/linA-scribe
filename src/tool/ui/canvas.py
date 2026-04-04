from tool.ui.controller import Controller
from PySide6.QtWidgets import QLabel, QWidget
from PySide6.QtGui import QGuiApplication, QKeySequence, QPixmap, Qt

class Canvas(QWidget):

    def __init__(self, controller: Controller | None = None, parent=None) -> None:
        super().__init__(parent) # registers parent within Qt's parent-child system for memory management
        self.controller = controller
        self._original_pixmap = None
        self.setAcceptDrops(True)
        self.setMinimumSize(400, 300)
        self.setAutoFillBackground(True)
        self.setAttribute(Qt.WA_StyledBackground, True)
        self.set_default_background()
        self.setFocusPolicy(Qt.StrongFocus)
        self.acceptDrops()
        # overlay text (hidden by default)
        self.overlay = QLabel("Drop a raw image here...\nor paste a screenshot (CTRL/CMD+V)", self)
        self.overlay.setAlignment(Qt.AlignCenter)
        self.overlay.setStyleSheet("""
            color: darkgray;
            font-size: 18px;
            background-color: rgba(0, 0, 0, 0);
        """)
        self.image_label = QLabel(self)
        self.image_label.setAlignment(Qt.AlignCenter)
        self.image_label.hide()

    def set_default_background(self):
        self.setStyleSheet("background-color: #232429;")

    def set_drag_background(self):
        self.setStyleSheet("background-color: lightblue;")

    def resizeEvent(self, event):
        # keep label covering entire canvas
        self.overlay.setGeometry(self.rect())
        self.image_label.setGeometry(self.rect())
        self._update_image_display()
        super().resizeEvent(event)

    # --- call this once when you want to show it ---
    def show_hint(self):
        self.overlay.show()

    # --- call this once when you want to remove it ---
    def clear_hint(self):
        self.overlay.hide()

    def dragEnterEvent(self, event):
        print("Drag entered")
        if event.mimeData().hasUrls():
            self.set_drag_background()
            event.acceptProposedAction()

    def dragLeaveEvent(self, event):
        print("Drag left")
        self.set_default_background()

    def dropEvent(self, event):
        self.set_default_background()
        file_path = event.mimeData().urls()[0].toLocalFile()
        if file_path.lower().endswith((".png", ".jpg", ".jpeg", ".bmp")):
            self.load_image(file_path)
    
    def keyPressEvent(self, event):
        print("Key pressed:", event.key())
        if event.matches(QKeySequence.StandardKey.Paste):
            print("Paste detected")
            self.handle_paste()
        else:
            super().keyPressEvent(event)

    def handle_paste(self):
        clipboard = QGuiApplication.clipboard()
        mime = clipboard.mimeData()

        # when several files are pasted
        if mime.hasUrls():
            urls = mime.urls()
            for url in urls:
                path = url.toLocalFile()
                if path:
                    print("Pasted file:", path)
                    self.load_image(path)

        # when one file is pasted
        elif mime.hasImage():
            image = clipboard.image()
            print("Pasted image")
            self.set_image(image)

        else:
            print("Clipboard does not contain usable data")

    def load_image(self, path):
        print("Load image:", path)
        pixmap = QPixmap(path)
        if pixmap.isNull():
            print("Failed to load image")
            return
        self._original_pixmap = pixmap
        self._update_image_display()

    def set_image(self, image):
        print("Set image from clipboard")
        pixmap = QPixmap.fromImage(image)
        if pixmap.isNull():
            print("Failed to convert clipboard image")
            return
        self._original_pixmap = pixmap
        self._update_image_display()

    def _update_image_display(self):
        if self._original_pixmap is None:
            return
        scaled = self._original_pixmap.scaled(
            self.size(),
            Qt.KeepAspectRatio,
            Qt.SmoothTransformation,
        )
        self.image_label.setPixmap(scaled)
        self.image_label.show()
        self.overlay.hide()

