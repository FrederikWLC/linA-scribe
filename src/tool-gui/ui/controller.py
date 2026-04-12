class Controller:
    def __init__(self):
        self.model = None

    def load_image(self, path):
        print(f"Loading image from: {path}")
        # Here you would add the actual image loading logic, e.g. using OpenCV or PIL
        # For example:
        # self.model.image = cv2.imread(path, cv2.IMREAD_GRAYSCALE)
        # self.model.update_canvas()