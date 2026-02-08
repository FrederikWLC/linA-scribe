from config import config
from app import AppController

app = AppController(config.DATA_DIR)
for i,_ in enumerate(app.list_images()):
	results = app.process(i)
	print(results)