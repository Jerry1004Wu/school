import os
os.environ["KMP_DUPLICATE_LIB_OK"]="TRUE"
from ultralytics import YOLO
from ultralytics.data.dataset import YOLODataset
import ultralytics.data.build as build
import numpy as np

if __name__== "__main__":
    model = YOLO(r'C:\Users\USER\Desktop\school\best.pt')

    #results = model.val(data=r'C:\Users\USER\Desktop\school\data.yaml',save_json=True)
    results = model.predict(source=r'C:\Users\USER\Desktop\school\Fisheye8K_all\test\images', save=True)
