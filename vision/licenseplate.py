import numpy as np
import imageio.v3 as iio
from licenseplates import model
from PIL import Image, ImageDraw
import cv2
from ultralytics import YOLO
from paddleocr import PaddleOCR
import yolov5
from pathlib import Path
import filetype 

class LicensePlateFinder:
    def __init__(self) -> None:
        self.ocr_model = PaddleOCR(use_angle_cls=True, lang='en', show_log=False)
        self.objects_model = YOLO('yolov8n.pt')
        self.licenseplate_model = yolov5.load('keremberke/yolov5m-license-plate')

    def __call__(self, source:str, destination:str) -> None:
        source = Path(source)
        destination = Path(destination)

        assert source.is_file(), f'Source path {source} is not a file!'
        assert destination.parents[0].is_dir() and destination.suffix!='', f'Destination path {destination} invalid!'
        
        file_kind = filetype.guess(source).mime
        
        if 'video' in file_kind:
            self.video_inference(source, destination)
        elif 'image' in file_kind:
            self.image_inference(source, destination)
        else:
            print(f'File {source} not a video or an image')
            raise NotImplementedError


    def video_inference(self, source:Path, destination:Path) -> None:
        pass

    def image_inference(self, source:Path, destination:Path) -> None:
        pass

    def frame_inference(self, frame:np.array) -> np.array:
        pass


if __name__ == '__main__':
    LicensePlateFinder()('/home/marat/Videos/1.mp4', '/home/marat/Videos/1_annotated.mp4')