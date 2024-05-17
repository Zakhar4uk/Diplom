import numpy as np
import imageio.v3 as iio
from PIL import Image, ImageDraw
import cv2
from ultralytics import YOLO
from paddleocr import PaddleOCR
import yolov5
from pathlib import Path
import filetype 
import re


class LicensePlateFinder:
    def __init__(self) -> None:
        self.ocr_model = PaddleOCR(use_angle_cls=True, lang='en', show_log=False)
        self.objects_model = YOLO('yolov8n.pt')
        self.licenseplate_model = yolov5.load('keremberke/yolov5m-license-plate')
        self.gosnome_reg_exp = '([АВЕКМНОРСТУХавекмнорстухABEKMHOPCTYXabekmhopctyx]\s*\d{3}\s*[АВЕКМНОРСТУХавекмнорстухABEKMHOPCTYXabekmhopctyx]{2}\s*\d{2,3})|([АВЕКМНОРСТУХавекмнорстухABEKMHOPCTYXabekmhopctyx]{2}\s*\d{3}\s*\d{2,3})'

    def __call__(self, source:str, destination:str) -> None:
        source = Path(source)
        destination = Path(destination)

        assert source.is_file(), f'Source path {source} is not a file!'
        assert destination.parents[0].is_dir() and destination.suffix!='', f'Destination path {destination} invalid!'
        
        file_kind = filetype.guess(source).mime
        
        if 'video' in file_kind:
            print('Video inference run!')
            self.video_inference(source, destination)
            print('Video inference completed')
        elif 'image' in file_kind:
            print('Image inference run!')
            self.image_inference(source, destination)
            print('Image inference completed')
        else:
            print(f'File {source} not a video or an image')
            raise NotImplementedError


    def video_inference(self, source:Path, destination:Path) -> None:
        fps = int(iio.immeta(source, plugin="pyav")["fps"])
        codec = iio.immeta(source, plugin="pyav")["codec"]

        with iio.imopen(destination, "w", plugin="pyav") as out_file:
            out_file.init_video_stream(codec=codec, fps=fps)

            for frame in iio.imiter(source, plugin="pyav"):
                frame_result = self.frame_inference(frame)
                out_file.write_frame(frame_result)

    def image_inference(self, source:Path, destination:Path) -> None:
        frame_image = Image.open(source)
        result = self.frame_inference(np.array(frame_image))
        Image.fromarray(result).save(destination)

    def frame_inference(self, frame:np.array) -> np.array:
        frame_image = Image.fromarray(frame)
        draw = ImageDraw.Draw(frame_image)

        # Run batched inference on a list of images
        results = self.objects_model(frame_image, imgsz=frame.shape[:-1])[0]  # return a list of Results objects


        frame_res = self.licenseplate_model(frame_image)

        n, _ = frame_res.pred[0].shape
        for i in range(n):
            # if len(i) > 0:
            x1, y1, x2, y2 = frame_res.pred[0][i, :4]
            draw.rectangle((x1, y1, x2, y2), outline=(255, 10, 0), width=1)

            gosnomer = self.ocr_model.ocr(np.array(frame_image.crop((int(x1), int(y1), int(x2), int(y2)))), cls=True)

            if gosnomer[0] is not None:
                gosnomer = gosnomer[0][0][1][0]
            else:
                gosnomer = ''

            gosnomer = gosnomer.replace(' ', '').upper()

            # print(gosnomer)

            # gosnomer = self.extract_gosnomer(gosnomer)


            # if gosnomer == None:
            #     gosnomer = '--------'
            # else:
            #     gosnomer = gosnomer.string
            print(gosnomer)
            self.draw_text(
                frame,
                text=gosnomer,
                font=cv2.FONT_HERSHEY_SIMPLEX,
                pos=(int(x1)-20, int(y1)-30),
                font_scale=0.5,
                font_thickness=1,
                text_color=(0, 0, 0),
                text_color_bg=(255, 255, 255)
                )
        
        # results.orig_img = frame
        # results.orig_shape = frame.shape[:-1]
        frame = results.plot(img=frame)
        return frame

    def extract_gosnomer(self, nomer: str) -> str:
        return re.search(self.gosnome_reg_exp, nomer)

    def draw_text(
            self,
            img,
            text,
            font=cv2.FONT_HERSHEY_PLAIN,
            pos=(0, 0),
            font_scale=3,
            font_thickness=2,
            text_color=(0, 255, 0),
            text_color_bg=(0, 0, 0)
          ):
        
        x, y = pos
        text_size, _ = cv2.getTextSize(text, font, font_scale, font_thickness)
        text_w, text_h = text_size
        cv2.rectangle(img, (x -5, y - 5), (x + text_w + 5, y + text_h + 5), text_color_bg, -1)
        cv2.putText(img, text, (int(x), int(y + text_h + font_scale - 1)), font, font_scale, text_color, font_thickness)

        return text_size

if __name__ == '__main__':
    LicensePlateFinder()('/home/marat/Videos/1.mp4', '/home/marat/Videos/1_annotated.mp4')