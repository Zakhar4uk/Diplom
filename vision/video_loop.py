#!/usr/bin/env python
# -*- coding: utf-8 -*-
import os
os.environ["KMP_DUPLICATE_LIB_OK"]="TRUE"
# os.environ["OMP_NUM_THREADS"] = "1"
import numpy as np
import imageio.v3 as iio
from licenseplates import get_license_plates, model, get_characters
from PIL import Image, ImageDraw, ImageFont
import cv2
import sys


from ultralytics import YOLO
from paddleocr import PaddleOCR


ocr = PaddleOCR(use_angle_cls=True, lang='en')

import re


source = sys.argv[1]
dest = 'D:/proverka/videos/primer_1.mp4'



def draw_text(img, text,
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

def is_gosnomer(nomer: str) -> bool:
    return re.search('([АВЕКМНОРСТУХавекмнорстухABEKMHOPCTYXabekmhopctyx]\s*\d{3}\s*[АВЕКМНОРСТУХавекмнорстухABEKMHOPCTYXabekmhopctyx]{2}\s*\d{2,3})|([АВЕКМНОРСТУХавекмнорстухABEKMHOPCTYXabekmhopctyx]{2}\s*\d{3}\s*\d{2,3})', nomer)

fps = iio.immeta(source, plugin="pyav")["fps"]

with iio.imopen(dest, "w", plugin="pyav") as out_file:
    out_file.init_video_stream(codec='vp9', fps=24)


    for frame in iio.imiter(source, plugin="pyav"):

        frame_image = Image.fromarray(frame)
        draw = ImageDraw.Draw(frame_image)
        
        objects_model = YOLO('yolov8n.pt')  # pretrained YOLOv8n model

        # Run batched inference on a list of images
        results = objects_model(frame_image)[0]  # return a list of Results objects

        frame_image = Image.fromarray(results.plot()[:, :, ::-1])


        frame_res = model(frame)
        n, _ = frame_res.pred[0].shape
        for i in range(n):
            # if len(i) > 0:
            x1, y1, x2, y2 = frame_res.pred[0][i, :4]
            draw.rectangle((x1, y1, x2, y2), outline=(255, 10, 0), width=1)
            img = np.array(frame_image)

            gosnomer = ocr.ocr(np.array(frame_image.crop((int(x1), int(y1), int(x2), int(y2)))), cls=True)

            if gosnomer[0] is not None:
                gosnomer = gosnomer[0][0][1][0]
            else:
                gosnomer = ''

            gosnomer = gosnomer.replace(' ', '').upper()

            print(gosnomer)

            gosnomer = is_gosnomer(gosnomer)

            if gosnomer != None:
                gosnomer = gosnomer.string
                draw_text(
                    img,
                    text=gosnomer,
                    font=cv2.FONT_HERSHEY_SIMPLEX,
                    pos=(int(x1)-20, int(y1)-30),
                    font_scale=0.5,
                    font_thickness=1,
                    text_color=(0, 0, 0),
                    text_color_bg=(255, 255, 255)
                    )
            frame_image = Image.fromarray(img)
            
        out_file.write_frame(np.array(frame_image))