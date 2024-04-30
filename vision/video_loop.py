import numpy as np
import imageio.v3 as iio
from licenseplates import get_license_plates, model
from PIL import Image, ImageDraw

source = '/home/marat/Videos/vlc-record-2024-04-30-12h56m23s-Street Sounds 2  _Road Nosie_.mp4-.mp4'
dest = '/home/marat/Videos/overley.mp4'


fps = iio.immeta(source, plugin="pyav")["fps"]

with iio.imopen(dest, "w", plugin="pyav") as out_file:
    out_file.init_video_stream(codec='vp9', fps=15)


    for frame in iio.imiter(source, plugin="pyav"):
        frame_image = Image.fromarray(frame)
        draw = ImageDraw.Draw(frame_image)
        frame_res = model(frame)
        for i in frame_res.pred:
            if len(i) > 0:
                x1, y1, x2, y2 = i[0, :4]
                draw.rectangle((x1, y1, x2, y2), outline=(0, 255, 0), width=5)
        out_file.write_frame(np.array(frame_image))