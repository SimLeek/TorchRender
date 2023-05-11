from displayarray.frame.frame_updater import FrameUpdater
from displayarray.input import key_loop
import torch
from torchrender.pixel_shader import pixel_shader
import numpy as np
import cv2

# from: https://www.geeks3d.com/hacklab/20180611/demo-simple-2d-grid-in-glsl/

width = 640
height = 480
img = np.zeros((height, width, 3))


def line(uv, line_segment, thickness):
    a, b = line_segment
    x1, y1 = a
    x2, y2 = b

    x, y = uv

    rise = (y2 - y1)
    run = x2 - x1
    b = y1 * run - x1 * rise

    bp1 = y1 * rise + x1 * run
    bp2 = y2 * rise + x2 * run

    is_line = (
            (y * run - (thickness / 2.0) * abs(rise) < x * rise + b + (thickness / 2.0) * abs(run)) *
            (y * run + (thickness / 2.0) * abs(rise) > x * rise + b - (thickness / 2.0) * abs(run)) *
            (y * rise > -x * run + bp1) *
            (y * rise < -x * run + bp2)
    )

    return is_line


lines = [
    ((-123, -456), (200, 300), 10),
    ((300, 300), (200, 300), 8),
    ((200, 400), (200, 300), 6),
    ((100, 300), (200, 300), 4)
]


def auto_size(lines, spacing=10):
    aspect_ratio = float(width) / height

    minx, miny = float('inf'), float('inf')
    maxx, maxy = -float('inf'), -float('inf')

    for l in lines:
        if l[0][0] < minx:
            minx = l[0][0]
        if l[1][0] < minx:
            minx = l[1][0]

        if l[0][1] < miny:
            miny = l[0][1]
        if l[1][1] < miny:
            miny = l[1][1]

        if l[0][0] > maxx:
            maxx = l[0][0]
        if l[1][0] > maxx:
            maxx = l[1][0]

        if l[0][1] > maxy:
            maxy = l[0][1]
        if l[1][1] > maxy:
            maxy = l[1][1]

    # todo: add assert for if any is still inf

    minx -= spacing
    miny -= spacing
    maxx += spacing
    maxy += spacing

    w = maxx - minx
    h = maxy - miny

    h_aspect = w*aspect_ratio
    w_aspect = h/aspect_ratio

    if w < w_aspect:
        new_w = h / aspect_ratio
        extra_w = new_w - w
        minx -= extra_w / 2
        maxx += extra_w / 2
    if h < h_aspect:
        new_h = w * aspect_ratio
        extra_h = new_h - h
        miny -= extra_h / 2
        maxy += extra_h / 2

    return (minx, miny), (maxx, maxy)

bounds = auto_size(lines)

def conway(frame, coords, finished):
    array = frame
    array = array.permute(2, 1, 0)[None, ...]

    x_mult = bounds[1][0]-bounds[0][0]
    y_mult = bounds[1][1]-bounds[0][1]

    uv = (
        coords[0]/height*x_mult+bounds[0][0],
        coords[1]/width*y_mult+bounds[0][1]
    )

    x = torch.zeros_like(coords[0], dtype=torch.bool)
    for l in lines:
        x |= line(uv, (l[0], l[1]), l[2])

    color = (torch.tensor([0.25, 0.75, 1.0])[None, :, None, None]).to(array.device)
    array = torch.ones_like(array) * color * (x.permute(2, 1, 0)[None, ...]).to(array.device)

    array = array.squeeze().permute(2, 1, 0)
    frame[coords] = array[coords]


conway_shader = pixel_shader(conway)

conway_shader(img)
cw = 12
ch = 18
tl = f"{bounds[0][0]},{bounds[0][1]}"
br = f"{bounds[1][0]},{bounds[1][1]}"
cv2.putText(img, tl, (int(cw), int(ch)), cv2.FONT_HERSHEY_COMPLEX_SMALL, 1, (.5, .5, .5))
cv2.putText(img, br, (int(width - cw * (len(br)+2)), int(height - ch * .5)), cv2.FONT_HERSHEY_COMPLEX_SMALL, 1,
            (.5, .5, .5))

FrameUpdater(video_source=img).display()
