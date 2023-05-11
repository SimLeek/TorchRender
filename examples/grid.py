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


def grid(st, res):
    g1 = (st[0] * res, st[1] * res)
    grid = [g - torch.floor(g) for g in g1]
    return (res < grid[0]) * (res < grid[1])


scale = 5.0
resolution = 0.1955


def conway(frame, coords, finished):
    array = frame
    array = array.permute(2, 1, 0)[None, ...]

    grid_uv = (coords[0] * scale, coords[1] * scale)
    x = grid(grid_uv, resolution)
    color = (torch.tensor([0.08,0.09,0.1])[None, :, None, None]).to(array.device)
    array = torch.ones_like(array) * color * (x.permute(2, 1, 0)[None, ...]).to(array.device)

    array = array.squeeze().permute(2, 1, 0)
    frame[coords] = array[coords]


conway_shader = pixel_shader(conway)

'''@mouse_loop
def conway_add(mouse_event  # type:MouseEvent
               ):
    with conway_shader.frame_lock:
        try:
            rnd_w = min(100, 50 + img.shape[1] - mouse_event.x, max(0, mouse_event.x) + 50)
            rnd_h = min(100, 50 + img.shape[0] - mouse_event.y, max(0, mouse_event.y) + 50)
            if mouse_event.flags == cv2.EVENT_FLAG_LBUTTON:
                img[max(0, mouse_event.y - 50):mouse_event.y + 50, max(0, mouse_event.x - 50):mouse_event.x + 50,
                0] = np.random.randint(0, 2, (rnd_h, rnd_w))
            elif mouse_event.flags == cv2.EVENT_FLAG_MBUTTON:
                img[max(0, mouse_event.y - 50):mouse_event.y + 50, max(0, mouse_event.x - 50):mouse_event.x + 50,
                :] = 0
            elif mouse_event.flags == cv2.EVENT_FLAG_RBUTTON:
                img[max(0, mouse_event.y - 50):mouse_event.y + 50, max(0, mouse_event.x - 50):mouse_event.x + 50,
                2] = np.random.randint(0, 2, (rnd_h, rnd_w))
        except ValueError as e:
            pass
'''


@key_loop
def keys(k):
    global scale, resolution
    if k == 'w':
        scale += 1
    elif k == 's':
        scale -= 1
    elif k == 'a':
        resolution -= .01
    elif k == 'd':
        resolution += .01
    conway_shader(img)
    print(f'scale:{scale}, resolution:{resolution}')

conway_shader(img)
FrameUpdater(video_source=img).display()
