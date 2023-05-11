from displayarray.frame.frame_updater import FrameUpdater
from displayarray.input import key_loop
import torch
from torchrender.pixel_shader import pixel_shader
import numpy as np
import cv2

# from: https://www.geeks3d.com/hacklab/20190225/demo-checkerboard-in-glsl/

width = 640
height = 480
img = np.zeros((height, width, 3))


def checker(uv, repeats):
    cx = torch.floor(repeats * uv[0])
    cy = torch.floor(repeats * uv[1])
    result = torch.fmod(cx+cy, 2.0)
    r = torch.sign(result)
    return r


checker_size = 10.0
def conway(frame, coords, finished):
    array = frame
    array = array.permute(2, 1, 0)[None, ...]

    uv = [coords[0] / torch.max(coords[0][:,0,0]) * (height/width),
          coords[1]/torch.max(coords[1][0,:,0])]
    c = (checker(uv, checker_size) + 1.0)/2.0
    color = (torch.tensor([0.08,0.09,0.1])[None, :, None, None]).to(array.device)
    array = torch.ones_like(array) * color * (c.permute(2, 1, 0)[None, ...]).to(array.device)

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
    global checker_size, img
    if k == 'w':
        checker_size += 1
    elif k == 's':
        checker_size -= 1
    conway_shader(img)

    print(f'checker_size:{checker_size}')

conway_shader(img)
FrameUpdater(video_source=img).display()
