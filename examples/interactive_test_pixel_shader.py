import torch
from torch import functional as F
import numpy as np
from cvpubsubs.webcam_pub import VideoHandlerThread
from cvpubsubs.input import mouse_loop
import cv2
from torchrender.pixel_shader import pixel_shader

img = np.ones((240, 320, 3))


def setup_conway_r(array):
    neighbor_weights = torch.ones(torch.Size([3, 3, 3, 3]))
    neighbor_weights[:, :, 1, 1, ...] = 0
    neighbor_weights[:, 1, :, :, ...] = .5
    neighbor_weights[:, 2, :, :, ...] = 0
    neighbor_weights[1, :, :, :, ...] = 0
    neighbor_weights[2, :, :, :, ...] = 0
    neighbor_weights = torch.Tensor(neighbor_weights).type_as(array).to(array.device)
    neighbors = torch.nn.functional.conv2d(array, neighbor_weights, stride=1, padding=1)
    return neighbors


def setup_conway_g(array):
    neighbor_weights = torch.ones(torch.Size([3, 3, 3, 3]))
    neighbor_weights[:, :, 1, 1, ...] = 0
    neighbor_weights[:, 0, :, :, ...] = 0
    neighbor_weights[:, 2, :, :, ...] = .5
    neighbor_weights[0, :, :, :, ...] = 0
    neighbor_weights[2, :, :, :, ...] = 0
    neighbor_weights = torch.Tensor(neighbor_weights).type_as(array).to(array.device)
    neighbors = torch.nn.functional.conv2d(array, neighbor_weights, stride=1, padding=1)
    return neighbors


def setup_conway_b(array):
    neighbor_weights = torch.ones(torch.Size([3, 3, 3, 3]))
    neighbor_weights[:, :, 1, 1, ...] = 0
    neighbor_weights[:, 0, :, :, ...] = .5
    neighbor_weights[:, 1, :, :, ...] = 0
    neighbor_weights[0, :, :, :, ...] = 0
    neighbor_weights[1, :, :, :, ...] = 0
    neighbor_weights = torch.Tensor(neighbor_weights).type_as(array).to(array.device)
    neighbors = torch.nn.functional.conv2d(array, neighbor_weights, stride=1, padding=1)
    return neighbors


from threading import Lock

frame_lock = Lock()


def conway(frame, coords, finished):
    array = frame
    array = array.permute(2, 1, 0)[None, ...]
    neighbors = setup_conway_r(array) + setup_conway_g(array) + setup_conway_b(array)
    live_array = torch.where((neighbors < 2) | (neighbors > 3),
                             torch.zeros_like(array),
                             torch.where((2 <= neighbors) & (neighbors <= 3),
                                         torch.ones_like(array),
                                         array
                                         )
                             )
    dead_array = torch.where(neighbors == 3,
                             torch.ones_like(array),
                             array)
    array = torch.where(array == 1.0,
                        live_array,
                        dead_array
                        )
    array = array.squeeze().permute(2, 1, 0)
    trans = np.zeros_like(coords)
    trans[0, ...] = np.ones(trans.shape[1:])
    frame[coords] = array[coords+trans]


conway_shader = pixel_shader(conway)


@mouse_loop
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


VideoHandlerThread(video_source=img, callbacks=conway_shader).display()
