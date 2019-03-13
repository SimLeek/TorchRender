import torch
from torch import functional as F
import numpy as np
from cvpubsubs.webcam_pub import VideoHandlerThread
from cvpubsubs.input import mouse_loop
import cv2
from torchrender.pixel_shader import pixel_shader
from torchrender.geometry2d import line

img = np.zeros((480, 640, 3))

from threading import Lock

if False:
    from cvpubsubs.input import MouseEvent

frame_lock = Lock()

geom = []


def liney(frame, coords, finished):
    array = frame
    array = array.permute(2, 1, 0)[None, ...]
    global geom
    doubs = coords[0].float(), coords[1].float(), coords[2].float()
    for g in geom:
        array = g(array, doubs[1], doubs[0], doubs[2])
    geom = []
    array = array.squeeze().permute(2, 1, 0)
    frame[coords] = array[coords]


liney_shader = pixel_shader(liney)

a = None
b = None

import random as rnd


@mouse_loop
def conway_add(mouse_event  # type:MouseEvent
               ):
    with liney_shader.frame_lock:
        try:
            global img, a, b
            if mouse_event.flags == cv2.EVENT_FLAG_LBUTTON:
                if a is None:
                    a = (mouse_event.x, mouse_event.y)
                elif b is None:
                    b = (mouse_event.x, mouse_event.y)
                    if b != a:
                        geom.append(
                            line(a, b, thickness=2.0, radius=1.0, color=(rnd.random(), rnd.random(), rnd.random())))
                        a = b = None
                    else:
                        b = None
        except (ValueError, ZeroDivisionError) as e:
            a = b = None
            print(e)


VideoHandlerThread(video_source=img, callbacks=liney_shader).display()
