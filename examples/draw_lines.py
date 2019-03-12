import torch
from torch import functional as F
import numpy as np
from cvpubsubs.webcam_pub import VideoHandlerThread
from cvpubsubs.input import mouse_loop
import cv2
from torchrender.pixel_shader import pixel_shader
from torchrender.geometry2d import line

img = np.zeros((600, 800, 3))

from threading import Lock
if False:
    from cvpubsubs.input import MouseEvent

frame_lock = Lock()

geom = []

def liney(frame, coords, finished):
    array = frame
    array = array.permute(2, 1, 0)[None, ...]
    global geom
    for g in geom:
        doubs = coords[0].double(), coords[1].double(), coords[2].double()
        applied = g(doubs[1], doubs[0], doubs[2]).permute(2, 1, 0)[None, ...]
        array += applied
    geom = []
    array = array.squeeze().permute(2, 1, 0)
    #trans = np.zeros_like(coords)
    #trans[0, ...] = np.ones(trans.shape[1:])
    frame[coords] = array[coords]


liney_shader = pixel_shader(liney)

a = None
b = None

@mouse_loop
def conway_add(mouse_event  # type:MouseEvent
               ):
    with liney_shader.frame_lock:
        try:
            global img, a, b
            if mouse_event.event == cv2.EVENT_LBUTTONDOWN:
                if a is None:
                    a = (mouse_event.x, mouse_event.y)
                elif b is None:
                    b = (mouse_event.x, mouse_event.y)
                    if b!=a:
                        geom.append(line(a, b, thickness=1.5, radius=1.0))
                        a = b = None
                    else:
                        b = None

            elif mouse_event.flags == cv2.EVENT_FLAG_MBUTTON:
                pass
            elif mouse_event.flags == cv2.EVENT_FLAG_RBUTTON:
                pass
        except (ValueError, ZeroDivisionError) as e:
            a = b = None
            print(e)




VideoHandlerThread(video_source=img, callbacks=liney_shader).display()
