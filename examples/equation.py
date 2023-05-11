from displayarray.frame.frame_updater import FrameUpdater
from displayarray.input import key_loop
import torch
from torchrender.pixel_shader import pixel_shader
import numpy as np
import cv2

# todo: to get lines, finding the exact 0-intercept is mandatory because 2 pixels could be -400z and 400z
#  with 0 in between. So to get the 0-intercept, you take the current pixel, the one to the right of it,
#  the one below it, and the one to the bottom right, and construct a plane going through all of those points.
#  You can then set z=0 in the plane equation to find the x&y line at zero. Then,
#  using the inverse slope and the top left pixel (0,0) you can find the line going from the current top left pixel to
#  the 0-intercept line. Then do a linear algebra system of equations to solve for the x and y point
#  where both lines meet. If 0>=x>1 and 0>=y>1, that is a valid point, and separate x tensors and y tensors should
#  be given the x and y coordinates, -1 otherwise.

# todo: run cv2.find_contours on the previous outcome (x or y) to get all the separate lines,
#  since there can be zero, one, or many per equation.

# todo: iterate over the the points in the contours and add the values from the x&y tensors to get
#  the exact point locations

# todo: iterate over the points in the contours, if x2 in contiguous points x1, x2, x3, has a distance
#  from the line from x1 to x3 that's smaller than the minimum specified distance specified by 'resolution',
#  remove x2. Repeat this process until no points are removes.

# todo: add the revolve function, 0-360, and generate a quad mesh based on the lines

# todo: convert the quad mesh into a tri mesh (trimesh.geometry.triangulate_quads)

# todo: use the python trimesh library for its boolean functions to combine meshes

# todo: use trimesh's watertight function to check if a mesh is valid for printing, highlight it (red) if not

# todo: use trimesh's stl exporter

# todo: use trimesh's vertex_defects function to find concave vs convex vertex points
# todo: combine neighboring vertices to find concave/convex regions and enumerate them

# todo: use trimesh's face ordering to be able to reference faces
# todo: show face numbers in UI
# todo: make/use face ordering algorithm on convex/concave regions

width = 1024
height = 768
img = np.zeros((height, width, 3))

from math import pi
from torch import sin, cos, tan, abs


def muscle_1(y, x):
    is_equation = (
            (8 * sin(16 * x + pi / 2)) / ((x ** 2) / 4 + 1) + 2 * y ** 2 + (x ** 2) / 4
            -
            3 ** 2
    )

    is_equation_2 = (
            (8 * sin(16 * x + pi / 2)) / ((x ** 2) / 5 + 1) + 1.5 * y ** 2 + (x ** 2) / 4
            -
            3.2 ** 2
    )

    closest = torch.where(
        torch.abs(is_equation) < torch.abs(is_equation_2),
        is_equation,
        is_equation_2
    )

    return closest


def muscle_2(y, x):
    is_equation = (
            (2 * sin(16 * x + pi / 2)) / ((abs(x) * 2 - 5) ** 2 + 3) + 2 * y ** 2 + (x ** 2) / 4
            -
            3 ** 2
    )

    is_equation_2 = (
            (2 * sin(16 * x + pi / 2)) / ((abs(x) * 1.5 - 4) ** 2 + 3) + 1.5 * y ** 2 + (x ** 2) / 4
            -
            3.2 ** 2
    )

    closest = torch.where(
        torch.abs(is_equation) < torch.abs(is_equation_2),
        is_equation,
        is_equation_2
    )

    return closest


def eq(uv):
    y, x = uv

    # is_equation = x**2 + y**2 - 2**2

    # is_equation = torch.tan(x ** 2) + torch.tan(y ** 2) - 0

    # is_equation = y - torch.sin(x)

    is_equation = muscle_2(y,x)

    return is_equation


bounds = ((-7, -7 * width / height), (7, 7 * width / height))

zminmax = 1.0
dist = 0.01


def conway(frame, coords, finished):
    array = frame
    array = array.permute(2, 1, 0)[None, ...]

    x_mult = bounds[1][0] - bounds[0][0]
    y_mult = bounds[1][1] - bounds[0][1]

    uv = (
        coords[0] / height * x_mult + bounds[0][0],
        coords[1] / width * y_mult + bounds[0][1]
    )

    # uv = (coords[0], coords[1])

    x = eq(uv)

    # eq to snap min/mix dist from equation to 0-1 so whole window is red/yellow
    # However, it looked bad for some equations, so changed it to a variable that the user could modify
    x = ((x + zminmax) / (zminmax + zminmax)) * 2 - 1

    x = (x.permute(2, 1, 0)[None, ...]).to(array.device)

    orange = (torch.tensor([0.0, 1.0, 1.0])[None, :, None, None]).to(array.device)
    yellow = (torch.tensor([0.0, 0.5, 1.0])[None, :, None, None]).to(array.device)
    red = (torch.tensor([0.0, 0.0, 1.0])[None, :, None, None]).to(array.device)

    y = torch.ones_like(x)

    # red
    y[:, 0, ...][x[:, 0, ...] > 0] = (
                                             1.0 - torch.minimum(
                                         x[:, 0, ...][x[:, 0, ...] > 0],
                                         torch.ones_like(x[:, 0, ...][x[:, 0, ...] > 0])
                                     )) * red[0, 0, 0, 0]
    y[:, 1, ...][x[:, 1, ...] > 0] = (
                                             1.0 - torch.minimum(
                                         x[:, 1, ...][x[:, 1, ...] > 0],
                                         torch.ones_like(x[:, 1, ...][x[:, 1, ...] > 0])
                                     )) * red[0, 1, 0, 0]
    y[:, 2, ...][x[:, 2, ...] > 0] = (
                                             1.0 - torch.minimum(
                                         x[:, 2, ...][x[:, 2, ...] > 0],
                                         torch.ones_like(x[:, 2, ...][x[:, 2, ...] > 0])
                                     )) * red[0, 2, 0, 0]
    # yellow
    y[:, 0, ...][x[:, 0, ...] < 0] = (
                                             1.0 + torch.maximum(
                                         x[:, 0, ...][x[:, 0, ...] < 0],
                                         -torch.ones_like(x[:, 0, ...][x[:, 0, ...] < 0])
                                     )) * yellow[0, 0, 0, 0]
    y[:, 1, ...][x[:, 1, ...] < 0] = (
                                             1.0 + torch.maximum(
                                         x[:, 1, ...][x[:, 1, ...] < 0],
                                         -torch.ones_like(x[:, 1, ...][x[:, 1, ...] < 0])
                                     )) * yellow[0, 1, 0, 0]
    y[:, 2, ...][x[:, 2, ...] < 0] = (
                                             1.0 + torch.maximum(
                                         x[:, 2, ...][x[:, 2, ...] < 0],
                                         -torch.ones_like(x[:, 2, ...][x[:, 2, ...] < 0])
                                     )) * yellow[0, 2, 0, 0]

    # orange
    x[:, 0, ...] = torch.where(
        torch.abs(x[:, 0, ...]) < dist,
        torch.ones_like(x[:, 0, ...]) * orange[0, 0, 0, 0],
        y[:, 0, ...]
    )
    x[:, 1, ...] = torch.where(
        torch.abs(x[:, 1, ...]) < dist,
        orange[0, 1, 0, 0],
        y[:, 1, ...]
    )
    x[:, 2, ...] = torch.where(
        torch.abs(x[:, 2, ...]) < dist,
        orange[0, 2, 0, 0],
        y[:, 2, ...]
    )
    '''x[:, 0, ...][torch.abs(x[:, 0, ...]) < 0.5] = orange[0, 0, 0, 0]
    x[:, 1, ...][torch.abs(x[:, 1, ...]) < 0.5] = orange[0, 1, 0, 0]
    x[:, 2, ...][torch.abs(x[:, 2, ...]) < 0.5] = orange[0, 2, 0, 0]

    '''

    array = x.to(array.dtype)

    array = array.squeeze().permute(2, 1, 0)
    frame[coords] = array[coords]


conway_shader = pixel_shader(conway)


# cw = 12
# ch = 18
# tl = f"{bounds[0][0]},{bounds[0][1]}"
# br = f"{bounds[1][0]},{bounds[1][1]}"
# cv2.putText(img, tl, (int(cw), int(ch)), cv2.FONT_HERSHEY_COMPLEX_SMALL, 1, (.5, .5, .5))
# cv2.putText(img, br, (int(width - cw * (len(br)+2)), int(height - ch * .5)), cv2.FONT_HERSHEY_COMPLEX_SMALL, 1,
#            (.5, .5, .5))

@key_loop
def keys(k):
    global zminmax, dist
    if k == 'q':
        zminmax += 1.0
    elif k == 'a':
        zminmax -= 1
    elif k == 'e':
        dist -= 0.01
    elif k == 'd':
        dist += 0.01
    '''elif k == 'w':
        zmax -= 1
    elif k == 's':
        zmax += 1'''

    conway_shader(img)
    print(f'zminmax:{zminmax}, dist:{dist}')


conway_shader(img)
FrameUpdater(video_source=img).display()
