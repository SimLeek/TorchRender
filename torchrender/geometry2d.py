import math as m
import torch
from numbers import Real


class Point2D(object):
    def __init__(self, x, y):
        self.x = x
        self.y = y

    def __len__(self):
        return 2

    def __getitem__(self, item):
        return {
            0: self.x,
            "0": self.x,
            "x": self.x,
            "X": self.x,
            1: self.y,
            "1": self.y,
            "y": self.y,
            "Y": self.y,
        }[item]


class ShaderLine2D(object):
    def __init__(self, point_pair, thickness=None, radius=None, color=None):
        if thickness is None:
            thickness = 2.0
        if radius is None:
            radius = 0.2
        if color is None:
            color = (1.0, 1.0, 1.0)

        if not isinstance(thickness, Real):
            raise ValueError("Line thickness should be a real number.")
        if not isinstance(radius, Real):
            raise ValueError("Line radius should be a real number.")
        if not len(point_pair) == 2:
            raise ValueError("Line should have two points")

        point1 = point_pair[0]
        point2 = point_pair[1]

        if not len(point1) == 2 or not len(point2) == 2:
            raise ValueError("Point should have length 2")
        if not all([isinstance(point1[0], Real), isinstance(point1[1], Real),
                    isinstance(point2[0], Real), isinstance(point2[1], Real)]):
            raise ValueError("Points should be composed of real numbers.")

        x0, y0, x1, y1 = point1[0], point1[1], point2[0], point2[1]
        k = 2.0 / ((2 * radius + thickness) * m.sqrt(abs((x0 - x1) ** 2 - (y0 - y1) ** 2)) + 1)
        self.e0 = (k * (y0 - y1), k * (x1 - x0), 1 + k * (x0 * y1 - x1 * y0))
        self.e1 = (k * (x1 - x0), k * (y1 - y0), k * (x0 ** 2 + y0 ** 2 - x0 * x1 - y0 * y1))
        self.e2 = (k * (y1 - y0), k * (x0 - x1), k * (x1 * y0 - x0 * y1))
        self.e3 = (k * (x0 - x1), k * (y0 - y1), k * (x1 ** 2 + y1 ** 2 - x0 * x1 - y0 * y1))
        self.color = color


def line(point1, point2, thickness, radius, color=None):
    # https://developer.nvidia.com/gpugems/GPUGems2/gpugems2_chapter22.html
    l = ShaderLine2D((point1, point2), thickness, radius, color)

    def apply(img, x, y, c):
        x = x.permute(2, 1, 0)[None, ...]
        y = y.permute(2, 1, 0)[None, ...]

        e0 = l.e0
        e1 = l.e1
        e2 = l.e2
        e3 = l.e3
        d0 = e0[0] * x + e0[1] * y + e0[2]
        d1 = e1[0] * x + e1[1] * y + e1[2]
        d2 = e2[0] * x + e2[1] * y + e2[2]
        d3 = e3[0] * x + e3[1] * y + e3[2]

        line_img = torch.min(d0, d2) * torch.min(d1, d3)

        for col in range(len(l.color)):
            line_img[..., col, :, :] = line_img[..., col, :, :] * l.color[col]

        img = torch.where((d0 > 0) & (d1 > 0) & (d2 > 0) & (d3 > 0),
                          line_img,
                          img)

        return img

    return apply
