import math as m
import torch


# def line(a, b, m):
#    def apply(x, y):
#        return x * a + y * b + m

#    return apply

def line(point1, point2, thickness, radius):
    # https://developer.nvidia.com/gpugems/GPUGems2/gpugems2_chapter22.html
    x0, y0, x1, y1 = point1[0], point1[1], point2[0], point2[1]
    k = 2.0 / ((2 * radius + thickness) * m.sqrt(abs((x0 - x1) ** 2 - (y0 - y1) ** 2))+1)
    e0 = (k * (y0 - y1), k * (x1 - x0), 1 + k * (x0 * y1 - x1 * y0))
    e1 = (k * (x1 - x0), k * (y1 - y0), k * (x0 ** 2 + y0 ** 2 - x0 * x1 - y0 * y1))
    e2 = (k * (y1 - y0), k * (x0 - x1), k * (x1 * y0 - x0 * y1))
    e3 = (k * (x0 - x1), k * (y0 - y1), k * (x1 ** 2 + y1 ** 2 - x0 * x1 - y0 * y1))

    def apply(x, y, c):
        d0 = e0[0] * x + e0[1] * y + e0[2]
        d1 = e1[0] * x + e1[1] * y + e1[2]
        d2 = e2[0] * x + e2[1] * y + e2[2]
        d3 = e3[0] * x + e3[1] * y + e3[2]

        img = torch.where((d0 > 0) & (d1 > 0) & (d2 > 0) & (d3 > 0),
                          torch.min(d0, d2) * torch.min(d1, d3),
                          torch.zeros_like(x))
        return img

    return apply
