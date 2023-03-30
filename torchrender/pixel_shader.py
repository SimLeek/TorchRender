import numpy as np
from torchrender.util import quit_window
from threading import RLock


class pixel_shader(object):  # NOSONAR

    __slots__ = ['frame_lock', 'looping', 'first_call', 'inner_function', 'device', 'min_bounds', 'max_bounds',
                 'space_grid', 'x', 'y', 'c']

    def __init__(self, display_function, finish_function=None):
        """Used for running arbitrary functions on pixels.

        >>> import random
        >>> import torch
        >>> from displayarray.frame.frame_updater import FrameUpdater
        >>> img = np.zeros((600, 800, 3))
        >>> def fun(array, coords, finished):
        ...     rgb = torch.empty(array.shape).uniform_(0,1).type(torch.DoubleTensor).to(array.device)/300.0
        ...     trans = torch.zeros(array.shape).to(array.device).type_as(coords[0])
        ...     trans[0,...] = torch.ones(trans.shape[1:])
        ...     # this should give an out of bounds error:
        ...     array[coords] =(array[tuple(c+t for c,t in zip(coords,trans))] + rgb[coords])%1.0
        >>> FrameUpdater(video_source=img, callbacks=pixel_shader(fun)).display()

        thanks: https://medium.com/@awildtaber/building-a-rendering-engine-in-tensorflow-262438b2e062

        :param display_function:
        :param finish_function:
        """

        import torch
        from torch.autograd import Variable

        if isinstance(display_function, pixel_shader):
            self.frame_lock = display_function.frame_lock
        else:
            self.frame_lock = RLock()

        self.looping = True
        self.first_call = True

        def _run_finisher(self, frame, finished, *args, **kwargs):
            if not callable(finish_function):
                quit_window()
            else:
                finished = finish_function(frame, Ellipsis, finished, *args, **kwargs)
                if finished:
                    quit_window()

        def _setup(self, frame, cam_id, *args, **kwargs):

            if "device" in kwargs:
                self.device = torch.device(kwargs["device"])
            else:
                if torch.cuda.is_available():
                    self.device = torch.device("cuda")
                else:
                    self.device = torch.device("cpu")

            self.min_bounds = [0 for _ in frame.shape]
            self.max_bounds = list(frame.shape)
            grid_slices = [torch.arange(self.min_bounds[d], self.max_bounds[d]) for d in range(len(frame.shape))]
            space_grid = torch.meshgrid(grid_slices)
            self.x = space_grid
            self.x = list(self.x)

        def _display_internal(self, frame, *args, **kwargs):
            finished = True
            if self.first_call:
                # return to display initial frame
                _setup(self, frame, finished, *args, **kwargs)
                self.first_call = False
            if self.looping:
                with self.frame_lock:
                    tor_frame = torch.from_numpy(frame).to(self.device)
                    tor_frame = tor_frame.half()

                    if self.x[0].device != tor_frame.device:
                        self.x[0] = self.x[0].to(device=tor_frame.device)
                        self.x[1] = self.x[1].to(device=tor_frame.device)
                    finished = display_function(tor_frame, self.x, finished, *args, **kwargs)
                    frame[...] = tor_frame.cpu().numpy()[...]
            if finished:
                self.looping = False
                with self.frame_lock:
                    _run_finisher(self, frame, finished, *args, **kwargs)

        self.inner_function = _display_internal

    def __call__(self, *args, **kwargs):
        return self.inner_function(self, *args, **kwargs)
