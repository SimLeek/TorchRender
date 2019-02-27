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
        >>> from cvpubsubs.webcam_pub import VideoHandlerThread
        >>> img = np.zeros((600, 800, 3))
        >>> def fun(array, coords, finished):
        ...     rgb = torch.empty(array.shape).uniform_(0,1).type(torch.DoubleTensor).to(array.device)/300.0
        ...     trans = np.zeros_like(coords)
        ...     trans[0,...] = np.ones(trans.shape[1:])
        ...     array[coords] = (array[coords+trans] + rgb[coords])%1.0
        >>> VideoHandlerThread(video_source=img, callbacks=pixel_shader(fun)).display()

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
            grid_slices = [slice(self.min_bounds[d], self.max_bounds[d]) for d in range(len(frame.shape))]
            space_grid = np.mgrid[grid_slices]
            space_grid.flags.writeable = False
            x_tens = torch.LongTensor(space_grid[0, ...]).to(self.device)
            y_tens = torch.LongTensor(space_grid[1, ...]).to(self.device)
            c_tens = torch.LongTensor(space_grid[2, ...]).to(self.device)
            self.x = Variable(x_tens, requires_grad=False)
            self.y = Variable(y_tens, requires_grad=False)
            self.c = Variable(c_tens, requires_grad=False)

        def _display_internal(self, frame, cam_id, *args, **kwargs):
            finished = True
            if self.first_call:
                # return to display initial frame
                _setup(self, frame, finished, *args, **kwargs)
                self.first_call = False
                return
            if self.looping:
                with self.frame_lock:
                    tor_frame = torch.from_numpy(frame).to(self.device)
                    finished = display_function(tor_frame, (self.x, self.y, self.c), finished, *args, **kwargs)
                    frame[...] = tor_frame.cpu().numpy()[...]
            if finished:
                self.looping = False
                with self.frame_lock:
                    _run_finisher(self, frame, finished, *args, **kwargs)

        self.inner_function = _display_internal

    def __call__(self, *args, **kwargs):
        return self.inner_function(self, *args, **kwargs)
