# TorchRender

A library for testing and debugging graphics algorithms with torch. Or just graphics, though that may be slow depending on the GPU.

## Examples

The current best way to see how to use this library is by looking through the examples and tests.

### Conways's Game of Life + Color

![Running Example](https://thumbs.gfycat.com/GrotesqueComposedIndigobunting-size_restricted.gif)

[This example](https://github.com/SimLeek/TorchRender/blob/master/examples/interactive_test_pixel_shader.py) adds interaction between colors of different pixels to Conway's game of life. This doesn't really require a pixel shader, and could be done on normal pytorch, but the scrolling I added after the video above demonstrates a very simple usage for a pixel shader:

```python
def conway(frame, coords, finished):
    ...
    
    trans = np.zeros_like(coords)
    trans[0, ...] = np.ones(trans.shape[1:])
    frame[coords] = array[coords+trans]
```

Here, we're adding 1 to a dimensions of 'coords' which is a tensor holding the locations of each pixel and color. This shifts the entire image by one pixel. If we did something more complicated based on the values of the coordinates, we could add more complicated shaders, like barrel distortions.

## Installation

torchrender is distributed on [PyPI](https://pypi.org) as a universal
wheel and is available on Linux/macOS and Windows and supports
Python 3.5+ and PyPy.

```bash
$ pip install torchrender
```

## License

torchrender is distributed under the terms of the
[MIT License](https://choosealicense.com/licenses/mit>).
