import numpy as np

from eurobot_render import EurobotCV2Renderer


def test_renderer_draws_image(world):
    renderer = EurobotCV2Renderer(world, size=(200, 150))
    img = renderer.draw_snapshot(show=False)
    assert isinstance(img, np.ndarray)
    assert img.shape == (150, 200, 3)
