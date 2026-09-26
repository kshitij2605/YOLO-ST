"""Input geometry helpers.

``data.img_size`` may be a single integer (square inputs, every config before
patch 0017) or ``[height, width]``. All code that needs the two sides goes
through :func:`image_hw`, so square configs keep their exact arithmetic.
"""


def image_hw(img_size):
    """Return ``(height, width)`` for an int or a two-element sequence."""
    if isinstance(img_size, (list, tuple)):
        if len(img_size) != 2:
            raise ValueError(f'img_size must be an int or [height, width], got {img_size!r}')
        return int(img_size[0]), int(img_size[1])
    return int(img_size), int(img_size)
