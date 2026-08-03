from astropy.time import Time


def require_tai(t: Time) -> None:
    """
    Asserts that a Time is in the TAI scale.
    TAI is made standard throughout this project for consistency and
    its convenience with scientific code.
    :raises ValueError: if t.scale != 'tai'.
    """
    if t.scale != 'tai':
        raise ValueError(f'Time scale must be "tai", got "{t.scale}"')