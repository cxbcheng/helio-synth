# Velocity bounds for colormaps with index `detrend_order`
# Partially based on values from `examples/print_velocity_stats.py`
V_MIN = {
    None: -3000,
    0: -3000,
    2: -500
}

V_MAX = {
    None: 3000,
    0: 3000,
    2: 500
}

# Equivalently, np.pi * (3 - np.sqrt(5))
GOLDEN_ANGLE = 2.399963229728653

# The fraction of the solar radius that is not the solar limbs and
# in general are points we want to sample from
DEFAULT_DISK_RADIUS_FRACTION = 0.95