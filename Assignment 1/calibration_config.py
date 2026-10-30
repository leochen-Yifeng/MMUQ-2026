"""Parameter bounds and explicit choices for Assignment 1."""

# Transcribed from the supplied Moodle screenshot, sketch (1).png.
# Parameter names, rather than screenshot comments, identify the model inputs.
PARAMETER_BOUNDS = {
    # Snow.
    'snw_dth': (0.00, 10.0),
    'snw_att': (-2.0, 3.00),
    'snw_pmf': (0.00, 3.00),
    'snw_amf': (0.00, 1e1),
    # Soil.
    'sl0_dth': (0.00, 50.0),
    'sl0_pwp': (5.00, 7e2),
    'sl0_fcy': (100., 7e2),
    'sl0_bt0': (0.01, 1e1),
    # Initial reservoir storage.
    'urr_dth': (0.00, 10.0),
    'lrr_dth': (0.00, 50.0),
    # Upper reservoir.
    'urr_wsr': (0.00, 1.00),
    'urr_ulc': (0.00, 1.00),
    'urr_tdh': (0.00, 2e2),
    'urr_tdr': (0.01, 1.00),
    'urr_ndr': (0.00, 1e-1),
    'urr_uct': (0.00, 1e-1),
    # Lower reservoir.
    'lrr_dre': (0.00, 1e-2),
    'lrr_lct': (0.00, 1e-2),
}

# These are implementation choices, not requirements stated in the assignment.
# All observations are used for calibration; there is no held-out validation.
WARMUP_STEPS = 0
RANDOM_SEED = 42
MAX_ITERATIONS = 3000
POPULATION_MULTIPLIER = 8
RELATIVE_TOLERANCE = 0.001

# Use DE alone so that all search-history plots describe the same algorithm.
POLISH = False
