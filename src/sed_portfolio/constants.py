"""Class ordering and feature ordering from the archived challenge notebook."""

CLASS_NAMES = (
    "bell_ringing", "coffee_machine", "cutlery_dishes", "door_open_close",
    "footsteps", "keyboard_typing", "keychain", "light_switch", "microwave",
    "phone_ringing", "running_water", "toilet_flushing", "vacuum_cleaner",
    "wardrobe_drawer_open_close", "window_open_close"
)

FEATURE_GROUPS = (
    "zcr", "melspect", "mfcc", "mfcc_d", "mfcc_d2", "flux",
    "flatness", "centroid", "bandwidth", "contrast", "rolloff_low",
    "rolloff_high", "energy", "power"
)
STATS = ("mean", "std", "min", "max")
FEATURE_NAMES = tuple(f"{group}_{stat}" for group in FEATURE_GROUPS for stat in STATS)
SAMPLE_RATE = 16000
WINDOW_SECONDS = 1.0
WINDOW_HOP_SECONDS = 0.5
