"""Website controls mapped explicitly to the original Blender add-on properties."""
from pathlib import Path
import math

ROOT = Path(__file__).resolve().parents[1]
DEFAULTS = dict(fingers=3, palm=168, spread=0, length=0, tip=100, surface=0, sides=4)
PRESETS = {
    'baseline': ('Default V2 hand', DEFAULTS),
    'compact': ('Two fingers', dict(DEFAULTS, fingers=2, palm=140)),
    'four-finger': ('Four fingers', dict(DEFAULTS, fingers=4, palm=210)),
    'five-finger': ('Five fingers', dict(DEFAULTS, fingers=5, palm=260)),
    'extended': ('Extended links', dict(DEFAULTS, length=8, tip=120)),
    'surface': ('Deformed contact pads', dict(DEFAULTS, surface=5)),
    'hexagonal': ('Six-sided palm', dict(DEFAULTS, palm=190, sides=6)),
}
LIMITS = dict(fingers=(2, 5, 1), palm=(130, 280, 1), spread=(0, 20, 1),
              length=(0, 10, 1), tip=(70, 140, 5), surface=(0, 6, .5), sides=(4, 8, 1))


def validate(values):
    if not isinstance(values, dict) or set(values) != set(DEFAULTS):
        raise ValueError('Specify exactly: ' + ', '.join(DEFAULTS))
    result = {}
    for key, (low, high, step) in LIMITS.items():
        value = values[key]
        if isinstance(value, bool) or not isinstance(value, (int, float)) or not math.isfinite(value):
            raise ValueError(key + ' must be a finite number.')
        if not low <= value <= high or abs((value - low) / step - round((value - low) / step)) > 1e-5:
            raise ValueError(f'{key} must be between {low} and {high}, in steps of {step}.')
        result[key] = int(value) if step >= 1 else float(value)
    return result


def addon_parameters(values):
    values = validate(values)
    result = dict(finger_number=values['fingers'], thumb_number=1, thumb_side='right',
                  palm_size_mm=values['palm'], outline_sides=values['sides'],
                  outline_aspect_ratio=1.55, copy_finger_settings=False,
                  show_pad_settings=True, pad_kernel_count=1 if values['surface'] else 0,
                  pad_max_intensity=values['surface'], pad_resolution=4)
    for index in range(values['fingers']):
        prefix = f'finger_{index}'
        result[prefix + '_angle'] = ((index / (values['fingers'] - 1)) * 2 - 1) * values['spread']
        result[prefix + '_link_added_length'] = values['length']
        result[prefix + '_fingertip_scale'] = [values['tip'] / 100] * 3
        result[prefix + '_show_pad'] = True
        result[prefix + '_pad_kernel_count'] = 1 if values['surface'] else 0
        result[prefix + '_pad_max_intensity'] = min(values['surface'], 4)
    result['thumb_0_fingertip_scale'] = [values['tip'] / 100] * 3
    return result
