"""Build the ADC kernel: ``python -m turboquant_pro._adc``."""

import sys

from . import build, is_available

path = build()
print(f"built {path}")
sys.exit(0 if is_available() else 1)
