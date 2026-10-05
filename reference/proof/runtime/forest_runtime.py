"""Match the frozen ARM64 reference's fused Gini split comparison."""
import sys
if sys.platform == 'emscripten':
    from forest_criterion import NativeGini
    from sklearn.tree import _classes
    _classes.CRITERIA_CLF['gini'] = NativeGini
