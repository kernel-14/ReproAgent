"""Tiny optional-backend stub used when real torchvision is not installed.

The reproduction keeps full model/dataset loading lazy.  This module lets
availability checks for ``torchvision.datasets`` resolve in minimal smoke
environments without claiming that real torchvision training is available.
"""

