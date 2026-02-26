"""Legacy model utilities.

The functionality that used to live in this file has been split into
`src/models/faster_rcnn.py` and `src/models/yolo_wrapper.py`.  This
module still exists for backward compatibility, but it simply raises a
:class:DeprecationWarning if used.
"""

import warnings


def create_model(*args, **kwargs):
    warnings.warn(
        "src.models.model.create_model is deprecated; use the specific factory "
        "functions in src.models.faster_rcnn or src.models.yolo_wrapper",
        DeprecationWarning,
    )
    raise NotImplementedError("Please import and call create_faster_rcnn or create_yolo")
