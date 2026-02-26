"""Simple wrapper for YOLO-style models.

Currently this is just a placeholder; depending on the YOLO implementation
used (e.g. Ultralytics/yolov5), this module can be fleshed out to provide a
uniform interface.
"""

import torch


class YOLOWrapper(torch.nn.Module):
    def __init__(self, num_classes: int, version: str = "v3", device: str = "cpu"):
        super().__init__()
        self.num_classes = num_classes
        self.version = version
        self.device = device
        # placeholder attribute; real code would load a YOLO model
        self.model = None

    def forward(self, x):
        if self.model is None:
            raise RuntimeError("YOLO model has not been initialized")
        return self.model(x)


def create_yolo(num_classes: int, version: str = "v3", device: str = "cpu") -> YOLOWrapper:
    wrapper = YOLOWrapper(num_classes, version, device)
    # TODO: instantiate actual YOLO network and assign to wrapper.model
    return wrapper
