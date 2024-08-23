from .nodes.kohya import *
from .nodes.auto_portrait_crop import *

NODE_CLASS_MAPPINGS = { 
    "Brookreator Kohya": Kohya,
    "Brookreator Auto Portrait Crop": AutoPortraitCrop
    }
    
print("\033[34mComfyUI Tutorial Nodes: \033[92mLoaded\033[0m")

WEB_DIRECTORY = "./js"

__all__ = ['NODE_CLASS_MAPPINGS', 'WEB_DIRECTORY']