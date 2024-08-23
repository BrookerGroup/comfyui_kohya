import torch
import time
from server import PromptServer
import folder_paths
class Kohya:
    CATEGORY = "kohya"
    @classmethod


    @classmethod    
    def INPUT_TYPES(s):
        return { "required": {  "portrait_images": ("IMAGE", {"tooltip": "The images to select from."}), 
                                "pretrained_model_name": (folder_paths.get_filename_list("checkpoints"), {"tooltip": "The name of the checkpoint (model) to load."}),
                                "models_output_name": ("STRING", {"tooltip": "The name of the output from the model."}),
                                "models_output_path": ("STRING", {"tooltip": "The path to the output from the model."}),
                                "step": ("INT", {"tooltip": "The step of training."}),
                             } }

    RETURN_TYPES = ("IMAGE",)
    RETURN_NAMES = ("result",)
    FUNCTION = "choose_image"

    def choose_image(self, portrait_images, pretrained_model_name,models_output_name,models_output_path,step):
            print("images:",portrait_images)
            print("ckpt_name:",step)
             

            # TODO: implement kohya call

            return (portrait_images,)
    


 
