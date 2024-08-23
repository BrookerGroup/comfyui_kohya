import torch
import time
from server import PromptServer
import folder_paths


# from ultralytics import YOLO
# from PIL import Image
# import os
# from util import CodeFormerRestorer
# from config import *
# from scipy.ndimage import binary_dilation
# import cv2

class AutoPortraitCrop:
    CATEGORY = "kohya"
    @classmethod


    @classmethod    
    def INPUT_TYPES(s):
        return { "required": {  "images": ("IMAGE", {"tooltip": "The images to select from."}), 
                                 "enabled_face_restore": ([True,False], {}),
                                 "remove_background": ([True,False], {}),
                                 "upscale":("INT", {"default": 1, "min": 1, "max": 4, "step": 1, "tooltip": "Upscale factor."}),
                                 "fidelity_weight":("FLOAT", {"default": 0.1, "min": 0.1, "max": 0.5, "step": 0.1, "tooltip": "Fidelity weight."}),
                                 "get_mask": ([True,False], {}),
                                 "handle_for_fail_crop": (["skip","remove","keep"], {}),
                             } }

    RETURN_TYPES = ("IMAGE",)
    RETURN_NAMES = ("cropped_images","complete_amount",)
    FUNCTION = "crop_image"

    def crop_image(self, images,enabled_face_restore,remove_background,upscale,fidelity_weight,get_mask,handle_for_fail_crop):
  
 
        # TODO: implement auto crop
        cropped_images = images
        complete_amount = len(cropped_images)
        # self.crop_images(images)
        return (cropped_images,complete_amount,)
    


 


    def crop_faces(images, remove_bg = True, upscale = 1, fidelity_weight=0.5, get_mask = False):
        pass
        # device = "cuda" if torch.cuda.is_available() else "cpu"
        # model_face = YOLO(MODEL_FACE).to(device)
        # model_seg = YOLO(MODEL_SEG).to(device)
        # model_face_restorer = CodeFormerRestorer(
        #                                     # bg_upsampler = "realesrgan", # uncomment if you want to use realesrgan for background upsample
        #                                     face_upsample = True, 
        #                                     bg_tile = 400, 
        #                                     detection_model = "retinaface_resnet50",#"YOLOv5n", #'retinaface_resnet50',  # Option: YOLOv5n (dont't forget to change in config.py)
        #                                     upscale = upscale, # upscale value: 1 - 4
        #                                     )

        # # incomplete_face = "" # was used to test if face is complete
        # t0 = time.time()
        # for image in images:
            

        #     # Resize and Fill image
        #     output_size = (WIDTH, HEIGHT)
        #     original_width, original_height = image.size
        #     aspect_ratio = min(output_size[0] / original_width, output_size[1] / original_height)
        #     new_width = int(original_width * aspect_ratio)
        #     new_height = int(original_height * aspect_ratio)
        #     resized_image = image.resize((new_width, new_height), Image.NEAREST)
        #     new_image = Image.new("RGB", output_size, (255, 255, 255))  # White background, change if needed
        #     x = (output_size[0] - new_width) // 2
        #     y = (output_size[1] - new_height) // 2
        #     new_image.paste(resized_image, (x, y))
        #     image = new_image.copy()

            

        #     # Process Instance Segmentation model 
        #     results = model_seg(image)  
        #     boxes = results[0].boxes
        #     classes = boxes.cls.cpu().numpy() 
        #     confs = boxes.conf.cpu().numpy()
        #     masks = results[0].masks.data.cpu().numpy() if results[0].masks is not None else None
        #     person_idxs = np.where(classes == CLASS_ID)[0]

        #     # Use only person class, ignore the rest. 
        #     if len(person_idxs) > 0:
        #         # Filter out low confidence detections
        #         filtered_idxs = [idx for idx in person_idxs if confs[idx] > MAX_CONFIDENCE]

        #         if len(filtered_idxs) > 0:
        #                 # Sort the filtered_idxs by confidence
        #                 sorted_idxs = np.argsort(confs[filtered_idxs])[::-1]
                        
        #                 # Initialize to the highest confidence index
        #                 next_best_idx = 0 

        #                 # Loop through the sorted_idxs until a face is detected
        #                 while next_best_idx < len(sorted_idxs):
        #                     max_conf_idx = filtered_idxs[sorted_idxs[next_best_idx]]
        #                     next_best_idx += 1

        #                     # Resize the mask to the image size
        #                     max_conf_mask = masks[max_conf_idx]
        #                     mask_resized = Image.fromarray(max_conf_mask).resize(image.size, resample=Image.NEAREST)
        #                     mask_bool = np.array(mask_resized) > 0

        #                     # Dilate the mask
        #                     dilated_mask = binary_dilation(mask_bool, iterations=MASK_SCALE)
        #                     white_bg = np.ones_like(np.array(image)) * 255
        #                     result_image = np.where(dilated_mask[..., None], np.array(image), white_bg)
        #                     result_image = Image.fromarray(result_image).convert("RGB")

        #                     # Process Face Detection model
        #                     results = model_face(result_image)

        #                     if len(results[0].boxes.xyxy) > 0:
        #                         if get_mask == True: # generate mask
        #                             dilated_mask = Image.fromarray(dilated_mask).convert("RGB")
        #                             dilated_mask.resize((WIDTH, HEIGHT), resample=Image.NEAREST).save(f"output/mask_{file_name}")
        #                             result_image.resize((WIDTH, HEIGHT), resample=Image.NEAREST).save(f"output/person_{file_name}")
        #                         break

        #                     # If after all iterations, no face is detected, next_best_idx will be len(sorted_idxs)
        #                     if next_best_idx == len(sorted_idxs):
        #                         print(f"No face detected after checking all candidates in {file_name}")
        #                         # result_image.save(f"output/{incomplete_face}{file_name}")


        #                 if len(results[0].boxes.xyxy) > 0:
        #                     x_min, y_min, x_max, y_max = results[0].boxes.xyxy[0].cpu().numpy()
        #                     # print(x_min, y_min, x_max, y_max, image.height, image.width)

        #                     # Calculate the size of the detected bounding box
        #                     box_width = x_max - x_min
        #                     box_height = y_max - y_min
        #                     box_area = box_width * box_height
        #                     image_area = image.width * image.height
        #                     scale = box_area * 100/ image_area
                            
        #                     # Calculate the amount to expand the bounding box based on the scale
        #                     expand_by_ratio = MAX_EXPAND_RATIO - ((SCALE_THRESHOLD - scale) / SCALE_THRESHOLD) * (MAX_EXPAND_RATIO - MIN_EXPAND_RATIO)
        #                     expand_by_ratio = min(max(expand_by_ratio, MIN_EXPAND_RATIO), MAX_EXPAND_RATIO)
        #                     # print(scale, expand_by_ratio, file_name)

        #                     # Calculate expansion pixels based on image resolution
        #                     expand_x_pixels = int(expand_by_ratio * image.size[0])
        #                     expand_y_pixels = int(expand_by_ratio * image.size[1])

        #                     # Apply the calculated expansion to the bounding box
        #                     x_min = max(x_min - expand_x_pixels, 0)
        #                     y_min = max(y_min - expand_y_pixels, 0)
        #                     x_max = min(x_max + expand_x_pixels, image.size[0])
        #                     y_max = min(y_max + expand_y_pixels, image.size[1])

        #                     # Calculate the size of the bounding box again after expansion
        #                     box_width = x_max - x_min
        #                     box_height = y_max - y_min

        #                     # Calculate the amount to expand the bounding box to make it square
        #                     square_box_size = max(box_width, box_height)
        #                     expand_x_pixels = (square_box_size - box_width) / 2  # for the x dimension
        #                     expand_y_pixels = (square_box_size - box_height) / 2  # for the y dimension

        #                     # Apply the calculated expansion to the bounding box
        #                     expanded_x_min = int(max(x_min - expand_x_pixels, 0))
        #                     expanded_y_min = int(max(y_min - expand_y_pixels, 0))
        #                     expanded_x_max = int(min(x_max + expand_x_pixels, image.size[0]))
        #                     expanded_y_max = int(min(y_max + expand_y_pixels, image.size[1]))

        #                     # If expanding the box goes out of image bounds, shift it back
        #                     if expanded_x_min == 0:
        #                         expanded_x_max = min(expanded_x_max + (expand_x_pixels - x_min), image.size[0])
        #                     if expanded_y_min == 0:
        #                         expanded_y_max = min(expanded_y_max + (expand_y_pixels - y_min), image.size[1])
        #                     if expanded_x_max == image.size[0]:
        #                         expanded_x_min = max(expanded_x_min - (x_max + expand_x_pixels - image.size[0]), 0)
        #                     if expanded_y_max == image.size[1]:
        #                         expanded_y_min = max(expanded_y_min - (y_max + expand_y_pixels - image.size[1]), 0)

        #                     # Ensure the bounding box is still square after shifting
        #                     # This step ensures that the box remains square if it was shifted due to being at the edge
        #                     final_box_width = expanded_x_max - expanded_x_min
        #                     final_box_height = expanded_y_max - expanded_y_min
        #                     if final_box_width > final_box_height:
        #                         diff = final_box_width - final_box_height
        #                         expanded_y_min = max(expanded_y_min - diff // 2, 0)
        #                         expanded_y_max = min(expanded_y_max + diff // 2, image.size[1])
        #                     elif final_box_height > final_box_width:
        #                         diff = final_box_height - final_box_width
        #                         expanded_x_min = max(expanded_x_min - diff // 2, 0)
        #                         expanded_x_max = min(expanded_x_max + diff // 2, image.size[0])

                            
        #                     if remove_bg: # only display the person without background
        #                         cropped_image = Image.fromarray(np.array(result_image)).crop((expanded_x_min, expanded_y_min, expanded_x_max, expanded_y_max))
        #                     else:
        #                         cropped_image = Image.fromarray(np.array(image)).crop((expanded_x_min, expanded_y_min, expanded_x_max, expanded_y_max))

        #                     cropped_aspect = cropped_image.width / cropped_image.height
        #                     target_aspect = WIDTH / HEIGHT

        #                     if cropped_aspect > target_aspect:
        #                         # Cropped image is wider than the target shape
        #                         new_height = HEIGHT
        #                         new_width = int(new_height * cropped_aspect)
        #                     else:
        #                         # Cropped image is taller than the target shape
        #                         new_width = WIDTH
        #                         new_height = int(new_width / cropped_aspect)

        #                     # Resize the image to the new dimensions
        #                     resized_image = cropped_image.resize((new_width, new_height), resample=Image.NEAREST)
        #                     # Create a new image with the desired size and fill color
        #                     final_image = Image.new("RGB", (WIDTH, HEIGHT), "white")
        #                     x = (WIDTH - resized_image.width) // 2
        #                     y = (HEIGHT - resized_image.height) // 2
        #                     final_image.paste(resized_image, (x, y))

        #                     # # Remove background, uncomment if you want to remove background with more control, read documentation of "rembg" for more info
        #                     # final_image = remove_background(np.array(final_image.convert("RGB")))
        #                     # final_image = Image.fromarray(final_image).convert("RGB")

        #                     # Enhance the face
        #                     _ , restored_img = model_face_restorer.enhance(
        #                                                         np.array(final_image.convert("RGB")), 
        #                                                         fidelity_weight=fidelity_weight,
        #                                                         has_aligned = False, # If true, restored_img will be None
        #                                                         only_center_face=False, 
        #                                                         )
                            
        #                     # restored_img = app.process(image, scale=Scale.X2, mode=Mode.STANDARD, delete_from_history=True, output_format=OutputFormat.PNG)
        #                     final_image.resize((512, 512), resample=Image.NEAREST).save(f"output/no_restro_{file_name}")
        #                     if restored_img is not None:
        #                         restored_img = restored_img[:, :, [2, 1, 0]]
        #                         final_image_en = Image.fromarray(restored_img)
        #                         print(f"Process time: {time.time() - t0} for {file_name}")
        #                         if remove_bg:
        #                             final_image_en.resize((512, 512), resample=Image.NEAREST).save(f"output/res_Face_noBG_{file_name}")
        #                         else:
        #                             # pass
        #                             final_image_en.resize((512, 512), resample=Image.NEAREST).save(f"output/3res_Face_BG_{file_name}")
        #                     else:
        #                         print(f"Image could not be enhanced in {file_name}")
        #                         # final_image.save(f"output/{incomplete_face}{file_name}")
        #                 else:
        #                     print(f"No face detected in {file_name}")
        #                     # result_image.resize((WIDTH, HEIGHT), resample=Image.NEAREST).save(f"output/{incomplete_face}{file_name}")
        #         else:
        #             print(f"Detected person confidence too low in {file_name}")
        #             # image.resize((WIDTH, HEIGHT), resample=Image.NEAREST).save(f"output/{incomplete_face}{file_name}")
        #     else:
        #         print(f"No person detected in {file_name}")
        #         # image.resize((WIDTH, HEIGHT), resample=Image.NEAREST).save(f"output/{incomplete_face}{file_name}")
        # print(f"Total_process_time: {time.time() - t0} for {len(os.listdir(folder_path))} images")