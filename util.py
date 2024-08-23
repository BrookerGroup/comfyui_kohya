
import os
import cv2
import torch
from torchvision.transforms.functional import normalize
from basicsr.utils import imwrite, img2tensor, tensor2img, registry
from facelib.utils import face_restoration_helper, misc as face_misc
from basicsr.utils.registry import ARCH_REGISTRY
from config import MODEL_CODE_FORMER, MODEL_REALESRGAN
from rembg import remove, new_session
from PIL import Image

model_name = "isnet-general-use"
session = new_session(model_name)
def remove_background(input_array, foreground_threshold = 0, background_threshold = 0, erode_size = 0):
    output_data = remove(input_array, session=session,
                              post_process_mask=True, 
                              bgcolor=(255, 255, 255, 255), 
                              alpha_matting=True, 
                              discard_threshold = 5.000000e-05,
                              alpha_matting_foreground_threshold= foreground_threshold,
                              alpha_matting_background_threshold=background_threshold, 
                              alpha_matting_erode_size=erode_size)
    return output_data

class CodeFormerRestorer:
    def __init__(self, bg_upsampler= None, face_upsample=True, bg_tile=400, detection_model='retinaface_resnet50', upscale=1):
        self.bg_tile = bg_tile
        self.upscale = upscale
        self.face_upsample = face_upsample

        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.net = self.setup_codeformer()
        self.bg_upsampler = None
        self.face_upsampler = None

        if bg_upsampler == 'realesrgan':
            self.bg_upsampler = self.set_realesrgan()
        else:
            self.bg_upsampler = None

        # set up face upsampler
        if face_upsample:
            if self.bg_upsampler is not None:
                self.face_upsampler = self.bg_upsampler
            else:
                self.face_upsampler = self.set_realesrgan()
        else:
            self.face_upsampler = None

        # set up FaceRestoreHelper
        self.face_helper = face_restoration_helper.FaceRestoreHelper(
            upscale,
            face_size=512,
            crop_ratio=(1, 1),
            det_model=detection_model,
            save_ext='png',
            use_parse=True,
            device=self.device)

    def setup_codeformer(self):
        print("Setting up CodeFormer")
        net = registry.ARCH_REGISTRY.get('CodeFormer')(
            dim_embd=512, codebook_size=1024, n_head=8, n_layers=9, 
            connect_list=['32', '64', '128', '256']
        ).to(self.device)

        ckpt_path = MODEL_CODE_FORMER
        checkpoint = torch.load(ckpt_path)['params_ema']
        net.load_state_dict(checkpoint)
        net.eval()
        return net

    def set_realesrgan(self):
        from basicsr.archs.rrdbnet_arch import RRDBNet
        from basicsr.utils.realesrgan_utils import RealESRGANer

        use_half = False  # By default, don't use half precision
        if torch.cuda.is_available():
            # Set to False for GPUs that don't support fp16
            no_half_gpu_list = ['1650', '1660']
            if not any(gpu in torch.cuda.get_device_name(0) for gpu in no_half_gpu_list):
                use_half = True

        # Initialize the RRDBNet model for RealESRGAN
        
        if MODEL_REALESRGAN == "weights/realESRGaN/RealESRGAN_x4plus.pth":
            model = RRDBNet(num_in_ch=3, num_out_ch=3, num_feat=64, num_block=23, num_grow_ch=32, scale=4)
            netscale = 4
        else:
            model = RRDBNet(num_in_ch=3, num_out_ch=3, num_feat=64, num_block=23, num_grow_ch=32, scale=2)  # Set the scale for upsampling
            netscale = 2
        # Initialize RealESRGANer with the model and other settings
        upsampler = RealESRGANer(
            scale=netscale,
            model_path=MODEL_REALESRGAN,
            model=model,
            tile=400,
            tile_pad=40,
            pre_pad=0,
            half=False  # Use half precision if applicable
        )

        if not torch.cuda.is_available():  # Check if running on CPU
            import warnings
            warnings.warn(
                'Running on CPU! The unoptimized RealESRGAN is slow on CPU. '
                'If you want to disable it, please remove `--bg_upsampler` and `--face_upsample` in command.',
                category=RuntimeWarning
            )

        return upsampler


    def enhance(self, img=None, fidelity_weight=0.5, has_aligned=False, 
                  only_center_face=False, draw_box=False,):
        
        img = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)
        
        w = fidelity_weight

    
        # start processing
        self.face_helper.clean_all()
        

        if has_aligned:
            img = cv2.resize(img, (512, 512), interpolation=cv2.INTER_LINEAR)
            self.face_helper.is_gray = face_misc.is_gray(img, threshold=10)
            self.face_helper.cropped_faces = [img]
        else:
            self.face_helper.read_image(img)
            num_det_faces = self.face_helper.get_face_landmarks_5(
                only_center_face=only_center_face, resize=640, eye_dist_threshold=5)
            print(f'\tdetect {num_det_faces} faces')
            self.face_helper.align_warp_face()

        # face restoration for each cropped face
        for idx, cropped_face in enumerate(self.face_helper.cropped_faces):
            # prepare data
            cropped_face_t = img2tensor(cropped_face / 255., bgr2rgb=True, float32=True)
            normalize(cropped_face_t, (0.5, 0.5, 0.5), (0.5, 0.5, 0.5), inplace=True)
            cropped_face_t = cropped_face_t.unsqueeze(0).to(self.device)

            try:
                with torch.no_grad():
                    output = self.net(cropped_face_t, w=w, adain=True)[0]
                    restored_face = tensor2img(output, rgb2bgr=True, min_max=(-1, 1))
                del output
                torch.cuda.empty_cache()
            except Exception as error:
                print(f'Failed inference for CodeFormer: {error}')
                restored_face = tensor2img(cropped_face_t, rgb2bgr=True, min_max=(-1, 1))

            restored_face = restored_face.astype('uint8')
            self.face_helper.add_restored_face(restored_face, cropped_face)

        # paste back and upsample background if needed
        # print(has_aligned)
        restored_img = None
        if not has_aligned:
            bg_img = self.bg_upsampler.enhance(img, outscale=self.upscale)[0] if self.bg_upsampler is not None else None
            self.face_helper.get_inverse_affine(None)
            if self.face_upsample and self.face_upsampler is not None:
                # print("Upsampling face") 
                restored_img = self.face_helper.paste_faces_to_input_image(upsample_img=bg_img, draw_box=draw_box, face_upsampler=self.face_upsampler)
            else:
                # print("Not upsampling face")
                restored_img = self.face_helper.paste_faces_to_input_image(upsample_img=bg_img, draw_box=draw_box)

        return restored_face, restored_img


# model_face_restorer = CodeFormerRestorer()

# img = cv2.imread("output/2_kimyoojung (61).jpg", cv2.IMREAD_COLOR)
# restored_face, restored_img = model_face_restorer.enhance(img, 
#                                         fidelity_weight=0.5, 
#                                         upscale=2, has_aligned=False, face_upsample = True,
#                                             only_center_face=False, detection_model='retinaface_resnet50',
#                                                 bg_tile=400)
# im_rgb = restored_face[:, :, [2, 1, 0]]
# final_image_en = Image.fromarray(im_rgb)
# final_image_en.save("output/2_kimyoojung_en.png")