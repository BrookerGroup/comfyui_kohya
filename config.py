MODEL_FACE = "weights/yolov8/yolov8l-face.pt"
MODEL_SEG = "weights/yolov8/yolov8x-seg.pt"
MODEL_FACE_LIB = "weights/facelib/parsing_parsenet.pth"
MODEL_CODE_FORMER = "weights/CodeFormer/codeformer.pth"
MODEL_REALESRGAN = "weights/realESRGaN/RealESRGAN_x4plus.pth"
# MODEL_FACE_DETECTION = "weights/facelib/yolov5n-face.pth" 
MODEL_FACE_DETECTION = "weights/facelib/detection_Resnet50_Final.pth"

MASK_SCALE = 2 # to increas or decrease the person mask size 
MAX_EXPAND_RATIO = 0.1 #0.1 # lower this value to increase the size of the face, higher this value to decrease the size of the face
MIN_EXPAND_RATIO = 0.02 #0.02 # lower this value to increase the size of the face, higher this value to decrease the size of the face
SCALE_THRESHOLD = 0.8 # threshold to check if the face is too small or too big
# THRESHOLD_FACE = 80 # To check if face is complete
WIDTH, HEIGHT = 768, 768
ACCEPTED_FILE_EXTENSIONS = ('.png', '.jpg', '.jpeg')
CLASS_ID = 0
MAX_CONFIDENCE = 0.2
CLASS_NAMES = [
    "person", "bicycle", "car", "motorcycle", "airplane",
    "bus", "train", "truck", "boat", "traffic light",
    "fire hydrant", "stop sign", "parking meter", "bench", "bird",
    "cat", "dog", "horse", "sheep", "cow",
    "elephant", "bear", "zebra", "giraffe", "backpack",
    "umbrella", "handbag", "tie", "suitcase", "frisbee",
    "skis", "snowboard", "sports ball", "kite", "baseball bat",
    "baseball glove", "skateboard", "surfboard", "tennis racket", "bottle",
    "wine glass", "cup", "fork", "knife", "spoon",
    "bowl", "banana", "apple", "sandwich", "orange",
    "broccoli", "carrot", "hot dog", "pizza", "donut",
    "cake", "chair", "couch", "potted plant", "bed",
    "dining table", "toilet", "tv", "laptop", "mouse",
    "remote", "keyboard", "cell phone", "microwave", "oven",
    "toaster", "sink", "refrigerator", "book", "clock",
    "vase", "scissors", "teddy bear"
]
