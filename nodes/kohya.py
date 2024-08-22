class Kohya:
    CATEGORY = "kohya"
    @classmethod

    def INPUT_TYPES(s):
        return { "required":  { "images": ("IMAGE",), } }


    RETURN_TYPES = ("IMAGE",)

    FUNCTION = "choose_image"