import base64
import io
from PIL import Image, ImageDraw

def draw_detections_on_image_and_save(input_img: Image.Image, detections: list):
    image = input_img.copy()
    draw = ImageDraw.Draw(image)

    for det in detections:
        x1, y1, x2, y2 = det["bbox"]
        draw.rectangle([x1, y1, x2, y2], outline='blue', width=4)

    if image.mode in ('RGBA', 'P'):
        image = image.convert('RGB')

    buffer = io.BytesIO()
    image.save(buffer, format="JPEG")
    base64_img = base64.b64encode(buffer.getvalue()).decode("utf-8")
    return base64_img
