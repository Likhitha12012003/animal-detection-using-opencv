import cv2
from ultralytics import YOLO
import numpy as np
import gradio as gr

# Load YOLO model (assuming yolov9c.pt is a valid custom model)
model = YOLO("yolov8m.pt")

# List of animal classes in COCO dataset
animal_classes = ["bird", "cat", "dog", "horse", "sheep", "cow", "elephant", "bear", "zebra", "giraffe",]

def predict_and_detect(chosen_model, img, classes=[], conf=0.5, rectangle_thickness=2, text_thickness=1):
    # Perform prediction
    results = chosen_model.predict(img, conf=conf)

    # Draw bounding boxes
    for result in results:
        for box in result.boxes:
            if chosen_model.names[int(box.cls[0])] in classes:
                cv2.rectangle(img, (int(box.xyxy[0][0]), int(box.xyxy[0][1])),
                              (int(box.xyxy[0][2]), int(box.xyxy[0][3])), (255, 0, 0), rectangle_thickness)
                cv2.putText(img, f"{chosen_model.names[int(box.cls[0])]}",
                            (int(box.xyxy[0][0]), int(box.xyxy[0][1]) - 10),
                            cv2.FONT_HERSHEY_PLAIN, 2, (255, 0, 0), text_thickness)
    return img

def detect_animals_with_gradio(image):
    # Convert Gradio image input (PIL format) to OpenCV format
    image = cv2.cvtColor(np.array(image), cv2.COLOR_RGB2BGR)

    # Detect animals
    result_img = predict_and_detect(model, image, classes=animal_classes, conf=0.5)

    # Convert result back to RGB for display
    result_img = cv2.cvtColor(result_img, cv2.COLOR_BGR2RGB)
    return result_img

# Create Gradio interface
iface = gr.Interface(
    fn=detect_animals_with_gradio,
    inputs=gr.Image(type="pil", label="Upload an Image"),
    outputs="image",
    title="Animal Recognization",
    description="Upload an image and the model will detect and mark animals in the image."
)2

# Launch the Gradio app
iface.launch()
