# animal-detection-using-opencv
Detect animal within a image using pre-trained model like yolo and identify and mark the animal in the image
To run the yolodetect.py file first you must download the yolo model like yolov8m.pt or any new version and must also have the coco.names within the directory
in the project terminal run 
-> pip install opencv-python
->Then install required packages like ultralytics and numpy then run the program
Gradio is used as an web interface where you upload any image and it will process and detect animal that are listed in coco.names file as only those animals are trained in this model.
