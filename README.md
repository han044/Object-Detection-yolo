# Object-Detection-yolo

This project was done on a Windows machine with Ultralyrics/yolov5.

## Yolo Model
- Clone the yolo model from [Model Link](https://github.com/ultralytics/yolov5) to the `part_1` directory.
- Use the `.pt` file inside your yolo clone and base directory (`Object-Detection-yolo`).
- Copy `detect.py` and `hubconf.py` to the base directory.
- You may need to modify the model in `webcam.py` and `webapp.py` according to your model (unless using the above model).

## Virtual Environment
- Install virtual environment with `virtualenv venv`.
- Use `venv\scripts\activate.bat`.
- Use `pip install -r requirements`.

## Model Working
- The model creates a bounding box around real-world objects (as per your model).

For further queries and collaboration requests, please visit my [LinkedIn profile](https://www.linkedin.com/in/rohan-raj-885764232/).
