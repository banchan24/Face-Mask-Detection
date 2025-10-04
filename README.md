# Face-Mask-Detection
A Python + OpenCV project that detects whether a person is wearing a mask in images.

Project Overview

This project is a mask detection tool that uses OpenCV’s Haar Cascade classifiers to detect human faces and determine if a mask is worn or not.

The program can:

✅ Process single images or entire folders of photos (e.g., Sample Pictures)

✅ Detect faces and highlight them with a green rectangle

✅ Decide whether the mouth is visible:

MASK ON (green text) → no mouth detected in lower half of face

NO MASK (red text) → mouth detected → likely no mask

✅ Save annotated copies of all processed images to an outputs folder

✅ Supports common image types (.jpg / .jpeg / .png / .bmp / .jfif)

This tool is useful for basic computer-vision learning projects and demonstrates image pre-processing, face detection, and annotation.
-------------------------------------------------------------------------------------------------------------
How the Program Works

1. Load Cascades:
Loads pre-trained Haar cascade XML files for face detection and mouth detection.

2. Scan Input Folder:
Collects all supported image files from Sample Pictures.

3. Face & Mouth Detection:
For each image:

- Converts to grayscale (improves detection speed and accuracy)

- Detects faces → draws green rectangles around them

- Checks for any mouth region in the lower half of each detected face

- If no mouth detected → assumes mask is on → labels MASK ON
  Else → labels NO MASK

4. Save Results:

Annotated images are saved into the outputs folder.
.jfif files are automatically converted to .jpg because OpenCV cannot save .jfif output.

5. Display Progress:

Shows progress for each image in the terminal.

-------------------------------------------------------------------------------------------------------------
Quick Start:
1. Clone the Repository:

git clone https://github.com/<your-username>/mask-detector.git
cd mask-detector

2. Create Virtual Environment
python -m venv venv
venv\Scripts\activate      # Windows
# or: source venv/bin/activate  # macOS/Linux

You must have python version 3.10

- Install opencv and numpy

- Open the Start Menu, type cmd, right-click Command Prompt, and choose Run as administrator. Then try to install opencv-python numpy
<img width="489" height="73" alt="image" src="https://github.com/user-attachments/assets/c830e1e6-345f-4458-ab7d-6e98da56979a" />

- Then activate Virtual environment after direction the face-mask-folder

cd C:\Users\Users_name\Face_Mask_Detection
venv\Scripts\activate

- After activation, your prompt should change to something like this:

(venv) C:\Users\Users_name\Face_Mask_Detection>

to check installed packages: (you should see opencv-python, numpy, and others)

<img width="157" height="42" alt="image" src="https://github.com/user-attachments/assets/bc85e9da-26eb-4c70-9c36-e81603a60d03" />

3. Run

<img width="291" height="39" alt="image" src="https://github.com/user-attachments/assets/460294a3-f64d-47ee-9d55-f89e36d9a463" />

4. Check output:

- Processed images will appear in the output/ folder with bounding boxes and labels.
