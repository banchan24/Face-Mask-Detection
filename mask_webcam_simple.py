# -------------------------------------------------------------
#  Mask Detector – Batch mode for still images
#  Scans all supported image files in "Sample Pictures"
#  Detects faces & mouths → labels MASK ON / NO MASK
#  Saves annotated copies in "outputs" folder
# -------------------------------------------------------------

import cv2                 # OpenCV library for computer vision
import os                  # For file and folder path operations
import sys                 # Lets us exit the script if something goes wrong
from glob import glob      # For finding files that match a pattern (*.jpg, etc.)

# -------------------------------------------------------------
# 1. Paths to the trained Haar cascade XML models
# -------------------------------------------------------------
FACE_XML  = r"C:\Users\User_name\Face_Mask_Detection\haarcascade_frontalface_default.xml"
MOUTH_XML = r"C:\Users\User_name\Face_Mask_Detection\Mouth.xml"  # use haarcascade_mcs_mouth.xml if that’s your file

# -------------------------------------------------------------
# 2. Input & output folders
# -------------------------------------------------------------
INPUT_DIR  = r"C:\Users\User_name\Face_Mask_Detection\Sample Pictures"  # folder with your sample photos
OUTPUT_DIR = r"C:\Users\User_name\Face_Mask_Detection\outputs"         # folder to save processed results

# -------------------------------------------------------------
# 3. Verify that the input folder exists
# -------------------------------------------------------------
if not os.path.isdir(INPUT_DIR):                      # if folder missing…
    print("Folder does not exist:", INPUT_DIR)        # print helpful info
    print("Contents of parent:", os.listdir(os.path.dirname(INPUT_DIR)))
    sys.exit(1)                                       # stop the program

# create the output folder if it’s not there already
os.makedirs(OUTPUT_DIR, exist_ok=True)

# -------------------------------------------------------------
# 4. Load the Haar cascades for face and mouth detection
# -------------------------------------------------------------
face_cascade  = cv2.CascadeClassifier(FACE_XML)
mouth_cascade = cv2.CascadeClassifier(MOUTH_XML)

# if any cascade fails to load, stop right away
if face_cascade.empty():
    print("ERROR: could not load face cascade:", FACE_XML)
    sys.exit(1)
if mouth_cascade.empty():
    print("ERROR: could not load mouth cascade:", MOUTH_XML)
    sys.exit(1)

# -------------------------------------------------------------
# 5. Gather all supported image files from Sample Pictures
#    (added JFIF so images.jfif will be included)
# -------------------------------------------------------------
exts = ("*.jpg","*.jpeg","*.png","*.bmp","*.jfif",
        "*.JPG","*.JPEG","*.PNG","*.BMP","*.JFIF")

images = []                                            # list to hold all found file paths
for e in exts:                                         # search for each extension pattern
    images.extend(glob(os.path.join(INPUT_DIR, e)))

# if no image found, inform and quit
if not images:
    print("No images found in", INPUT_DIR)
    sys.exit(0)

# -------------------------------------------------------------
# 6. Set font for labeling text on each image
# -------------------------------------------------------------
font = cv2.FONT_HERSHEY_SIMPLEX

# -------------------------------------------------------------
# 7. Process each image
# -------------------------------------------------------------
for idx, img_path in enumerate(images, 1):             # enumerate gives file number
    img = cv2.imread(img_path)                         # read the image
    if img is None:                                    # if OpenCV fails to read
        print(f"[{idx}/{len(images)}] Skipped (could not read): {img_path}")
        continue

    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)       # convert to grayscale for cascades

    faces = face_cascade.detectMultiScale(gray, 1.1, 4)   # detect faces
    label_for_image = "NO FACE"                           # default if none found

    for (x, y, w, h) in faces:                             # for every face found
        cv2.rectangle(img, (x, y), (x+w, y+h), (0,255,0), 2)  # draw green box

        # region of interest (only the face area) to look for mouth
        roi_gray = gray[y:y+h, x:x+w]

        # detect mouths in the face region
        mouth_rects = mouth_cascade.detectMultiScale(roi_gray, 1.5, 11)

        # assume wearing a mask unless we see a mouth in lower half
        wearing_mask = True
        for (mx, my, mw, mh) in mouth_rects:
            if my > h * 0.4:        # mouth usually appears in bottom half of the face
                wearing_mask = False
                break

        # decide final label and choose color
        label_for_image = "MASK ON" if wearing_mask else "NO MASK"
        color = (0,255,0) if wearing_mask else (0,0,255)

        # put label above the rectangle
        cv2.putText(img, label_for_image, (x, y-10), font, 0.8, color, 2, cv2.LINE_AA)

    # ---------------------------------------------------------
    # 8. Save the annotated image to outputs folder
    #    - rename .jfif/.jpe to .jpg because cv2.imwrite can’t save jfif
    # ---------------------------------------------------------
    base = os.path.basename(img_path)                  # file name only
    root, ext = os.path.splitext(base)                 # split into name + extension
    if ext.lower() in (".jfif", ".jpe"):               # unsupported for writing
        base = root + ".jpg"                            # save as jpg
    out_path = os.path.join(OUTPUT_DIR, base)

    cv2.imwrite(out_path, img)                          # save the processed image
    print(f"[{idx}/{len(images)}] {os.path.basename(img_path)} -> "
          f"{label_for_image} (saved to {out_path})")

# -------------------------------------------------------------
# 9. Done
# -------------------------------------------------------------
print("Done. Check:", OUTPUT_DIR)
