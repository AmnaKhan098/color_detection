import cv2
from PIL import Image
import numpy as np

def limits_yellow():
    # Define lower and upper HSV limits for yellow
    lower = np.array([20, 103, 100], dtype=np.uint8)   # Adjust these values for yellow
    upper = np.array([30, 255, 255], dtype=np.uint8)
    return lower, upper

def limits_red():
    # Define lower and upper HSV limits for red
    lower1 = np.array([0, 100, 100], dtype=np.uint8)   # Adjust these values for red
    upper1 = np.array([10, 255, 255], dtype=np.uint8)
    lower2 = np.array([170, 100, 100], dtype=np.uint8) # Adjust these values for red
    upper2 = np.array([180, 255, 255], dtype=np.uint8)
    return (lower1, upper1), (lower2, upper2)

def limits_green():
    # Define lower and upper HSV limits for green
    lower = np.array([40, 50, 100], dtype=np.uint8)   # Adjust these values for green
    upper = np.array([80, 200, 255], dtype=np.uint8)
    return lower, upper

def limits_blue():
    # Define lower and upper HSV limits for blue
    lower = np.array([100, 100, 100], dtype=np.uint8)  # Adjust these values for blue
    upper = np.array([140, 255, 255], dtype=np.uint8)
    return lower, upper

# --- WEB CAM ---
cam = cv2.VideoCapture(0)

while True:
    _, frames = cam.read()
    
    # Convert to HSV color space
    hsv_image = cv2.cvtColor(frames, cv2.COLOR_BGR2HSV)

    # Get lower and upper limits for each color
    l_limit_yellow, u_limit_yellow = limits_yellow()
    (l_limit_red1, u_limit_red1), (l_limit_red2, u_limit_red2) = limits_red()
    l_limit_green, u_limit_green = limits_green()
    l_limit_blue, u_limit_blue = limits_blue()

    # Create masks for each color
    mask_yellow = cv2.inRange(hsv_image, l_limit_yellow, u_limit_yellow)
    mask_red1 = cv2.inRange(hsv_image, l_limit_red1, u_limit_red1)
    mask_red2 = cv2.inRange(hsv_image, l_limit_red2, u_limit_red2)
    mask_green = cv2.inRange(hsv_image, l_limit_green, u_limit_green)
    mask_blue = cv2.inRange(hsv_image, l_limit_blue, u_limit_blue)

    # Combine the red masks
    mask_red = cv2.bitwise_or(mask_red1, mask_red2)

    # Convert masks to images and get bounding boxes
    masks = {'Yellow': mask_yellow, 'Red': mask_red, 'Green': mask_green, 'Blue': mask_blue}
    new_frame = frames.copy()

    for color, mask in masks.items():
        mask_array = Image.fromarray(mask)
        boundary = mask_array.getbbox()
        
        if boundary is not None:
            x1, y1, x2, y2 = boundary
            
            # Draw rectangle around the detected color
            new_frame = cv2.rectangle(new_frame, (x1, y1), (x2, y2), (0, 255, 0), 5)
            
            # Add color name on top of the boundary box
            cv2.putText(new_frame, color, (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)

    # Display the result
    cv2.imshow('Camera', new_frame)
    
    # Exit loop if 'q' is pressed
    if cv2.waitKey(1) == ord('q'):
        break 

# Release the webcam and close windows
cam.release()
cv2.destroyAllWindows()

      
