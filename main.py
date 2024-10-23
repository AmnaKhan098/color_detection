import cv2
import numpy as np
import streamlit as st
from PIL import Image

def limits_yellow():
    lower = np.array([20, 103, 100], dtype=np.uint8)
    upper = np.array([30, 255, 255], dtype=np.uint8)
    return lower, upper

def limits_red():
    lower1 = np.array([0, 100, 100], dtype=np.uint8)
    upper1 = np.array([10, 255, 255], dtype=np.uint8)
    lower2 = np.array([170, 100, 100], dtype=np.uint8)
    upper2 = np.array([180, 255, 255], dtype=np.uint8)
    return (lower1, upper1), (lower2, upper2)

def limits_green():
    lower = np.array([40, 50, 100], dtype=np.uint8)
    upper = np.array([80, 200, 255], dtype=np.uint8)
    return lower, upper

def limits_blue():
    lower = np.array([100, 100, 100], dtype=np.uint8)
    upper = np.array([140, 255, 255], dtype=np.uint8)
    return lower, upper

def main():
    st.title("Webcam Color Detection")
    run = st.checkbox('Run the webcam')

    # Create a placeholder for the image
    image_placeholder = st.empty()

    # Start video capture
    cam = cv2.VideoCapture(0)

    while run:
        ret, frames = cam.read()
        if not ret:
            st.error("Failed to grab frame")
            break

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

        # Create a dictionary of masks
        masks = {'Yellow': mask_yellow, 'Red': mask_red, 'Green': mask_green, 'Blue': mask_blue}
        new_frame = frames.copy()

        for color, mask in masks.items():
            # Find contours for the mask
            contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

            for contour in contours:
                if cv2.contourArea(contour) > 500:  # Only consider large enough contours
                    x, y, w, h = cv2.boundingRect(contour)
                    new_frame = cv2.rectangle(new_frame, (x, y), (x + w, y + h), (0, 255, 0), 2)
                    cv2.putText(new_frame, color, (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)

        # Convert frame to image
        img = Image.fromarray(new_frame)
        image_placeholder.image(img, caption="Detected Colors", use_column_width=True)

    cam.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()
