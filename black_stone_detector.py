import cv2
import numpy as np

def find_only_black_stone(image_path, save_path="only_black_stone_result.jpg"):
    # Load image
    image = cv2.imread(image_path)
    if image is None:
        print("Error: Image could not be read. Please check the file name.")
        return

    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    blurred = cv2.medianBlur(gray, 7)
    output_image = image.copy()

    # Create ROI (Region of Interest) mask for the bottom 2/3 of the image
    mask_roi = np.zeros_like(gray)
    roi_y_start = image.shape[0] // 3
    mask_roi[roi_y_start:, :] = 255

    # Detect dark objects
    lower_bound_dark = 20
    upper_bound_dark = 80
    mask_dark = cv2.inRange(blurred, lower_bound_dark, upper_bound_dark)

    # Apply ROI mask
    mask_final = cv2.bitwise_and(mask_dark, mask_roi)

    # Morphological operations to clean noise
    kernel = np.ones((7, 7), np.uint8)
    mask_final = cv2.morphologyEx(mask_final, cv2.MORPH_OPEN, kernel, iterations=1)
    mask_final = cv2.morphologyEx(mask_final, cv2.MORPH_CLOSE, kernel, iterations=3)

    # Find contours
    contours, _ = cv2.findContours(mask_final, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    found_stones = []
    min_area = 10000
    min_solidity = 0.8

    for cnt in contours:
        area = cv2.contourArea(cnt)
        if area > min_area:
            hull = cv2.convexHull(cnt)
            hull_area = cv2.contourArea(hull)

            if hull_area > 0:
                solidity = float(area) / hull_area

                if solidity > min_solidity:
                    x, y, w, h = cv2.boundingRect(cnt)

                    stone_mask = np.zeros(gray.shape, dtype=np.uint8)
                    cv2.drawContours(stone_mask, [cnt], -1, 255, -1)
                    avg_darkness = cv2.mean(gray, mask=stone_mask)[0]

                    found_stones.append({
                        'coords': (x, y, w, h),
                        'darkness': avg_darkness,
                        'contour': cnt
                    })

    if not found_stones:
        print("No black stone matching the criteria was found.")
        return

    # Sort by darkness and identify the darkest stone
    found_stones.sort(key=lambda stone: stone['darkness'])
    darkest_stone = found_stones[0]

    # Draw bounding boxes and labels
    for stone in found_stones:
        x, y, w, h = stone['coords']
        darkness = stone['darkness']

        if stone == darkest_stone:
            color = (0, 0, 255) # Red for the darkest
            label = f"DARKEST ({darkness:.1f})"
            thickness = 4
        else:
            color = (255, 0, 0) # Blue for others
            label = f"Black Stone ({darkness:.1f})"
            thickness = 2

        cv2.rectangle(output_image, (x, y), (x + w, y + h), color, thickness)
        cv2.putText(output_image, label, (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.7, color, 2)

    cv2.imwrite(save_path, output_image)
    print(f"Process completed. Marked image saved as '{save_path}'.")


# Run the function
find_only_black_stone('stone.jpeg', save_path='only_black_stone.jpg')


"""
This script uses OpenCV to detect and analyze dark-colored stones within an image. 
It first isolates the lower two-thirds of the image (Region of Interest) to ignore 
background noise like the sky. It then applies grayscale conversion, blurring, and 
color thresholding to isolate dark objects. By filtering these contours based on 
their size (minimum area) and shape integrity (solidity), it distinguishes actual 
stones from random noise. Finally, the script calculates the average darkness of 
each detected stone, highlights the darkest one with a thick red bounding box, 
marks the others in blue, and saves the annotated image.

"""
