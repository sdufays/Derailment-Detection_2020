import argparse
import cv2
import numpy as np
import os

def mask(img, points):
    mask = np.zeros_like(img)
    cv2.fillPoly(mask, points, 255)
    return cv2.bitwise_and(img, mask)

def draw_lines(img, lines, color=(255, 0, 0), thickness=5):
    line_img = np.zeros_like(img)
    if lines is not None:
        for line in lines:
            for x1, y1, x2, y2 in line:
                cv2.line(line_img, (x1, y1), (x2, y2), color, thickness)
    return cv2.addWeighted(img, 0.8, line_img, 1, 0)

def process_frame(image, video_in):
    height, width = image.shape[:2]
    region_of_interest_points = get_roi_vertices(width, height, video_in)
    gray_image = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
    canny_image = cv2.Canny(gray_image, 100, 200)
    cropped_image = mask(canny_image, np.array([region_of_interest_points], np.int32))

    lines = cv2.HoughLinesP(
        cropped_image, rho=6, theta=np.pi / 60, threshold=160,
        lines=np.array([]), minLineLength=40, maxLineGap=25
    )

    if lines is None:
        circle_color = (0, 255, 0) if video_in.endswith("3674.mp4") else (0, 255, 255)
        return cv2.circle(image, (int(width/6), 3*int(height/4)), 50, circle_color, -1)
    else:
        return draw_lines(image, lines)

def get_roi_vertices(width, height, video_in):
    if video_in.endswith("3686.mp4"):
        return [(0, height + 200 + 20), (11 * width / 24 - 140, height / 3), (7 * width / 12 + 200 + 40, height)]
    elif video_in.endswith("3674.mp4"):
        return [(0, height), (11 * width / 24 + 200, height / 3 - 250), (7 * width / 12, height + 70 + 60)]
    else:
        return [(50, height), (width / 2, height / 2), (width - 50, height)]

def main(video_in):
    vid = cv2.VideoCapture(video_in)
    while True:
        ret, frame = vid.read()
        if not ret:
            print("Failed to capture frame or end of video reached.")
            break

        processed_frame = process_frame(frame, video_in)
        cv2.imshow('Processed Frame', processed_frame)

        if cv2.waitKey(25) & 0xFF == ord('q'):
            break

    vid.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("-vin", "--video_in", required=True, help="Path to the video file")
    args = parser.parse_args()
    
    if not os.path.isfile(args.video_in):
        print(f"The file {args.video_in} does not exist.")
        exit()

    main(args.video_in)
