import argparse
import cv2
import numpy as np
import os
import shutil
from pathlib import Path

def cleanup_frame_folder(folder):
    if not os.path.exists(folder):
        os.makedirs(folder)
    else:
        for filename in os.listdir(folder):
            file_path = os.path.join(folder, filename)
            try:
                if os.path.isfile(file_path) or os.path.islink(file_path):
                    os.unlink(file_path)
                elif os.path.isdir(file_path):
                    shutil.rmtree(file_path)
            except Exception as e:
                print(f'Failed to delete {file_path}. Reason: {e}')

def region_of_interest(img, vertices):
    mask = np.zeros_like(img)
    cv2.fillPoly(mask, vertices, 255)
    masked_image = cv2.bitwise_and(img, mask)
    return masked_image

def draw_lines(img, lines, color=(255, 0, 0), thickness=5):
    line_img = np.zeros_like(img)
    if lines is not None:
        for line in lines:
            for x1, y1, x2, y2 in line:
                cv2.line(line_img, (x1, y1), (x2, y2), color, thickness)
    img = cv2.addWeighted(img, 0.8, line_img, 1.0, 0.0)
    return img

def get_roi_vertices(width, height, video_in):
    if video_in.endswith("3686.mp4"):
        return [(0, height + 200 + 20), (11 * width / 24 - 140, height / 3), (7 * width / 12 + 200 + 40, height)]
    elif video_in.endswith("3674.mp4"):
        return [(0, height), (11 * width / 24 + 200, height / 3 - 250), (7 * width / 12, height + 70 + 60)]
    else:
        # Default ROI vertices
        return [(50, height), (width / 2, height / 2), (width - 50, height)]

def process_frame(image, video_in):
    gray_image = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
    cannyed_image = cv2.Canny(gray_image, 100, 200)
    roi_vertices = get_roi_vertices(image.shape[1], image.shape[0], video_in)
    cropped_image = region_of_interest(cannyed_image, np.array([roi_vertices], np.int32))
    lines = cv2.HoughLinesP(cropped_image, rho=6, theta=np.pi / 60, threshold=160, lines=np.array([]), minLineLength=40, maxLineGap=25)
    return draw_lines(image, lines) if lines is not None else image

def main(video_in, frame_folder, fps):
    cleanup_frame_folder(frame_folder)
    vid = cv2.VideoCapture(video_in)
    frame_index = 0

    while True:
        ret, frame = vid.read()
        if not ret:
            break

        processed_frame = process_frame(frame, video_in)
        cv2.imshow('Frame', processed_frame)
        cv2.imwrite(os.path.join(frame_folder, f"frame_{frame_index:04d}.png"), processed_frame)
        frame_index += 1

        if cv2.waitKey(25) & 0xFF == ord('q'):
            break

    vid.release()
    cv2.destroyAllWindows()

    # Video reconstruction from frames
    frame_files = [f for f in sorted(os.listdir(frame_folder)) if f.startswith("frame_")]
    if frame_files:
        frame_path = os.path.join(frame_folder, frame_files[0])
        frame = cv2.imread(frame_path)
        height, width, layers = frame.shape
        size = (width, height)
        output_video = video_in.replace('.mp4', '_processed.mp4')
        out_vid = cv2.VideoWriter(output_video, cv2.VideoWriter_fourcc(*'mp4v'), fps, size)

        for f in frame_files:
            frame_path = os.path.join(frame_folder, f)
            frame = cv2.imread(frame_path)
            out_vid.write(frame)

        out_vid.release()
        print(f"Processed video saved as {output_video}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("-vin", "--video_in", required=True, help="Path to the video")
    parser.add_argument("-vout", "--video_out", required=True, help="Directory to store frames in")
    parser.add_argument("-
