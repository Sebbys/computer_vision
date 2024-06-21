import cv2
from ultralytics import YOLO
import time

model = YOLO('yolov8n-pose.pt')
cap = cv2.VideoCapture(0)

total_keypoints = 0
missing_keypoints_sum = 0
frame_count = 0

start_time = time.time()
print_interval = 1


def main():
    global total_keypoints, missing_keypoints_sum, frame_count, start_time
    while True:
        hasFrame, frame = cap.read()
        if not hasFrame:
            print("Error: Can't receive frame from webcam.")
            break
        output = model(frame, save=False, verbose=False)
        if output is not None:
            total_keypoints_in_frame = sum(len(k.data[0]) for k in output[0].keypoints[0])
            total_keypoints += total_keypoints_in_frame
        if output is not None:
            missing_keypoints_in_frame = total_keypoints - total_keypoints_in_frame
            missing_keypoints_ratio = missing_keypoints_in_frame / total_keypoints if total_keypoints > 0 else 0
            missing_keypoints_sum += missing_keypoints_ratio
            frame_count += 1

        if output is not None:
            try:
                for idx, kpt in enumerate(output[0].keypoints[0]):
                    for k in kpt.data[0][:, 0:2].cpu().detach().numpy():
                        k = [int(x) for x in k]
                        frame = cv2.circle(frame, (k[0], k[1]), 8, (0, 0, 255), thickness=3, lineType=cv2.FILLED)
            except Exception as e:
                print(e)
                pass

        cv2.imshow("display", frame)

        if time.time() - start_time >= print_interval:
            print("Sum of missing keypoints ratios per second:", missing_keypoints_sum)
            start_time = time.time()
            missing_keypoints_sum = 0

        # Exit loop on 'q' key press
        if cv2.waitKey(1) == ord('q'):
            break

    # Calculate the average missing keypoints ratio per second
    average_missing_keypoints_ratio_per_second = missing_keypoints_sum / frame_count if frame_count > 0 else 0
    print("Average missing keypoints ratio per second:", average_missing_keypoints_ratio_per_second)
    print("Total keypoints  :",total_keypoints)
    print("Total missing keypoints ratio :", missing_keypoints_sum)

    # Release capture and destroy windows
    cap.release()
    cv2.destroyAllWindows()


if __name__ == '__main__':
    main()
