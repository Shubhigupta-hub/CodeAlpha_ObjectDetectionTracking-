import argparse
import cv2
from ultralytics import YOLO
from deep_sort_realtime.deepsort_tracker import DeepSort


def get_args():
    parser = argparse.ArgumentParser(description="Object Detection and Tracking with YOLOv8 + Deep SORT")
    parser.add_argument(
        "--source",
        type=str,
        default="0",
        help="Webcam index (0, 1, ...) or path to video file",
    )
    parser.add_argument(
        "--model",
        type=str,
        default="yolov8n.pt",
        help="Path to YOLOv8 model file",
    )
    parser.add_argument(
        "--conf",
        type=float,
        default=0.4,
        help="Confidence threshold",
    )
    parser.add_argument(
        "--img-size",
        type=int,
        default=640,
        help="Inference image size",
    )
    return parser.parse_args()


def main():
    args = get_args()

    # Load YOLO model
    model = YOLO(args.model)
    class_names = model.names

    # Initialize Deep SORT tracker
    tracker = DeepSort(max_age=30, n_init=2, max_cosine_distance=0.3)

    # Open video source
    source = int(args.source) if args.source.isdigit() else args.source
    cap = cv2.VideoCapture(source)

    if not cap.isOpened():
        print("Error: Could not open video source.")
        return

    print("Press 'q' to quit.")

    while True:
        success, frame = cap.read()
        if not success:
            print("End of video or unable to read frame.")
            break

        # YOLO inference
        results = model.predict(
            source=frame,
            imgsz=args.img_size,
            conf=args.conf,
            verbose=False
        )[0]

        detections = []

        # Prepare detections for Deep SORT
        if results.boxes is not None and len(results.boxes) > 0:
            for box in results.boxes:
                x1, y1, x2, y2 = box.xyxy[0].cpu().numpy()
                conf = float(box.conf[0].cpu().numpy())
                cls_id = int(box.cls[0].cpu().numpy())
                class_name = class_names[cls_id]

                # Deep SORT expects: ([x, y, w, h], confidence, class_name)
                detections.append((
                    [float(x1), float(y1), float(x2 - x1), float(y2 - y1)],
                    conf,
                    class_name
                ))

        # Update tracker
        tracks = tracker.update_tracks(detections, frame=frame)

        # Draw tracking results
        for track in tracks:
            if not track.is_confirmed() or track.time_since_update > 1:
                continue

            ltrb = track.to_ltrb()
            x1, y1, x2, y2 = map(int, ltrb)
            track_id = track.track_id
            label = track.get_det_class() if track.get_det_class() else "Object"

            # Bounding box
            cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)

            # Label background
            text = f"{label} ID:{track_id}"
            (tw, th), _ = cv2.getTextSize(text, cv2.FONT_HERSHEY_SIMPLEX, 0.6, 2)
            cv2.rectangle(frame, (x1, y1 - 25), (x1 + tw + 10, y1), (0, 255, 0), -1)

            # Label text
            cv2.putText(
                frame,
                text,
                (x1 + 5, y1 - 7),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.6,
                (0, 0, 0),
                2
            )

        cv2.imshow("Object Detection and Tracking", frame)

        if cv2.waitKey(1) & 0xFF == ord("q"):
            break

    cap.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
