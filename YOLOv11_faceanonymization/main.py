from facenet_pytorch import MTCNN
import torch
import cv2
from ultralytics import YOLO
from ultralytics.utils.plotting import Annotator, colors

# Initialize YOLO and MTCNN models
model = YOLO("yolo11n.pt")
names = model.names
mtcnn = MTCNN(keep_all=True, device='cuda' if torch.cuda.is_available() else 'cpu')


print(torch.cuda.is_available())
# exit()

# Define the target class for face detection
target_class = "person"

cap = cv2.VideoCapture("test_videos/test3.mp4")
assert cap.isOpened(), "Error reading video file"
w, h, fps = (int(cap.get(x)) for x in (cv2.CAP_PROP_FRAME_WIDTH, cv2.CAP_PROP_FRAME_HEIGHT, cv2.CAP_PROP_FPS))

blur_ratio = 30
video_writer = cv2.VideoWriter("output_videos/face_blurring_output.avi", cv2.VideoWriter_fourcc(*"mp4v"), fps, (w, h))

while cap.isOpened():
    success, im0 = cap.read()
    if not success:
        print("Video frame is empty or video processing has been successfully completed.")
        break

    results = model.predict(im0, show=False)
    boxes = results[0].boxes.xyxy.cpu().tolist()
    clss = results[0].boxes.cls.cpu().tolist()
    annotator = Annotator(im0, line_width=2, example=names)

    if boxes is not None:
        for box, cls in zip(boxes, clss):
            class_name = names[int(cls)]
            if class_name == target_class:
                # annotator.box_label(box, color=colors(int(cls), True), label=class_name)

                # Extract person region for face detection
                person_region = im0[int(box[1]):int(box[3]), int(box[0]):int(box[2])]

                # Ensure person region is large enough
                if person_region.shape[0] > 30 and person_region.shape[1] > 30:
                    # Convert to RGB before passing to MTCNN
                    person_region_rgb = cv2.cvtColor(person_region, cv2.COLOR_BGR2RGB)

                    # Detect faces using MTCNN within the person region
                    face_boxes, _ = mtcnn.detect(person_region_rgb)

                    # Ensure face_boxes is not None and not empty
                    if face_boxes is not None and len(face_boxes) > 0:
                        for fbox in face_boxes:
                            x1, y1, x2, y2 = map(int, fbox)
                            
                            # Ensure the face coordinates are within bounds
                            if x1 < x2 and y1 < y2:
                                face = person_region[y1:y2, x1:x2]

                                # Check if the face region is not empty
                                if face.size > 0:
                                    blurred_face = cv2.blur(face, (blur_ratio, blur_ratio))
                                    person_region[y1:y2, x1:x2] = blurred_face

                    # Replace the original person region with the processed one
                    im0[int(box[1]):int(box[3]), int(box[0]):int(box[2])] = person_region
    cv2.imshow("Face Blur", im0)
    video_writer.write(im0)
    if cv2.waitKey(1) & 0xFF == ord("q"):
        break

cap.release()
video_writer.release()
cv2.destroyAllWindows()