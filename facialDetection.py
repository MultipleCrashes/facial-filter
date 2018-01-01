
import cv2

# Open the capture device - change as needed. With MBP, camera accessed via 0
CAMERA = cv2.VideoCapture(0)
FRONT_FACIAL_CLASSIFIER = cv2.CascadeClassifier('haar_frontalface.xml')
hat = cv2.imread('santahat.png', -1)

def overlay(roi, overlay_image):
  # resize the overlay image to fit the ROI target
  resized_dimentions = (roi.shape[1], roi.shape[0])
  resized_overlay_dst = cv2.resize(overlay_image, (resized_dimentions), fx=0, fy=0, interpolation=cv2.INTER_NEAREST)

  # Huge thanks to Dan Masek: https://stackoverflow.com/a/37198079
  bgr_image = resized_overlay_dst[:,:,:3] # BGR
  alpha_mask_1 = resized_overlay_dst[:,:,3:]
  alpha_mask_2 = cv2.bitwise_not(alpha_mask_1) # alpha mask 1 + alpha mask 2 = [1,1,1,1...]

  three_chan_alpha_mask_1 = cv2.cvtColor(alpha_mask_1, cv2.COLOR_GRAY2BGR)
  three_chan_alpha_mask_2 = cv2.cvtColor(alpha_mask_2, cv2.COLOR_GRAY2BGR)

  # Use inverted alpha mask to multiply pixels from background image by pixel of the inverted alphamask
  # The inverted alpha mask will have 255 set for pixels where the pixel is completely transparent.
  # As a result, during multiplication, the background pixel will have 100% of the weight for that pixel
  background_roi = (roi * 1/255.0) * (three_chan_alpha_mask_2 * 1/255.0)

  # Use the alpha mask here so the overlay image will have its transparent pixels equal to zero.
  # This is useful when we add the foreground ROI and background ROI together as the correct pixel
  # value from the background  will be used in transparent pixels (alpha=0)
  foreground_roi = (bgr_image * 1/255.0) * (three_chan_alpha_mask_1 * 1/255.0)

  # TODO: dive deeper into OpenCV's Mat multiplications - does a weird wrap around/average?
  return cv2.addWeighted(background_roi, 255.0, foreground_roi, 255.0, 0.0)

def overlay_img_above_facial_frame(facial_frame, x, w, y, h, overlay_t_img):
  top_of_frame_coord = max(0, y-h)
  rightmost_frame_coord = x+w

  # Squeeze the ROI to fit within screen bounds
  roi_above_face = facial_frame[top_of_frame_coord:y, x:rightmost_frame_coord]
  overlayed_roi = overlay(roi_above_face, overlay_t_img)

  # Overlay
  facial_frame[top_of_frame_coord:y, x:rightmost_frame_coord] = overlayed_roi
  pass

def get_gray_and_3_chan_colored_frames(frame):
  return cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY), cv2.cvtColor(frame, cv2.COLOR_BGRA2BGR)

def get_face_rects(frame):
  return FRONT_FACIAL_CLASSIFIER.detectMultiScale(frame, scaleFactor=1.3,
      minNeighbors=10, minSize=(75, 75))

def main():
  while(True):
    _, frame = CAMERA.read()

    # Gray and colored instances of frame
    # Note: COLORED_IMAGE is mutated
    gray_frame, colored_frame = get_gray_and_3_chan_colored_frames(frame)

    # Detect faces
    face_rects = get_face_rects(gray_frame)

    # Draw rectangles around all detected faces - this was taken directly out of the cat detection face toturial
    for (index, (x, y, w, h)) in enumerate(face_rects):
      print('Face at  ' + str(x) + ',' + str(y))
      # Overlay the hat at target ROI at top of detected rectangle
      overlay_img_above_facial_frame(colored_frame, x, w, y, h, hat)

    cv2.imshow('frame', colored_frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
      break

  cameraCapture.release()
  cv2.destroyAllWindows()

main()