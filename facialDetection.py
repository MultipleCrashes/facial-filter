
import cv2

# Open the capture device - change as needed. With MBP, camera accessed via 0
CAMERA = cv2.VideoCapture(0)
FRONT_FACIAL_CLASSIFIER = cv2.CascadeClassifier('haar_frontalface.xml')
hat = cv2.imread('santahat.png', -1)

def above_face_overlay(frame, x, w, y, h, overlay_t_img):
  ROI_ABOVE_FACE = frame[max(0, y-h):y, x:x+w]
  OVERLAYED_ROI = overlay_roi(ROI_ABOVE_FACE, overlay_t_img)
  frame[max(0, y-h):y, x:x+w] = OVERLAYED_ROI
  pass

def overlay_roi(roi, overlay_image):
  # resize the overlay image to fit the ROI target
  resizedDimensions = (roi.shape[1], roi.shape[0])
  resizedOverlayDst = cv2.resize(overlay_image, (resizedDimensions), fx=0,fy=0, interpolation=cv2.INTER_NEAREST)

  # Huge thanks to Dan Masek: https://stackoverflow.com/a/37198079
  bgrImage = resizedOverlayDst[:,:,:3] # BGR
  alphaMask1 = resizedOverlayDst[:,:,3:]  # Alpha
  alphaMask2 = cv2.bitwise_not(alphaMask1) # alphaMask1 + alphaMask2 = [1,1,1,1...]

  threeChanAlphaMask1 = cv2.cvtColor(alphaMask1, cv2.COLOR_GRAY2BGR)
  threeChanAlphaMask2 = cv2.cvtColor(alphaMask2, cv2.COLOR_GRAY2BGR)

  # Use inverted alpha mask to multiply pixels from background image by pixel of the inverted alphamask
  # The inverted alpha mask will have 255 set for pixels there is nothing (i.e. the pixel is completely transparent).
  # As a result, during multiplication, the background pixel will have 100% of the weight (importance) for that pixel
  backgroundROI = (roi * 1/255.0) * (threeChanAlphaMask2 * 1/255.0)

  # Use the alpha mask here so the overlay image will have its transparent pixels equal to zero.
  # This is useful when we add the foregroundROI and backgroundROI, the correct pixel value from the background
  # will be used in transparent pixels (alpha=0)
  foregroundROI = (bgrImage * 1/255.0) * (threeChanAlphaMask1 * 1/255.0)

  # TODO: dive deeper into OpenCV's Mat multiplications - does a weird wrap around/average?
  return cv2.addWeighted(backgroundROI, 255.0, foregroundROI, 255.0, 0.0)

while(True):
  # Capture the frame
  _, FRAME = CAMERA.read()

  # Gray and colored instances of frame
  GRAY = cv2.cvtColor(FRAME, cv2.COLOR_BGR2GRAY)
  COLORED_IMAGE= cv2.cvtColor(FRAME, cv2.COLOR_BGRA2BGR)

  # Detect faces
  rects = FRONT_FACIAL_CLASSIFIER.detectMultiScale(GRAY, scaleFactor=1.3,
	  minNeighbors=10, minSize=(75, 75))

  # Draw rectangles around all detected faces - this was taken directly out of the cat detection face toturial
  for (i, (x, y, w, h)) in enumerate(rects):
    print('Face at  ' + str(x) + ',' + str(y))
    # Overlay the hat at target ROI at top of detected rectangle
    above_face_overlay(COLORED_IMAGE, x, w, y, h, hat)
  # show the image
  cv2.imshow('frame', COLORED_IMAGE)

  if cv2.waitKey(1) & 0xFF == ord('q'):
    break

cameraCapture.release()
cv2.destroyAllWindows()