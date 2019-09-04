""" Utility functions for """
import cv2

def draw_detection_with_label(img, bbox, label, known, font=cv2.FONT_HERSHEY_SIMPLEX, fontScale=0.5, fontThickness=1):
    """ Draw bounding box and label around detection in an image

        Args:
            img (H x W np.array): Source image
            bbox (tuple): Bounding box of the face - (x, y, w, h)
            label (str): Label of the face
            known (bool): Boolean indication whether the face was known
            font (int): OpenCV font type
            fontScale (float): Size of output font compared to original font
            fontThickness (float): Thickness of font strokes

        Return:
            img ((H x W) np.array) Source image with added bounding box and labels
    """
    x, y, w, h = bbox

    # if knonw then green, else red
    color = ((0,255,00) if known else (0,0,255))

    cv2.rectangle(img, (x,y),(x+w, y+h), color)

    #Get size of the label
    (text_width, text_height), _ = cv2.getTextSize(label, font, fontScale=fontScale, thickness=fontThickness)

    # Fit rectangle to text
    cv2.rectangle(img, (x,y),(x+text_width, y-text_height), color, -1)
    cv2.putText(img, label, (x, y-1), font, fontScale, (0,0,0), fontThickness, cv2.LINE_AA)

    return img
