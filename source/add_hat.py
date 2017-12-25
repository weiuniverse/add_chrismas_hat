import dlib
import cv2
import sys
from imutils import face_utils


def add_hat_helper(img,rect,shape,hat_mask,hat_img):
    '''
    find the hat for one face in the image
    '''
    (x, y, w, h) = face_utils.rect_to_bb(rect)
    # get the size of hat
    point1 = shape.part(0)
    point2 = shape.part(2)
    eyes_center = ((point1.x+point2.x)//2,(point1.y+point2.y)//2)
    factor = 1.5
    resized_hat_h = int(round(hat_img.shape[0]*w/hat_img.shape[1]*factor))
    resized_hat_w = int(round(hat_img.shape[1]*w/hat_img.shape[1]*factor))
    resized_hat = cv2.resize(hat_img,(resized_hat_w,resized_hat_h))

    if resized_hat_h > y:
      resized_hat_h = y-1

    mask = cv2.resize(hat_mask,(resized_hat_w,resized_hat_h))
    mask_inv = cv2.bitwise_not(mask)
    dh = 0
    dw = 0
    # get the background
    bg_roi = img[y+dh-resized_hat_h:y+dh,(eyes_center[0]-resized_hat_w//3):(eyes_center[0]+resized_hat_w//3*2)]
    bg_roi = bg_roi.astype(float)

    mask_inv = cv2.merge((mask_inv,mask_inv,mask_inv))
    alpha = mask_inv.astype(float)/255
    alpha = cv2.resize(alpha,(bg_roi.shape[1],bg_roi.shape[0]))
    # bg = cv2.bitwise_and(bg_roi,bg_roi,mask = alpha)
    bg = cv2.multiply(alpha, bg_roi)
    bg = bg.astype('uint8')

    # get the hat
    hat = cv2.bitwise_and(resized_hat,resized_hat,mask = mask)
    hat = cv2.resize(hat,(bg_roi.shape[1],bg_roi.shape[0]))

    # add the hat
    add_hat = cv2.add(bg,hat)
    img[y+dh-resized_hat_h:y+dh,(eyes_center[0]-resized_hat_w//3):(eyes_center[0]+resized_hat_w//3*2)] = add_hat
    return img


def add_hat(filepath='../images/Solvay.jpg',hatpath= '../images/hat.jpg',maskpath= '../images/mask.jpg'):
    try:
        img = cv2.imread(filepath)
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    except:
        print("cannot find the file, check the path")
        return 0

    rects = detector(gray,2)
    hat_img = cv2.imread(hatpath)
    hat_mask = cv2.imread(maskpath,0)
    hat_mask = cv2.resize(hat_mask,(hat_img.shape[0],hat_img.shape[1]))
    if len(rects)>0:
        for rect in rects:
            (x, y, w, h) = face_utils.rect_to_bb(rect)
            cv2.rectangle(img,(x,y),(x+w,y+h),(255,0,0),2,8,0)
            shape = predictor(img,rect)
            img_hat = add_hat_helper(img,rect,shape,hat_mask,hat_img)
            cv2.imwrite('../chrismashat.jpg',img_hat)
    else:
        print("no face")
    return img_hat

# load the model
detector = dlib.get_frontal_face_detector()
predictor = dlib.shape_predictor("../model/shape_predictor_5_face_landmarks.dat")

def main(argv):
    print("usage: python add_hat.py imagepath")
    if argv == None:
        add_hat()
    else:
        filepath = str(argv[1])
        add_hat(filepath)

if __name__ == '__main__':
    main(sys.argv)
