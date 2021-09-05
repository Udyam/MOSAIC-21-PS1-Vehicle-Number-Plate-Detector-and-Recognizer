import cv2
import numpy as np
from matplotlib import pyplot as plt
import scipy.ndimage
import numpy as np
import cv2
import os
import scipy.fftpack
from imutils.perspective import four_point_transform
import math
from scipy import ndimage




def getSkewAngle(cvImage) -> float:
    # Prep image, copy, convert to gray scale, blur, and threshold
    newImage = cvImage.copy()
    # Apply dilate to merge text into meaningful lines/paragraphs.
    # Use larger kernel on X axis to merge characters into single line, cancelling out any spaces.
    # But use smaller kernel on Y axis to separate between different blocks of text
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (30, 1))
    dilate = cv2.dilate(newImage, kernel, iterations=2)
    #cv2.imshow("dilate",dilate)
    #cv2.waitKey(0)
    # Find all contours
    contours, hierarchy = cv2.findContours(dilate, cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)
    contours = sorted(contours, key = cv2.contourArea, reverse = True)
    # Find largest contour and surround in min area box
    largestContour = contours[0]
    minAreaRect = cv2.minAreaRect(largestContour)

    # Determine the angle. Convert it to the value that was originally used to obtain skewed image
    angle = minAreaRect[-1]
    print(angle)
    if angle>45:
        return 90-angle
    else :
        return -angle

def rotateImage(cvImage, angle: float):
    newImage = cvImage.copy()
    (h, w) = newImage.shape[:2]
    center = (w // 2, h // 2)
    M = cv2.getRotationMatrix2D(center, angle, 1.0)
    newImage = cv2.warpAffine(newImage, M, (w, h), flags=cv2.INTER_CUBIC, borderMode=cv2.BORDER_REPLICATE)
    return newImage



# Deskew image
def deskew(cvImage):
    angle = getSkewAngle(cvImage)
    return rotateImage(cvImage, -1.0 * angle)



def preprocess(image):
    cp=image.copy()
    cnts,_ = cv2.findContours(cp, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    area=[]
    for cntr in cnts:
        if(cv2.contourArea(cntr)<1000):
            cv2.drawContours(cp, [cntr], -1, 0, -1)
        else :
            area.append(cv2.contourArea(cntr))

    area.sort()
    meda=area[len(area)//2]
    for cntr in cnts:
        if(cv2.contourArea(cntr)<meda-9000 or  cv2.contourArea(cntr)>meda+9000):
            area.append(cv2.contourArea(cntr))
            cv2.drawContours(cp, [cntr], -1, 0, -1)
    #cv2.imshow("NMOS",cp)
    #cv2.waitKey()
    return(cp)


def imclearborder(imgBW, radius):

    # Given a black and white image, first find all of its contours
    imgBWcopy = imgBW.copy()
    contours,hierarchy = cv2.findContours(imgBWcopy.copy(), cv2.RETR_LIST,
        cv2.CHAIN_APPROX_SIMPLE)

    # Get dimensions of image
    imgRows = imgBW.shape[0]
    imgCols = imgBW.shape[1]

    contourList = [] # ID list of contours that touch the border
    # For each contour...
    for idx in np.arange(len(contours)):
        # Get the i'th contour
        cnt = contours[idx]

        # Look at each point in the contour
        for pt in cnt:
            rowCnt = pt[0][1]
            colCnt = pt[0][0]

            # If this is within the radius of the border
            # this contour goes bye bye!
            check1 = (rowCnt >= 0 and rowCnt < radius) or (rowCnt >= imgRows-1-radius and rowCnt < imgRows)
            check2 = (colCnt >= 0 and colCnt < radius) or (colCnt >= imgCols-1-radius and colCnt < imgCols)

            if check1 or check2:
                contourList.append(idx)
                break

    for idx in contourList:
        cv2.drawContours(imgBWcopy, contours, idx, (0,0,0), -1)
    #cv2.imshow("ISSSS",imgBWcopy)
    #cv2.waitKey()
    return imgBWcopy


def connectedcomp(thresh):
    connectivity = 4

    # Perform the operation
    output = cv2.connectedComponentsWithStats(thresh, connectivity, cv2.CV_32S)
    numLabels, labels, stats, centroids=output
    mask = np.zeros(thresh.shape, dtype="uint8")
    new_mask=mask.copy()
    #cv2.imshow("winname",thresh)
    #cv2.waitKey(0)
    indx=np.argsort(stats[:,-1])[::-1]
    vis=np.zeros(thresh.shape[1])
    sep=new_mask.copy()
    coord=[]
    for i in indx:
        if i==0:
            continue
        x = stats[i, cv2.CC_STAT_LEFT]
        y = stats[i, cv2.CC_STAT_TOP]
        w = stats[i, cv2.CC_STAT_WIDTH]
        h = stats[i, cv2.CC_STAT_HEIGHT]
        area = stats[i, cv2.CC_STAT_AREA]
        keepWidth = w >=20 and w < 180
        keepHeight = h >= 30 and h < 512
        keepArea = area >1300 and area < 17000
        pos=(x+w//2)>35 and (x+w//2)<1024-35 and (y+h//2)>65 and (y+h//2)<512-65 and y>15 and y<495 and x>15 and x<1005
        # ensure the connected component we are examining passes all
        # three tests
        ratio=(1.0*h)/w
        if all((keepWidth, keepHeight, keepArea,pos)) and ratio>=0.80 and ratio<=10.0 and vis[x+w//2]==0:
            #print(ratio)
            componentMask = (labels == i).astype("uint8") * 255
            mask = cv2.bitwise_or(mask, componentMask)
            #mask=cv2.circle(mask,(x+w//2,y+h//2),5,(255,255,255))
            #print(x+w//2,y+h//2,w,h,area)
            coord.append((y+h//2,x+w//2))
            for j in range(x,x+w):
                vis[j]=1
            cv2.imshow("ccur", mask)
            #cv2.imshow("ComponentMask",componentMask)

            cv2.waitKey(500)

    #cv2.destroyAllWindows()
    cv2.imwrite("./Connected_components.jpg",mask)
    return mask,coord



def extract(image):
    newimg=image.copy()
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (1,7))
    dilate = cv2.dilate(newimg, kernel, iterations=2)
    #cv2.imshow("j",dilate)
    #cv2.waitKey(0)
    cnts,_ = cv2.findContours(newimg, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    #print(np.mean(cnts[0],axis=0))
    cnts.sort(key=lambda x:np.mean(x,axis=0)[0][0])
    ls=[]
    for cntr in cnts:
        if(cv2.contourArea(cntr)>=1000):
            x,y,w,h = cv2.boundingRect(cntr)
            cropped=image[y:y+h,x:x+w]
            cropped=cv2.resize(cropped,(256,256))
            kernel = np.ones((3,3), np.uint8)
            cropped=cv2.erode(cropped, kernel,iterations=1)
            ls.append(cropped)
    return(ls)


def printextracted(ls):
    for img in ls:
        cv2.imshow("cur",img)
        cv2.waitKey(1000)
    cv2.destroyAllWindows()
    return




def apply_brightness_contrast(input_img, brightness = 0, contrast = 0):

    if brightness != 0:
        if brightness > 0:
            shadow = brightness
            highlight = 255
        else:
            shadow = 0
            highlight = 255 + brightness
        alpha_b = (highlight - shadow)/255
        gamma_b = shadow

        buf = cv2.addWeighted(input_img, alpha_b, input_img, 0, gamma_b)
    else:
        buf = input_img.copy()

    if contrast != 0:
        f = 131*(contrast + 127)/(127*(131-contrast))
        alpha_c = f
        gamma_c = 127*(1-f)

        buf = cv2.addWeighted(buf, alpha_c, buf, 0, gamma_c)
        cv2.imwrite("./after_adding_brightness_contrast.jpg",buf)

    return buf

#img = cv2.imread("C:\\Users\\Dell\\Downloads\\MosaicPS2\\images\\Data-Images\\Plates\\1.jpg")


def getprocessed(img):
    img=cv2.resize(img,(1024,512))
    cv2.imwrite("./original_resized.jpg",img)
    #dilated_img = cv2.dilate(img, np.ones((7,7), np.uint8),iterations=7)
    #bg_img = cv2.medianBlur(dilated_img,7)
    #diff_img =255-cv2.absdiff(img, bg_img)

    #norm_img = diff_img.copy() # Needed for 3.x compatibility
    #cv2.normalize(diff_img, norm_img, alpha=0, beta=255, norm_type=cv2.NORM_MINMAX, dtype=cv2.CV_8UC1)


    #_, thr_img = cv2.threshold(norm_img, 180, 0, cv2.THRESH_TRUNC)
    #cv2.normalize(thr_img, thr_img, alpha=0, beta=255, norm_type=cv2.NORM_MINMAX, dtype=cv2.CV_8UC1)

    img=cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)
    img=cv2.fastNlMeansDenoising(img,None,10,7,21)


    # Number of rows and columns
    rows = img.shape[0]
    cols = img.shape[1]
    # Convert image to 0 to 1, then do log(1 + I)
    imgLog = np.log1p(np.array(img, dtype="float") / 255)
    # Create Gaussian mask of sigma = 10
    M = 2*rows + 1
    N = 2*cols + 1
    sigma = 10
    (X,Y) = np.meshgrid(np.linspace(0,N-1,N), np.linspace(0,M-1,M))
    centerX = np.ceil(N/2)
    centerY = np.ceil(M/2)
    gaussianNumerator = (X - centerX)**2 + (Y - centerY)**2
    # Low pass and high pass filters
    Hlow = np.exp(-gaussianNumerator / (2*sigma*sigma))
    Hhigh = 1 - Hlow
    # Move origin of filters so that it's at the top left corner to
    # match with the input image
    HlowShift = scipy.fftpack.ifftshift(Hlow.copy())
    HhighShift = scipy.fftpack.ifftshift(Hhigh.copy())
    # Filter the image and crop
    If = scipy.fftpack.fft2(imgLog.copy(), (M,N))
    Ioutlow = scipy.real(scipy.fftpack.ifft2(If.copy() * HlowShift, (M,N)))
    Iouthigh = scipy.real(scipy.fftpack.ifft2(If.copy() * HhighShift, (M,N)))
    # Set scaling factors and add
    gamma1 = 0.2
    gamma2 = 4.3
    Iout = gamma1*Ioutlow[0:rows,0:cols] + gamma2*Iouthigh[0:rows,0:cols]
    # Anti-log then rescale to [0,1]
    Ihmf = np.expm1(Iout)
    Ihmf = (Ihmf - np.min(Ihmf)) / (np.max(Ihmf) - np.min(Ihmf))
    Ihmf2 = np.array(255*Ihmf, dtype="uint8")
    gray = Ihmf2.copy()
    cv2.imwrite("./homomorphic.jpg",gray)




    equalized = cv2.equalizeHist(gray)
    #kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (8,5))
    #equalized=cv2.erode(equalized, kernel,iterations=1)
    equalized=apply_brightness_contrast(equalized,20,50)
    cv2.imshow("equalized",equalized)
    cv2.imwrite("./equalized.jpg",equalized)
    cv2.waitKey(0)
    gray_img = cv2.GaussianBlur(equalized, (11,11), 0)
    th1=cv2.adaptiveThreshold(gray_img, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY_INV, 199, 20)
    ret, th2 = cv2.threshold(gray_img,23, 255, cv2.THRESH_BINARY_INV)
    #th2=cv2.erode(th2, kernel=np.ones((5,5),np.uint8),iterations=1)
    finalthresh=th2
    final,coord=connectedcomp(finalthresh)
    final=preprocess(final)
    warped=deskew(final)
    warped=cv2.resize(warped,(1024,212))
    kernel = np.ones((3,3),np.uint8)
    warped=cv2.dilate(warped, kernel,iterations=1)
    ls=extract(warped)
    printextracted(ls)
    cv2.imshow("final",warped)
    cv2.imwrite("./final.jpg",warped)
    cv2.waitKey(0)
    cv2.destroyAllWindows()
    return warped
    #kernel=np.ones((5,5),np.uint8)
    #final=cv2.erode(final, kernel,iterations=1)