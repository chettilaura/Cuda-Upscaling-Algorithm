import cv2
import sys
import os

# # This will display all the available mouse click events
# events = [i for i in dir(cv2) if 'EVENT' in i]
# print(events)

# This variable we use to store the pixel location
refPt = []
evt_cnt = 0
img = cv2.imread("./sample640x426.ppm")

# click event function
def click_event(event, x, y, flags, param):
    if event == cv2.EVENT_LBUTTONDOWN:
        print(x, ",", y)
        refPt.append([x, y])
        font = cv2.FONT_HERSHEY_SIMPLEX
        strXY = str(x)+", "+str(y)
        cv2.putText(img, strXY, (x, y), font, 0.5, (255, 255, 0), 2)
        cv2.imshow("image", img)

    if event == cv2.EVENT_RBUTTONDOWN:
        blue = img[y, x, 0]
        green = img[y, x, 1]
        red = img[y, x, 2]
        font = cv2.FONT_HERSHEY_SIMPLEX
        strBGR = str(blue)+", "+str(green)+","+str(red)
        cv2.putText(img, strBGR, (x, y), font, 0.5, (0, 255, 255), 2)
        cv2.imshow("image", img)

def cb_set(imagePath):
    # Here, you need to change the image name and it's path according to your directory
    img = cv2.imread(imagePath)
    cv2.imshow("image", img)
    # calling the mouse click event
    cv2.setMouseCallback("image", click_event)
    while (refPt.__len__() != 2):
        cv2.waitKey(1)
    #cv2.waitKey(0)
    cv2.destroyAllWindows()
    

def cb_get(args):
    print(refPt[0], refPt[1])
    cutOutCenterX = int((refPt[0][0] + refPt[1][0]) / 2)
    cutOutCenterY = int((refPt[0][1] + refPt[1][1]) / 2)
    cutOutWidth = int(abs(refPt[0][0] - refPt[1][0]))
    cutOutHeight = int(abs(refPt[0][1] - refPt[1][1]))
    if len(args) == 6:
        imagePath = args[1]
        flags = args[2]
        zoomLevel = args[3]
        gaussLength = args[4]
        gaussSigma = args[5]
        if flags.find("g") == -1:
            print("Error: gaussLength and gaussSigma are only valid if the g flag is set")
            return
        syscall = "./upsCu.exe " + flags + " " + imagePath + " " + " " + str(cutOutCenterX) + " " + str(cutOutCenterY) + " " + str(cutOutWidth) + " " + str(cutOutHeight) + " " + zoomLevel + " " + gaussLength + " " + gaussSigma
    elif len(args) == 5:
        imagePath = args[1]
        flags = args[2]
        zoomLevel = args[3]
        inputKernel = args[4]
        if flags.find("c") == -1:
            print("Error: inputKernel is only valid if the c flag is set")
            return
        syscall = "./upsCu.exe " + flags + " " + imagePath + " " + str(cutOutCenterX) + " " + str(cutOutCenterY) + " " + str(cutOutWidth) + " " + str(cutOutHeight) + " " + zoomLevel + " " + inputKernel
    print(syscall)
    os.system(syscall)

def main():
    args = sys.argv
    if len(args) == 6 or len(args) == 5:
        imagePath = args[1]
    else:
        print("Usage: upsCu.exe <imagePath> <flags> <zoomLevel> <gaussLength> <gaussSigma>")
        print("Usage: upsCu.exe <imagePath> <flags> <zoomLevel> <inputKernel>")
        return
    cb_set(imagePath)
    cb_get(args)
    output = cv2.imread("output.ppm")
    cv2.imshow("image", output)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()