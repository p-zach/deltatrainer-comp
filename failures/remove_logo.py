# Author: Porter Zach
# Python 3.9
import cv2
import numpy as np

# proto = r"pose\coco\pose_deploy_linevec.prototxt"
# caffe = r"pose\coco\pose_iter_440000.caffemodel"
proto = r"pose\mpi\pose_deploy_linevec_faster_4_stages.prototxt"
caffe = r"pose\mpi\pose_iter_160000.caffemodel"
# proto = "pose_body25/pose_deploy.prototxt"
# caffe = "pose_body25/pose_iter_584000.caffemodel"

path = "test_cases/easy/easy_0.mp4"
path = "test_cases/hard/hard_1.mp4"
path = "test_cases/hard/hard_2.mp4"

net = cv2.dnn.readNetFromCaffe(proto, caffe)

cap = cv2.VideoCapture(path)

pairs_mpi = [[0,1], [1,2], [2,3], [3,4], [1,5], [5,6], [6,7],
            [1,14], [14,8], [8,9], [9,10], [14,11], [11,12], [12,13]]
relative_points = [1, 2, 5, 14, 8, 11]

# def drawKeypoints(img, pts):
#     """Draws the supplied keypoints and their connections onto an image.

#     Args:
#         img (numpy.ndarray): The image to draw the keypoints onto.
#         pts (numpy.ndarray): The list of keypoints.
#     """
#     def toPoint(pt):
#         return (int(pt[0]), int(pt[1]))

#     for i in range(len(pts)):
#         cv2.circle(img, toPoint(pts[i]), 5, (0, 255, 255), thickness=-1, lineType=cv2.FILLED)
    
#     for pair in pairs_mpi:
#         a = pair[0]
#         b = pair[1]
#         if not(pts[a][0] == 0 and pts[a][1] == 0) and not(pts[b][0] == 0 and pts[b][1] == 0):
#             cv2.line(img, toPoint(pts[a]), toPoint(pts[b]), (0, 255, 0), 2, lineType=cv2.LINE_AA)

ret, img = cap.read()
try:
    while ret:
        if False:
            cv2.imshow("output", img)
            cv2.waitKey(1)
            ret, img = cap.read()
            continue

        ret, img = cap.read()
        scalar = 2

        poseImg = cv2.resize(img, (img.shape[1] // scalar, img.shape[0] // scalar))
        w = poseImg.shape[1]
        h = poseImg.shape[0]

        inp = cv2.dnn.blobFromImage(poseImg, 1.0 / 255, (w, h), (0, 0, 0), swapRB = False, crop = False)

        net.setInput(inp)

        output = net.forward()

        oh = output.shape[2]
        ow = output.shape[3]

        numPoints = 15
        points = []

        for i in range(numPoints):
            probMap = output[0, i, ...]

            minVal, maxVal, minPos, maxPos = cv2.minMaxLoc(probMap)

            x = (w * maxPos[0]) / ow
            y = (h * maxPos[1]) / oh

            points.append((int(x * scalar), int(y * scalar)))

        x1, x2, y1, y2 = 99999, 0, 99999, 0

        for pt in points:
            if pt[0] < x1:
                x1 = pt[0]
            elif pt[0] > x2:
                x2 = pt[0]
            if pt[1] < y1:
                y1 = pt[1]
            elif pt[1] > y2:
                y2 = pt[1]
        
        imW = w * scalar
        imH = h * scalar

        x1 = int(max(x1 - imW * .01, 0))
        x2 = int(min(x2 + imW * .01, imW))
        y1 = int(max(y1 - imH * .01, 0))
        y2 = int(min(y2 + imH * .01, imH))

        roi = img[y1:y2, x1:x2]

        minBlack = np.array([0, 0, 0])
        maxBlack = np.array([360, 255, 75])

        minWhite = np.array([0, 0, 150])
        maxWhite = np.array([360, 50, 255])

        hsv = cv2.cvtColor(roi, cv2.COLOR_BGR2HSV)

        mask_black = cv2.inRange(hsv, minBlack, maxBlack)
        mask_white = cv2.inRange(hsv, minWhite, maxWhite)

        roi_black = cv2.bitwise_and(roi, roi, mask = mask_black)
        roi_white = cv2.bitwise_and(roi, roi, mask = mask_white)

        black_grayscale = cv2.cvtColor(roi_black, cv2.COLOR_BGR2GRAY)
        _, black_threshed = cv2.threshold(black_grayscale, 5, 255, cv2.THRESH_BINARY)

        # roi_white = cv2.bitwise_not(roi_white)

        # mask the white mask by the black mask dilated

        kernel_small = np.ones((3,3), np.uint8)
        kernel_5 = np.ones((5,5), np.uint8)
        kernel_big = np.ones((7,7),np.uint8)
        kernel_9 = np.ones((13,13),np.uint8)

        black_threshed_dilated = cv2.morphologyEx(black_threshed, cv2.MORPH_DILATE, kernel_big)

        roi_white = cv2.bitwise_and(roi_white, roi_white, mask = black_threshed_dilated)

        result_morph = cv2.cvtColor(roi_white, cv2.COLOR_BGR2GRAY)
        _, result_morph = cv2.threshold(result_morph, 10, 255, cv2.THRESH_BINARY)

        orb = cv2.ORB_create()
        kp = orb.detect(result_morph)

        kp_mask = np.zeros_like(mask_black)
        avg = (0, 0)
        for keypoint in kp:
            avg = (avg[0] + keypoint.pt[0], avg[1] + keypoint.pt[1])
            kp_mask[int(keypoint.pt[1]), int(keypoint.pt[0])] = 255
        avg = (int(avg[0] / len(kp)), int(avg[1] / len(kp)))

        avg_mask = np.zeros_like(kp_mask)
        cv2.circle(avg_mask, avg, 100, (255,255,255), thickness=-1)

        kp_mask = cv2.bitwise_and(kp_mask, kp_mask, mask=avg_mask)
        
        kp_mask = cv2.morphologyEx(kp_mask, cv2.MORPH_CLOSE, kernel_big)
        kp_mask = cv2.morphologyEx(kp_mask, cv2.MORPH_OPEN, kernel_small)
        kp_mask = cv2.morphologyEx(kp_mask, cv2.MORPH_DILATE, kernel_9)


        # result_morph = cv2.adaptiveThreshold(result_morph, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, 11, 2)
        # result_morph = cv2.morphologyEx(result_morph, cv2.MORPH_BLACKHAT, kernel_big)
        # result_morph = cv2.morphologyEx(result_morph, cv2.MORPH_CLOSE, kernel_big)
        # result_morph = cv2.morphologyEx(result_morph, cv2.MORPH_OPEN, kernel_small)


        #result_morph = cv2.bitwise_and(result_morph, result_morph, mask=kp_mask)
        # imperative
        #result_morph = cv2.morphologyEx(result_morph, cv2.MORPH_DILATE, kernel_big)

        contours = cv2.findContours(result_morph, cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)[0]

        mask_final = np.zeros_like(roi)
        cv2.drawContours(mask_final, [max(contours, key=cv2.contourArea)], -1, (255,255,255), thickness=-1)
        
        mask_final = cv2.cvtColor(mask_final, cv2.COLOR_BGR2GRAY)
        result = cv2.inpaint(roi, mask_final, 9, cv2.INPAINT_TELEA)

        
        result_morph = cv2.cvtColor(result_morph, cv2.COLOR_GRAY2BGR)
        black_threshed = cv2.cvtColor(black_threshed_dilated, cv2.COLOR_GRAY2BGR)
        mask_final = cv2.cvtColor(mask_final, cv2.COLOR_GRAY2BGR)
        # result_morph = cv2.drawKeypoints(result_morph, kp, None, color = (0, 255, 0), flags = 0)

        # cv2.circle(result_morph, avg, 10, (0, 0, 255), thickness = -1)
        img[y1:y2, x1:x2] = result#_morph#cv2.cvtColor(kp_mask, cv2.COLOR_GRAY2BGR)
        # drawKeypoints(img, points)

        cv2.imshow("output", img)

        cv2.waitKey(1)
except KeyboardInterrupt:
    pass

cv2.destroyAllWindows()