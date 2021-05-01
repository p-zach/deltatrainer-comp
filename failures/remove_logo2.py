# Author: Porter Zach
# Python 3.9

import cv2
import numpy as np
import argparse

# path = "test_cases/easy/easy_0.mp4"
# path = "test_cases/hard/hard_1.mp4"
# path = "test_cases/hard/hard_0.mp4"

parser = argparse.ArgumentParser(description="")

parser.add_argument(action="append", dest="paths", nargs="?", help="The path to the video to remove the logo from and the path to save the edited video to (default: \'<videoPath>_edit.avi\').")
parser.add_argument("--scalar", action="store", dest="scalar", default=1, type=float, help="The scalar to resize the labeling image by (in case it's too big or small).")

args = parser.parse_args()

savePath = args.paths[1] if len(args.paths) > 1 else (args.paths[0] + "_edit.avi")

video = cv2.VideoCapture(args.paths[0])

opacity = .3

ix, iy = 0, 0
drawing = False
rect = { "x1": 0, "y1": 0, "x2": 0, "y2": 0 }
done_drawing = False

ret, img = video.read()
overlay = img.copy()

#region Initial Labeling

def scaleMouse(x, y):
    """Scales the mouse in case a scalar was specified.

    Args:
        x (int): The X coordinate of the mouse.
        y (int): The Y coordinate of the mouse.

    Returns:
        int, int: The scaled X and Y mouse coordinates.
    """
    return int(x / args.scalar), int(y / args.scalar)

def handleMouse(event, x, y, flags, param):
    """Handles mouse actions. Begins drawing on mouse down, sets label on mouse up.

    Args:
        event (int): The type of mouse event.
        x (int): The X coordinate of the mouse.
        y (int): The Y coordinate of the mouse.
        flags (int): Flags relevant to the mouse action.
        param (obj): Parameters regarding the mouse action.
    """
    # get global variables
    global drawing, ix, iy, done_drawing
    
    x, y = scaleMouse(x, y)

    # if the mouse was pressed:
    if event == cv2.EVENT_LBUTTONDOWN:
        # start drawing
        drawing = True
        # set initial values
        ix, iy = x, y
    # if the mouse was moved:
    elif event == cv2.EVENT_MOUSEMOVE:
        # update the rect if drawing
        if drawing:
            rect["x1"] = min(ix, x)
            rect["y1"] = min(iy, y)
            rect["x2"] = max(ix, x)
            rect["y2"] = max(iy, y)
    # if the mouse was released:
    elif event == cv2.EVENT_LBUTTONUP:
        if drawing:
            done_drawing = True

# def jump(frames):
#     """Jumps the video by the specified amount of frames.

#     Args:
#         frames (int): The number of frames to jump by.
#     """
#     # get global variables
#     global video, img

#     # get the position in the video
#     pos = video.get(cv2.CAP_PROP_POS_FRAMES)

#     # jump to the new position
#     video.set(cv2.CAP_PROP_POS_FRAMES, pos + frames)

#     # read the new frame 
#     ret, temp = video.read()
#     if ret:
#         img = temp
#     else:
#         # reset if it is out of the video bounds
#         print("New frame position " + str(pos + frames) + " is invalid.")
#         video.set(cv2.CAP_PROP_POS_FRAMES, pos)

# initialize the window
cv2.namedWindow("Window")
cv2.setMouseCallback("Window", handleMouse)

try:
    while not done_drawing:
        cv2.rectangle(overlay, (rect["x1"], rect["y1"]), (rect["x2"], rect["y2"]), (0, 255, 0), thickness=-1)

        cv2.addWeighted(overlay, opacity, img, 1 - opacity, 0, overlay)

        # resize image if specified
        if args.scalar != 1:
            overlay = cv2.resize(overlay, 
                (int(overlay.shape[1] * args.scalar), int(overlay.shape[0] * args.scalar)), 
                interpolation=cv2.INTER_AREA)
        
        cv2.imshow("Window", overlay)

        k = cv2.waitKey(1) & 0xFF
        # # handle frame jumps
        # if k == 44 & 0xFF: # ,
        #     jump(-5)
        # if k == 46 & 0xFF: # .
        #     jump(5)
        # exit if escape is pressed
        if k == 27:
            exit()
except KeyboardInterrupt:
    print("Editing interrupted. Not saving.")
    cv2.destroyAllWindows()
    video.release()
    exit()
    
cv2.destroyAllWindows()

#endregion

feature_params = dict( maxCorners = 500,
                       qualityLevel = 0.3,
                       minDistance = 7,
                       blockSize = 7 )

lk_params = dict( winSize  = (31, 31),
                  maxLevel = 2,
                  criteria = (cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 30, 0.1))

track_length = 10
detect_interval = 1
tracks = []
frame_index = 0
prev_gray = None

def getRect():
    return min(rect["x1"], rect["x2"]), min(rect["y1"], rect["y2"]), max(rect["x1"], rect["x2"]), max(rect["y1"], rect["y2"])
def setRect(x1, y1, x2, y2):
    global rect
    rect = { "x1": x1, "y1": y1, "x2": x2, "y2": y2 }

def avg(points):
    ax = 0
    ay = 0
    p = np.int32(points).reshape(-1, 2)
    for x, y in p:
        ax += x
        ay += y
    return (int(ax / p.shape[0]), int(ay / p.shape[0]))

def moveRect(center):
    x1, y1, x2, y2 = getRect()
    w = x2 - x1
    h = y2 - y1
    x1 = center[0] - w / 2
    x2 = center[0] + w / 2
    y1 = center[1] - h / 2
    y2 = center[1] + h / 2
    setRect(x1, y1, x2, y2)

def rectMask(img):
    r = getRect()
    w = img.shape[1]
    h = img.shape[0]
    o = .02
    return int(r[0] - w * o), int(r[1] - h * o), int(r[2] + w * o), int(r[3] + h * o)

def drawRect(img):
    x1, y1, x2, y2 = map(int, rectMask(img))
    overlay = img.copy()
    cv2.rectangle(overlay, (x1, y1), (x2, y2), (0, 255, 0), thickness=-1)

    img = cv2.addWeighted(overlay, .3, img, 1 - .3, 0, overlay)
    return img

try:
    while True:
        ret, img = video.read()
        if not ret:
            break
        
        minBlack = np.array([0, 0, 0])
        maxBlack = np.array([360, 255, 75])

        minWhite = np.array([0, 0, 150])
        maxWhite = np.array([360, 50, 255])

        hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)

        mask_black = cv2.inRange(hsv, minBlack, maxBlack)
        mask_white = cv2.inRange(hsv, minWhite, maxWhite)

        img_black = cv2.bitwise_and(img, img, mask = mask_black)
        img_white = cv2.bitwise_and(img, img, mask = mask_white)
        
        kernel_small = np.ones((3,3), np.uint8)
        kernel_5 = np.ones((5,5), np.uint8)
        kernel_big = np.ones((7,7),np.uint8)
        kernel_9 = np.ones((13,13),np.uint8)

        black_grayscale = cv2.cvtColor(img_black, cv2.COLOR_BGR2GRAY)
        _, black_threshed = cv2.threshold(black_grayscale, 5, 255, cv2.THRESH_BINARY)

        black_threshed_dilated = cv2.morphologyEx(black_threshed, cv2.MORPH_DILATE, kernel_big)

        img_white_masked = cv2.bitwise_and(img_white, img_white, mask = black_threshed_dilated)

        result_morph = cv2.cvtColor(img_white_masked, cv2.COLOR_BGR2GRAY)
        _, result_morph = cv2.threshold(result_morph, 10, 255, cv2.THRESH_BINARY)

        orb = cv2.ORB_create()
        kp = orb.detect(result_morph)

        # kp_mask = np.zeros_like(mask_black)
        # avg = (0, 0)
        # for keypoint in kp:
        #     avg = (avg[0] + keypoint.pt[0], avg[1] + keypoint.pt[1])
        #     kp_mask[int(keypoint.pt[1]), int(keypoint.pt[0])] = 255
        # avg = (int(avg[0] / len(kp)), int(avg[1] / len(kp)))

        # avg_mask = np.zeros_like(kp_mask)
        # cv2.circle(avg_mask, avg, 100, (255,255,255), thickness=-1)

        # kp_mask = cv2.bitwise_and(kp_mask, kp_mask, mask=avg_mask)
        
        # kp_mask = cv2.morphologyEx(kp_mask, cv2.MORPH_CLOSE, kernel_big)
        # kp_mask = cv2.morphologyEx(kp_mask, cv2.MORPH_OPEN, kernel_small)
        # kp_mask = cv2.morphologyEx(kp_mask, cv2.MORPH_DILATE, kernel_9)

        # result_morph = cv2.bitwise_and(result_morph, result_morph, mask=kp_mask)

        frame_gray = result_morph
        if prev_gray is None:
            prev_gray = frame_gray

        if frame_index == 0:
            mask = np.zeros_like(frame_gray)
            x1,y1,x2,y2 = rectMask(mask)
            mask[y1:y2, x1:x2] = 255
            p = cv2.goodFeaturesToTrack(frame_gray, mask=mask, **feature_params)
            if p is not None:
                for x, y in np.float32(p).reshape(-1, 2):
                    tracks.append([(x, y)])

        overlay = img.copy()

        if len(tracks) > 0:
            # get quick refs to previous and new frames
            img0, img1 = prev_gray, frame_gray
            p0 = np.float32([tr[-1] for tr in tracks]).reshape(-1, 1, 2)
            # get the current frame's optical flow
            p1, _, _ = cv2.calcOpticalFlowPyrLK(img0, img1, p0, None, **lk_params)
            # get the flow from this frame to the previous
            p0r, _, _ = cv2.calcOpticalFlowPyrLK(img1, img0, p1, None, **lk_params)
            d = abs(p0-p0r).reshape(-1, 2).max(-1)
            good = d < 1
            new_tracks = []
            for tr, (x, y), good_flag in zip(tracks, p1.reshape(-1, 2), good):
                if not good_flag:
                    continue
                tr.append((x, y))
                if len(tr) > track_length:
                    del tr[0]
                new_tracks.append(tr)
                cv2.circle(overlay, (int(x), int(y)), 2, (0, 255, 0), -1)
            tracks = new_tracks
            cv2.polylines(overlay, [np.int32(tr) for tr in tracks], False, (0, 255, 0))

        if frame_index % detect_interval == 0:
            mask = np.zeros_like(frame_gray)
            x1,y1,x2,y2 = rectMask(mask)
            mask[y1:y2, x1:x2] = 255
            for x, y in [np.int32(tr[-1]) for tr in tracks]:
                cv2.circle(mask, (x, y), 15, 255, -1)
            p = cv2.goodFeaturesToTrack(frame_gray, mask=mask, **feature_params)
            if p is not None:
                a = avg(p)
                moveRect(a)
                for x, y in np.float32(p).reshape(-1, 2):
                    tracks.append([(x, y)])
        
        frame_index += 1
        prev_gray = frame_gray

        

        overlay = drawRect(overlay)
        result_morph = cv2.cvtColor(result_morph, cv2.COLOR_GRAY2BGR)
        result_morph = cv2.drawKeypoints(result_morph, kp, None, color = (0, 255, 0), flags = 0)
        # cv2.rectangle(overlay, (rect["x1"], rect["y1"]), (rect["x2"], rect["y2"]), (0, 255, 0), thickness=-1)

        # cv2.addWeighted(overlay, opacity, img, 1 - opacity, 0, overlay)

        cv2.imshow("output", result_morph)

        if cv2.waitKey(1) == 27:
            break
except KeyboardInterrupt:
    pass

cv2.destroyAllWindows()

# hard 1 and 3