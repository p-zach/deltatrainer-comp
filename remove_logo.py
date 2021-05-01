# Author: Porter Zach
# Python 3.9

import cv2
import numpy as np
import argparse

class App:
    def __init__(self, video, save, scalar):
        self.opacity = .3

        self.resetLabeling()

        self.video = cv2.VideoCapture(video)
        self.ret, self.img = self.video.read()

        self.scalar = scalar

        self.done_analyzing = False

        self.feature_params = dict( maxCorners = 500,
                                    qualityLevel = 0.3,
                                    minDistance = 7,
                                    blockSize = 7 )

        self.lk_params = dict( winSize  = (31, 31),
                               maxLevel = 2,
                               criteria = (cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 30, 0.1))

    #region Initial Labeling

    def scaleMouse(self, x, y):
        """Scales the mouse in case a scalar was specified.

        Args:
            x (int): The X coordinate of the mouse.
            y (int): The Y coordinate of the mouse.

        Returns:
            int, int: The scaled X and Y mouse coordinates.
        """
        return int(x / self.scalar), int(y / self.scalar)

    def resetLabeling(self):
        self.ix, self.iy = 0, 0
        self.drawing = False
        self.rect = { "x1": 0, "y1": 0, "x2": 0, "y2": 0 }
        self.done_drawing = False

        cv2.namedWindow("Window")
        cv2.setMouseCallback("Window", self.handleMouse)

    def handleMouse(self, event, x, y, flags, param):
        """Handles mouse actions. Begins drawing on mouse down, sets label on mouse up.

        Args:
            event (int): The type of mouse event.
            x (int): The X coordinate of the mouse.
            y (int): The Y coordinate of the mouse.
            flags (int): Flags relevant to the mouse action.
            param (obj): Parameters regarding the mouse action.
        """
        x, y = self.scaleMouse(x, y)

        # if the mouse was pressed:
        if event == cv2.EVENT_LBUTTONDOWN:
            # start drawing
            self.drawing = True
            # set initial values
            self.ix, self.iy = x, y
        # if the mouse was moved:
        elif event == cv2.EVENT_MOUSEMOVE:
            # update the rect if drawing
            if self.drawing:
                self.rect["x1"] = min(self.ix, x)
                self.rect["y1"] = min(self.iy, y)
                self.rect["x2"] = max(self.ix, x)
                self.rect["y2"] = max(self.iy, y)
        # if the mouse was released:
        elif event == cv2.EVENT_LBUTTONUP:
            if self.drawing:
                self.done_drawing = True

    def label(self):
        self.resetLabeling()
        overlay = self.img.copy()
        try:
            while not self.done_drawing:
                cv2.rectangle(overlay, (self.rect["x1"], self.rect["y1"]), (self.rect["x2"], self.rect["y2"]), (0, 255, 0), thickness=-1)

                cv2.addWeighted(overlay, self.opacity, self.img, 1 - self.opacity, 0, overlay)

                # resize image if specified
                if self.scalar != 1:
                    overlay = cv2.resize(overlay, 
                        (int(overlay.shape[1] * self.scalar), int(overlay.shape[0] * self.scalar)), 
                        interpolation=cv2.INTER_AREA)
                
                cv2.imshow("Window", overlay)

                k = cv2.waitKey(1) & 0xFF
                # get a new frame if space pressed
                if k == ord(" "):
                    self.ret, self.img = self.video.read()
                    if not self.ret: 
                        break
                    overlay = self.img.copy()
                # exit if escape is pressed
                if k == 27:
                    exit()
        except KeyboardInterrupt:
            print("Editing interrupted. Not saving.")
            cv2.destroyAllWindows()
            self.video.release()
            exit()
        cv2.destroyAllWindows()
    
    def analyze(self):
        track_length = 10
        detect_interval = 1
        tracks = []
        frame_index = 0
        prev_gray = None

        try:
            while True:
                self.ret, self.img = self.video.read()
                if not self.ret:
                    self.done_analyzing = True
                    break
                
                minBlack = np.array([0, 0, 0])
                maxBlack = np.array([180, 255, 75])

                minWhite = np.array([0, 0, 100])
                maxWhite = np.array([180, 50, 255])

                minSkin = np.array([0, 10, 40])
                maxSkin = np.array([35, 255, 255])
                minSkin2 = np.array([170, 10, 40])
                maxSkin2 = np.array([180, 50, 200])

                hsv = cv2.cvtColor(self.img, cv2.COLOR_BGR2HSV)

                mask_black = cv2.inRange(hsv, minBlack, maxBlack)
                mask_white = cv2.inRange(hsv, minWhite, maxWhite)
                mask_skin = cv2.bitwise_or(cv2.inRange(hsv, minSkin, maxSkin), cv2.inRange(hsv, minSkin2, maxSkin2))

                img_black = cv2.bitwise_and(self.img, self.img, mask = mask_black)
                img_white = cv2.bitwise_and(self.img, self.img, mask = mask_white)
                
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

                frame_gray = result_morph
                if prev_gray is None:
                    prev_gray = frame_gray

                if frame_index == 0:
                    mask = np.zeros_like(frame_gray)
                    x1,y1,x2,y2 = self.rectMask(mask)
                    mask[y1:y2, x1:x2] = 255
                    p = cv2.goodFeaturesToTrack(frame_gray, mask=mask, **self.feature_params)
                    if p is not None:
                        for x, y in np.float32(p).reshape(-1, 2):
                            tracks.append([(x, y)])

                overlay = self.img.copy()

                if len(tracks) > 0:
                    # get quick refs to previous and new frames
                    img0, img1 = prev_gray, frame_gray
                    p0 = np.float32([tr[-1] for tr in tracks]).reshape(-1, 1, 2)
                    # get the current frame's optical flow
                    p1, _, _ = cv2.calcOpticalFlowPyrLK(img0, img1, p0, None, **self.lk_params)
                    # get the flow from this frame to the previous
                    p0r, _, _ = cv2.calcOpticalFlowPyrLK(img1, img0, p1, None, **self.lk_params)
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
                    # x1,y1,x2,y2 = rectMask(mask)
                    # mask[y1:y2, x1:x2] = 255
                    for x, y in [np.int32(tr[-1]) for tr in tracks]:
                        # find features within 10px of existing ones
                        cv2.circle(mask, (x, y), 10, 255, -1)
                    p = cv2.goodFeaturesToTrack(frame_gray, mask=mask, **self.feature_params)
                    if p is not None:
                        # a = avg(p)
                        # moveRect(a)
                        for x, y in np.float32(p).reshape(-1, 2):
                            tracks.append([(x, y)])

                if len(tracks) == 0:
                    break
                
                frame_index += 1
                prev_gray = frame_gray

                mask_pts = np.zeros_like(result_morph)
                for x, y in [np.int32(tr[-1]) for tr in tracks]:
                        cv2.circle(mask_pts, (x, y), 10, 255, -1)

                result_morph = cv2.bitwise_and(result_morph, result_morph, mask=mask_pts)

                # imperative
                result_morph = cv2.morphologyEx(result_morph, cv2.MORPH_DILATE, kernel_big)

                # mask result_morph by inv(not_black_or_white)
                black_or_white = cv2.cvtColor(cv2.bitwise_or(img_white, img_black), cv2.COLOR_BGR2GRAY)
                black_or_white = cv2.morphologyEx(black_or_white, cv2.MORPH_DILATE, kernel_small)
                black_or_white = cv2.bitwise_and(black_or_white, black_or_white, mask=cv2.bitwise_not(mask_skin))

                result_morph = cv2.bitwise_and(result_morph, result_morph, mask=black_or_white)
                # remove little spaces in logo from thinking the white is skin
                result_morph = cv2.morphologyEx(result_morph, cv2.MORPH_CLOSE, kernel_5)
                
                img_black_blurred = cv2.GaussianBlur(img_black, (31, 31), cv2.BORDER_DEFAULT)
                inpainted = cv2.inpaint(img_black_blurred, result_morph, 9, cv2.INPAINT_TELEA)
                img_logo_masked = cv2.bitwise_and(self.img, self.img, mask=cv2.bitwise_not(result_morph))
                inpainted = cv2.bitwise_and(inpainted, inpainted, mask=result_morph)
                result = cv2.bitwise_or(inpainted, img_logo_masked)

                # overlay = drawRect(overlay)
                result_morph = cv2.cvtColor(result_morph, cv2.COLOR_GRAY2BGR)
                skin = cv2.bitwise_and(self.img, self.img, mask=mask_skin)
                # result_morph = cv2.drawKeypoints(result_morph, kp, None, color = (0, 255, 0), flags = 0)
                # cv2.rectangle(overlay, (rect["x1"], rect["y1"]), (rect["x2"], rect["y2"]), (0, 255, 0), thickness=-1)

                # cv2.addWeighted(overlay, opacity, img, 1 - opacity, 0, overlay)

                cv2.imshow("output", result)

                # mask the final dilated whitespace img by an inverted skin color mask
                # OR inpaint from an image with only black (!) do this first

                # exit if esc pressed
                if cv2.waitKey(1) == 27:
                    self.done_analyzing = True
                    break
        except KeyboardInterrupt:
            pass
        cv2.destroyAllWindows()

    def getRect(self):
        return min(self.rect["x1"], self.rect["x2"]), min(self.rect["y1"], self.rect["y2"]), max(self.rect["x1"], self.rect["x2"]), max(self.rect["y1"], self.rect["y2"])
    def setRect(self, x1, y1, x2, y2):
        self.rect = { "x1": x1, "y1": y1, "x2": x2, "y2": y2 }

    def avg(self, points):
        ax = 0
        ay = 0
        p = np.int32(points).reshape(-1, 2)
        for x, y in p:
            ax += x
            ay += y
        return (int(ax / p.shape[0]), int(ay / p.shape[0]))

    def moveRect(self, center):
        x1, y1, x2, y2 = self.getRect()
        w = x2 - x1
        h = y2 - y1
        x1 = center[0] - w / 2
        x2 = center[0] + w / 2
        y1 = center[1] - h / 2
        y2 = center[1] + h / 2
        self.setRect(x1, y1, x2, y2)

    def rectMask(self, img):
        r = self.getRect()
        w = img.shape[1]
        h = img.shape[0]
        o = .02
        return int(r[0] - w * o), int(r[1] - h * o), int(r[2] + w * o), int(r[3] + h * o)

    def run(self):
        while not self.done_analyzing:
            self.label()
            self.analyze()

def main(): 
    parser = argparse.ArgumentParser(description="")

    parser.add_argument(action="append", dest="paths", nargs="?", help="The path to the video to remove the logo from and the path to save the edited video to (default: \'<videoPath>_edit.avi\').")
    parser.add_argument("--scalar", action="store", dest="scalar", default=1, type=float, help="The scalar to resize the labeling image by (in case it's too big or small).")

    args = parser.parse_args()

    savePath = args.paths[1] if len(args.paths) > 1 else (args.paths[0] + "_edit.avi")

    App(args.paths[0], savePath, args.scalar).run()

if __name__ == "__main__" :
    main()

# write frames even during the label skip phase