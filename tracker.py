import cv2
import numpy as np

MIN_MATCHES = 10
RATIO = 0.75


def box_is_ok(corners, height, width):
    if corners is None or not np.isfinite(corners).all():
        return False
    points = corners.reshape(-1, 2)
    if points[:, 0].min() < -8 or points[:, 1].min() < -8:
        return False
    if points[:, 0].max() > width + 8 or points[:, 1].max() > height + 8:
        return False
    for index in range(4):
        first = points[index]
        second = points[(index + 1) % 4]
        side = float((first[0] - second[0]) ** 2 + (first[1] - second[1]) ** 2)
        if side < 20 * 20:
            return False
    wide = float(points[:, 0].max() - points[:, 0].min())
    tall = float(points[:, 1].max() - points[:, 1].min())
    return wide * tall < width * height * 1.3


class ObjectTracker:
    def __init__(self, template_bgr):
        gray = cv2.cvtColor(template_bgr, cv2.COLOR_BGR2GRAY)
        height, width = gray.shape[:2]
        self.height = height
        self.width = width
        self.sift = cv2.SIFT_create(nfeatures=500)
        self.matcher = cv2.BFMatcher()
        self.keypoints, self.descriptors = self.sift.detectAndCompute(gray, None)
        if self.descriptors is None or len(self.keypoints) < MIN_MATCHES:
            raise RuntimeError("на первом кадре слишком мало ключевых точек")
        self.corners = np.float32(
            [
                [0, 0],
                [width - 1, 0],
                [width - 1, height - 1],
                [0, height - 1],
            ]
        ).reshape(-1, 1, 2)
        self.previous_box = None
        self.previous_gray = None

    def locate(self, frame_bgr):
        gray = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2GRAY)
        box = self._match_sift(gray)
        # На дальнем кадре своих точек мало, рамку двигают углы вокруг прошлого положения.
        if box is None:
            box = self._follow_motion(gray)
        if box is None:
            return None
        self.previous_box = box
        self.previous_gray = gray
        return box

    def _match_sift(self, gray):
        keypoints, descriptors = self.sift.detectAndCompute(gray, None)
        if descriptors is None or len(keypoints) < 2:
            return None
        pairs = self.matcher.knnMatch(self.descriptors, descriptors, k=2)
        good = []
        for pair in pairs:
            if len(pair) < 2:
                continue
            best, second = pair
            if best.distance < RATIO * second.distance:
                good.append(best)
        if len(good) < MIN_MATCHES:
            return None

        source = np.float32(
            [self.keypoints[match.queryIdx].pt for match in good]
        ).reshape(-1, 1, 2)
        destination = np.float32(
            [keypoints[match.trainIdx].pt for match in good]
        ).reshape(-1, 1, 2)
        homography, mask = cv2.findHomography(source, destination, cv2.RANSAC, 5.0)
        if homography is None or mask is None or int(mask.sum()) < MIN_MATCHES:
            return None
        corners = cv2.perspectiveTransform(self.corners, homography)
        if not box_is_ok(corners, self.height, self.width):
            return None
        return corners

    def _follow_motion(self, gray):
        if self.previous_box is None or self.previous_gray is None:
            return None
        points = self.previous_box.reshape(-1, 2)
        x0, y0 = points.min(axis=0)
        x1, y1 = points.max(axis=0)
        pad_x = 0.8 * (x1 - x0)
        pad_y = 0.8 * (y1 - y0)
        left = int(max(0, np.floor(x0 - pad_x)))
        top = int(max(0, np.floor(y0 - pad_y)))
        right = int(min(self.width, np.ceil(x1 + pad_x)))
        bottom = int(min(self.height, np.ceil(y1 + pad_y)))
        if right - left < 20 or bottom - top < 20:
            return None
        mask = np.zeros((self.height, self.width), dtype=np.uint8)
        mask[top:bottom, left:right] = 255
        before = cv2.goodFeaturesToTrack(
            self.previous_gray, 200, 0.01, 6, mask=mask, blockSize=5
        )
        if before is None or len(before) < 12:
            return None
        after, status, _error = cv2.calcOpticalFlowPyrLK(
            self.previous_gray,
            gray,
            before,
            None,
            winSize=(21, 21),
            maxLevel=3,
        )
        if status is None:
            return None
        kept = status.ravel().astype(bool)
        if int(kept.sum()) < 12:
            return None
        homography, inliers = cv2.findHomography(
            before[kept], after[kept], cv2.RANSAC, 3.0
        )
        if homography is None or inliers is None or int(inliers.sum()) < 12:
            return None
        box = cv2.perspectiveTransform(self.previous_box, homography)
        if not box_is_ok(box, self.height, self.width):
            return None
        old_area = float((x1 - x0) * (y1 - y0))
        new_points = box.reshape(-1, 2)
        new_area = float(
            (new_points[:, 0].max() - new_points[:, 0].min())
            * (new_points[:, 1].max() - new_points[:, 1].min())
        )
        if old_area <= 0 or not (0.75 <= new_area / old_area <= 1.35):
            return None
        return box


def draw_box(frame_bgr, corners, label):
    image = frame_bgr.copy()
    points = np.int32(corners.reshape(-1, 1, 2))
    cv2.polylines(image, [points], True, (40, 220, 40), 2, cv2.LINE_AA)
    x = int(points[:, 0, 0].min())
    y = int(points[:, 0, 1].min()) - 10
    cv2.putText(
        image,
        label,
        (max(x, 8), max(y, 24)),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.7,
        (40, 220, 40),
        2,
        cv2.LINE_AA,
    )
    return image
