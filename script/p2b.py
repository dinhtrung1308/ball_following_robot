#!/usr/bin/env python3
import rospy
import message_filters
import numpy as np
import cv2

from sensor_msgs.msg import Image
from geometry_msgs.msg import Twist
from cv_bridge import CvBridge, CvBridgeError


class GreenBallFollower:
    def __init__(self):
        rospy.init_node('green_ball_follower', anonymous=True)

        self.desired_dist  = rospy.get_param('~desired_dist', 1.0)
        self.tol_dist      = rospy.get_param('~tol_dist', 0.08)
        self.tol_ang       = rospy.get_param('~tol_ang', 0.03)

        self.p_gain_lin    = rospy.get_param('~p_gain_lin', 0.35)
        self.p_gain_ang    = rospy.get_param('~p_gain_ang', 1.0)

        self.max_speed     = rospy.get_param('~max_speed', 0.4)
        self.max_rot       = rospy.get_param('~max_rot', 0.8)

        self.min_ball_area = rospy.get_param('~min_ball_area', 300)
        self.search_window = rospy.get_param('~search_window', 12)

        rgb_topic   = rospy.get_param('~rgb_topic',   '/camera/color/image_raw')
        depth_topic = rospy.get_param('~depth_topic', '/camera/depth/image_raw')
        vel_topic   = rospy.get_param('~vel_topic',   '/cmd_vel_mux/input/navi')

        self.bridge = CvBridge()
        self.img_width    = None
        self.img_height   = None
        self.depth_width  = None
        self.depth_height = None

        self.vel_pub = rospy.Publisher(vel_topic, Twist, queue_size=1)

        rgb_sub   = message_filters.Subscriber(rgb_topic,   Image)
        depth_sub = message_filters.Subscriber(depth_topic, Image)

        self.sync = message_filters.ApproximateTimeSynchronizer(
            [rgb_sub, depth_sub],
            queue_size=5,
            slop=0.08
        )
        self.sync.registerCallback(self.sync_callback)

        rospy.loginfo("Green Ball Follower ready.")
        rospy.loginfo("  RGB topic   : %s", rgb_topic)
        rospy.loginfo("  Depth topic : %s", depth_topic)
        rospy.loginfo("  Vel topic   : %s", vel_topic)

        rospy.spin()

    # ------------------------------------------------------------------
    def sync_callback(self, rgb_msg, depth_msg):
        # ---- depth conversion ----------------------------------------
        try:
            if depth_msg.encoding == '16UC1':
                raw = self.bridge.imgmsg_to_cv2(depth_msg, desired_encoding='16UC1')
                depth_image = raw.astype(np.float32) / 1000.0   # mm → m
            elif depth_msg.encoding == '32FC1':
                depth_image = self.bridge.imgmsg_to_cv2(depth_msg, desired_encoding='32FC1')
                # FIX: some 32FC1 streams are actually in mm — detect and correct
                sample = depth_image[depth_image.shape[0] // 2, depth_image.shape[1] // 2]
                if np.isfinite(sample) and sample > 50.0:
                    rospy.logwarn_once(
                        "32FC1 depth looks like millimetres (sample=%.1f) — dividing by 1000", sample)
                    depth_image = depth_image / 1000.0
            else:
                rospy.logwarn_once("Unknown depth encoding: %s", depth_msg.encoding)
                depth_image = self.bridge.imgmsg_to_cv2(depth_msg, desired_encoding='passthrough')
                depth_image = depth_image.astype(np.float32)
        except CvBridgeError as e:
            rospy.logwarn("Depth bridge error: %s", e)
            return

        # ---- RGB conversion ------------------------------------------
        try:
            frame = self.bridge.imgmsg_to_cv2(rgb_msg, desired_encoding='bgr8')
        except CvBridgeError as e:
            rospy.logwarn("RGB bridge error: %s", e)
            return

        self.img_height,   self.img_width   = frame.shape[:2]
        self.depth_height, self.depth_width = depth_image.shape[:2]

        # ---- detect --------------------------------------------------
        cx, cy, mask = self.detect_green_ball(frame)

        rospy.loginfo_throttle(1, "Ball detected: cx=%s cy=%s", cx, cy)

        # ---- debug overlay -------------------------------------------
        debug = frame.copy()
        if cx is not None:
            cv2.circle(debug, (cx, cy), 10, (0, 255, 0), -1)
            cv2.line(
                debug,
                (self.img_width // 2, 0),
                (self.img_width // 2, self.img_height),
                (255, 0, 0), 1
            )

        # ---- drive ---------------------------------------------------
        self.drive(cx, cy, depth_image)

    # ------------------------------------------------------------------
    def detect_green_ball(self, frame):
        hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)

        # FIX: slightly wider HSV range to cope with different lighting
        lower_green = np.array([30,  40,  40])
        upper_green = np.array([90, 255, 255])

        mask = cv2.inRange(hsv, lower_green, upper_green)

        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (7, 7))
        mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN,  kernel)
        mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)

        contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        if not contours:
            rospy.loginfo_throttle(1, "No contours found in green mask")
            return None, None, mask

        largest = max(contours, key=cv2.contourArea)
        area    = cv2.contourArea(largest)
        rospy.loginfo_throttle(1, "Largest contour area: %.1f (min=%d)", area, self.min_ball_area)

        if area < self.min_ball_area:
            return None, None, mask

        M = cv2.moments(largest)
        if M['m00'] == 0:
            return None, None, mask

        cx = int(M['m10'] / M['m00'])
        cy = int(M['m01'] / M['m00'])
        return cx, cy, mask

    # ------------------------------------------------------------------
    def color_to_depth_pixel(self, cx, cy):
        if None in (self.img_width, self.img_height, self.depth_width, self.depth_height):
            return None, None

        dx = int(float(cx) * self.depth_width  / self.img_width)
        dy = int(float(cy) * self.depth_height / self.img_height)
        dx = max(0, min(self.depth_width  - 1, dx))
        dy = max(0, min(self.depth_height - 1, dy))
        return dx, dy

    # ------------------------------------------------------------------
    def get_depth_at(self, depth_image, cx, cy):
        dx, dy = self.color_to_depth_pixel(cx, cy)
        if dx is None:
            return None

        r = self.search_window
        h, w = depth_image.shape[:2]
        x0, x1 = max(dx - r, 0), min(dx + r + 1, w)
        y0, y1 = max(dy - r, 0), min(dy + r + 1, h)

        patch = depth_image[y0:y1, x0:x1]

        # FIX: log raw patch stats so encoding issues are immediately visible
        if patch.size > 0:
            finite_vals = patch[np.isfinite(patch)]
            if finite_vals.size > 0:
                rospy.loginfo_throttle(1,
                    "Depth patch  min=%.3f  max=%.3f  finite_pixels=%d",
                    finite_vals.min(), finite_vals.max(), finite_vals.size)
            else:
                rospy.logwarn_throttle(1, "Depth patch has NO finite values")

        valid = patch[np.isfinite(patch) & (patch > 0.1) & (patch < 10.0)]
        if valid.size == 0:
            return None

        return float(np.median(valid))

    # ------------------------------------------------------------------
    def drive(self, cx, cy, depth_image):
        cmd = Twist()

        if cx is None or self.img_width is None:
            rospy.loginfo_throttle(2, "No green ball detected — stopping.")
            self.vel_pub.publish(cmd)
            return

        # FIX: normalise error to [-0.5, +0.5]; positive → ball is to the right
        # Robot must turn LEFT (positive angular.z in ROS) when ball is to the right
        ang_error = (cx - self.img_width / 2.0) / self.img_width

        if abs(ang_error) > self.tol_ang:
            # FIX: negate so robot turns toward the ball
            cmd.angular.z = -ang_error * self.p_gain_ang
            cmd.angular.z = max(-self.max_rot, min(self.max_rot, cmd.angular.z))

        depth = self.get_depth_at(depth_image, cx, cy)
        rospy.loginfo_throttle(1, "Depth at ball: %s", depth)

        if depth is None:
            # FIX: creep forward instead of freezing so depth can be acquired
            rospy.logwarn_throttle(2, "No valid depth — creeping forward.")
            cmd.linear.x = 0.05
        else:
            dist_error = depth - self.desired_dist

            if abs(dist_error) > self.tol_dist:
                cmd.linear.x = dist_error * self.p_gain_lin
                cmd.linear.x = max(-self.max_speed, min(self.max_speed, cmd.linear.x))

            rospy.loginfo(
                "depth: %.2f m | dist_err: %+.2f | ang_err: %+.3f | lin: %+.2f | ang: %+.2f",
                depth, dist_error, ang_error, cmd.linear.x, cmd.angular.z
            )

        self.vel_pub.publish(cmd)


# ----------------------------------------------------------------------
if __name__ == '__main__':
    try:
        GreenBallFollower()
    except rospy.ROSInterruptException:
        pass
