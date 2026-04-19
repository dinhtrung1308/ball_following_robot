#!/usr/bin/env python3
import rospy
import message_filters
import numpy as np
import cv2

from sensor_msgs.msg import Image
from geometry_msgs.msg import Twist
from cv_bridge import CvBridge, CvBridgeError


class BallFollower:
    def __init__(self):
        rospy.init_node('ball_follower', anonymous=True)

        self.setDefaults()

        #in my robot, the topics look like these:
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

        self.sync.registerCallback(self.run_pipeline)
        rospy.spin()

    def setDefaults(self):
        self.desired_dist  = rospy.get_param('~desired_dist', 1.0)
        self.tol_dist      = rospy.get_param('~tol_dist', 0.08)
        self.tol_ang       = rospy.get_param('~tol_ang', 0.03)
        self.p_gain_lin    = rospy.get_param('~p_gain_lin', 0.35)
        self.p_gain_ang    = rospy.get_param('~p_gain_ang', 1.0)
        self.max_speed     = rospy.get_param('~max_speed', 0.4)
        self.max_rot       = rospy.get_param('~max_rot', 0.8)
        self.min_ball_area = rospy.get_param('~min_ball_area', 300)
        self.search_window = rospy.get_param('~search_window', 12)

    def read_rgb_image(self, rgb_msg):
        try:
            frame = self.bridge.imgmsg_to_cv2(rgb_msg, desired_encoding='bgr8')
            return frame
        except CvBridgeError as e:
            rospy.logwarn("RGB bridge error: %s", e)
            return None
        
    def read_depth_image(self, depth_msg):
        #msg encodings
        MSG_16UC1 = '16UC1'
        MSG_32FC1 = '32FC1'

        try:
            if depth_msg.encoding == MSG_16UC1:
                raw = self.bridge.imgmsg_to_cv2(depth_msg, desired_encoding=MSG_16UC1)
                depth_image = raw.astype(np.float32) / 1000.0
            elif depth_msg.encoding == MSG_32FC1:
                depth_image = self.bridge.imgmsg_to_cv2(depth_msg, desired_encoding=MSG_32FC1)
                sample = depth_image[depth_image.shape[0] // 2, depth_image.shape[1] // 2]
                if np.isfinite(sample) and sample > 50.0:
                    depth_image = depth_image / 1000.0
            else:
                depth_image = self.bridge.imgmsg_to_cv2(depth_msg, desired_encoding='passthrough')
                depth_image = depth_image.astype(np.float32)
            return depth_image
        except CvBridgeError as e:
            rospy.logwarn("Depth bridge error: %s", e)
            return None
        
    def color_2_d(self, cx, cy):
        if None in (self.img_width, self.img_height, self.depth_width, self.depth_height):
            return None, None

        dx = int(float(cx) * self.depth_width  / self.img_width)
        dy = int(float(cy) * self.depth_height / self.img_height)
        dx = max(0, min(self.depth_width  - 1, dx))
        dy = max(0, min(self.depth_height - 1, dy))
        return dx, dy

    def depth_at(self, depth_image, cx, cy):
        dx, dy = self.color_2_d(cx, cy)
        if dx is None:
            return None

        r = self.search_window
        h, w = depth_image.shape[:2]
        x0, x1 = max(dx - r, 0), min(dx + r + 1, w)
        y0, y1 = max(dy - r, 0), min(dy + r + 1, h)

        patch = depth_image[y0:y1, x0:x1]

        valid = patch[np.isfinite(patch) & (patch > 0.1) & (patch < 10.0)]
        if valid.size == 0:
            return None

        return float(np.median(valid))
    
    def run_pipeline(self, rgb_msg, depth_msg):
        depth_image = self.read_depth_image(depth_msg)
        frame = self.read_rgb_image(rgb_msg)

        self.img_height,   self.img_width   = frame.shape[:2]
        self.depth_height, self.depth_width = depth_image.shape[:2]

        cx, cy, msk = self.detect_ball(frame)

        rospy.loginfo_throttle(1, "Ball detected: cx=%s cy=%s", cx, cy)
        self.drive(cx, cy, depth_image)

    def detect_ball(self, frame):
        # I used green ball
        hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
        lower_green = np.array([30,  40,  40])
        upper_green = np.array([90, 255, 255])
        msk = cv2.inRange(hsv, lower_green, upper_green)
        k = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (7, 7))
        msk = cv2.morphologyEx(msk, cv2.MORPH_OPEN,  k)
        msk = cv2.morphologyEx(msk, cv2.MORPH_CLOSE, k)
        ct, _ = cv2.findContours(msk, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

        if not ct:
            return None, None, msk

        largest = max(ct, key=cv2.contourArea)
        a    = cv2.contourArea(largest)

        if a < self.min_ball_area:
            return None, None, msk

        M = cv2.moments(largest)
        if M['m00'] == 0:
            return None, None, msk

        cx = int(M['m10'] / M['m00'])
        cy = int(M['m01'] / M['m00'])

        return cx, cy, msk


    def drive(self, cx, cy, depth_image):
        cmd = Twist()

        if cx is None or self.img_width is None:
            self.vel_pub.publish(cmd)
            return

        ang_error = (cx - self.img_width / 2.0) / self.img_width

        if abs(ang_error) > self.tol_ang:
            cmd.angular.z = -ang_error * self.p_gain_ang
            cmd.angular.z = max(-self.max_rot, min(self.max_rot, cmd.angular.z))

        depth = self.depth_at(depth_image, cx, cy)

        if depth is None:
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


if __name__ == '__main__':
    try:
        BallFollower()
    except rospy.ROSInterruptException:
        pass
