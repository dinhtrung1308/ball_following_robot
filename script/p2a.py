#!/usr/bin/env python3
import math, sys, rospy, actionlib
from move_base_msgs.msg import MoveBaseAction, MoveBaseGoal
from geometry_msgs.msg import Quaternion
from tf.transformations import quaternion_from_euler
from actionlib_msgs.msg import GoalStatus

# Initial pose for the robot at L1
LOCATIONS = {
    "L1": {"x": -0.511,  "y":  13.283, "yaw":  0.00},
    "L2": {"x": 0, "y": 18.1, "yaw": 0.00},
    "L3": {"x": 12.261, "y":  18.009, "yaw": 0.00},
}
def print_status(gl, client, name):
    rospy.loginfo("Going to %s ...", name)
    client.send_goal(gl)
    client.wait_for_result()

    if client.get_state() == GoalStatus.SUCCEEDED:
        rospy.loginfo("Reached %s!", name)
    else:
        rospy.logwarn("Failed to reach %s.", name)

def send_goal(client, name):
    loc  = LOCATIONS[name]
    gl = MoveBaseGoal()
    gl.target_pose.header.frame_id = "map"
    gl.target_pose.header.stamp    = rospy.Time.now()
    gl.target_pose.pose.position.x  = loc["x"]
    gl.target_pose.pose.position.y  = loc["y"]
    q = quaternion_from_euler(0, 0, loc["yaw"])
    gl.target_pose.pose.orientation = Quaternion(*q)

    print_status(gl, client, name)

def main():
    rospy.init_node("simple_navigation")
    client = actionlib.SimpleActionClient("move_base", MoveBaseAction)
    client.wait_for_server()

    for stop in ["L2", "L3", "L1"]:
        send_goal(client, stop)
        rospy.sleep(2.0)

if __name__ == "__main__":
    main()
