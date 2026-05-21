import numpy as np
import math


class Robot:
    def __init__(self, x_init, y_init, theta_init, wheelbase=2.0, dt=0.1):
        self.x = x_init
        self.y = y_init
        self.theta = theta_init

        self.odom_x = x_init
        self.odom_y = y_init
        self.odom_theta = theta_init

        self.L = wheelbase
        self.dt = dt

        self.v_noise_std = 0.05
        self.steer_noise_std = 0.02

    def move(self, v, steering_angle):
        # Gerçek Hareket
        self.x += v * math.cos(self.theta) * self.dt
        self.y += v * math.sin(self.theta) * self.dt
        self.theta += (v / self.L) * math.tan(steering_angle) * self.dt
        self.theta = math.atan2(math.sin(self.theta), math.cos(self.theta))

        # Dead Reckoning (Odometri) Hareketi
        v_measured = v + np.random.normal(0, self.v_noise_std * abs(v))
        steer_measured = steering_angle + np.random.normal(0, self.steer_noise_std)

        self.odom_x += v_measured * math.cos(self.odom_theta) * self.dt
        self.odom_y += v_measured * math.sin(self.odom_theta) * self.dt
        self.odom_theta += (v_measured / self.L) * math.tan(steer_measured) * self.dt
        self.odom_theta = math.atan2(math.sin(self.odom_theta), math.cos(self.odom_theta))

        return self.x, self.y, self.theta

    def pure_pursuit_control(self, target_x, target_y):
        dx = target_x - self.x
        dy = target_y - self.y
        distance = math.hypot(dx, dy)

        target_angle = math.atan2(dy, dx)
        alpha = target_angle - self.theta
        alpha = math.atan2(math.sin(alpha), math.cos(alpha))

        max_steer = math.pi / 4.0
        steering_angle = math.atan2(2.0 * self.L * math.sin(alpha), distance)
        steering_angle = max(min(steering_angle, max_steer), -max_steer)

        v = 5.0 if distance > 2.0 else distance * 2.5
        return v, steering_angle

    def get_true_pose(self):
        return np.array([self.x, self.y, self.theta])

    def get_odometry_pose(self):
        return np.array([self.odom_x, self.odom_y, self.odom_theta])