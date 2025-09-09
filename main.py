import pygame
import pymunk
import pymunk.pygame_util
import math
import numpy as np
from scipy import linalg
from scipy.optimize import minimize

# 初始化Pygame
pygame.init()
WIDTH, HEIGHT = 1000, 700
screen = pygame.display.set_mode((WIDTH, HEIGHT))
pygame.display.set_caption("PID/LQR/MPC控制倒立摆调参系统(驰史系统)")

# 尝试加载中文字体，如果失败则使用默认字体
try:
    # 你可以根据系统更改字体路径
    chinese_font_path = "C:/Windows/Fonts/simfang.ttf"
    font = pygame.font.Font(chinese_font_path, 15)
    small_font = pygame.font.Font(chinese_font_path, 15)
    chinese_isfont = True
except:
    print("无法加载中文字体，使用默认字体")
    font = pygame.font.Font(None, 15)
    small_font = pygame.font.Font(None, 15)

clock = pygame.time.Clock()
# 创建物理空间
space = pymunk.Space()
space.gravity = (0, 981)  # 重力加速度（向下为正）


# ========== 初始参数 ==========
init_kp = 3300.0
init_ki = 50.0
init_kd = 400.0

# 控制模式
control_mode = "LQR"  # "MANUAL", "PID", "LQR", 或 "MPC"
use_position_control = False  # 是否使用位置控制（仅PID模式）
is_swing_up_phase = True  # 是否处于甩起阶段
auto_swing_up = False  # 是否自动甩起

# 目标位置管理
target_position = 0  # 世界坐标中的目标位置
follow_mode = "SMOOTH"  # "SMOOTH" 或 "INSTANT" - 相机跟随模式

initial_angle = math.radians(45)  # 初始角度位置

# 主循环
running = True
dt = 1 / 60.0
trail_points = []
max_trail_points = 200

# 参数调节速度
param_adjust_speeds = {
    "Swing-up": {"K_energy": 0.5, "K_damp": 0.1, "Switch angle": math.radians(1)},
    "PID": {"Kp": 10.0, "Ki": 1.0, "Kd": 10.0},
    "LQR": {"Q": 1.0, "R": 0.01},
    "MPC": {"Q": 1.0, "R": 0.01, "Horizon": 1}
}

# ========== 相机系统 ==========
class Camera:
    def __init__(self, width, height):
        self.width = width
        self.height = height
        self.x = 0  # 相机中心的世界坐标x
        self.y = 0  # 相机中心的世界坐标y
        self.target_x = 0
        self.target_y = 0
        self.smoothing = 0.1  # 相机平滑移动系数

    def update(self, target_x, target_y=None):
        """更新相机位置，平滑跟随目标"""
        self.target_x = target_x
        if target_y is not None:
            self.target_y = target_y

        # 平滑移动
        self.x += (self.target_x - self.x) * self.smoothing
        self.y += (self.target_y - self.y) * self.smoothing

    def world_to_screen(self, world_x, world_y):
        """将世界坐标转换为屏幕坐标"""
        screen_x = world_x - self.x + self.width // 2
        screen_y = world_y - self.y + self.height // 2
        return int(screen_x), int(screen_y)

    def screen_to_world(self, screen_x, screen_y):
        """将屏幕坐标转换为世界坐标"""
        world_x = screen_x + self.x - self.width // 2
        world_y = screen_y + self.y - self.height // 2
        return world_x, world_y


# 创建相机
camera = Camera(WIDTH, HEIGHT)

# 导轨参数（世界坐标）
rail_y = HEIGHT // 2  # 导轨在世界中的Y坐标
initial_x = 0  # 初始位置在世界原点

# 系统物理参数（用于LQR和MPC计算）
M = 2.0  # 滑块质量 (kg)
m = 1.0  # 摆球质量 (kg)
l = 0.15  # 摆杆长度 (m) - 200像素约等于0.2m
g = 9.81  # 重力加速度 (m/s^2)
b = 0.1  # 阻尼系数

# 创建滑块
slider_mass = M
slider_radius = 15
slider_moment = pymunk.moment_for_circle(slider_mass, 0, slider_radius)
slider_body = pymunk.Body(slider_mass, slider_moment)
slider_body.position = initial_x, rail_y
slider_shape = pymunk.Circle(slider_body, slider_radius)
slider_shape.friction = 0.3
slider_shape.color = (255, 0, 0, 255)
space.add(slider_body, slider_shape)

# 限制滑块只能在Y轴上的rail_y位置移动（使用约束而不是固定导轨）
# 使用PivotJoint固定Y坐标
static_body = space.static_body
# 创建一个沿着滑块移动的锚点
groove_joint = pymunk.GrooveJoint(
    static_body,
    slider_body,
    (-10000, rail_y),  # 使用非常长的导轨
    (10000, rail_y),
    (0, 0)
)
space.add(groove_joint)

# 创建摆球
pendulum_mass = m
pendulum_radius = 20
pendulum_length = l * 1000
pendulum_moment = pymunk.moment_for_circle(pendulum_mass, 0, pendulum_radius)
pendulum_body = pymunk.Body(pendulum_mass, pendulum_moment)
# 初始位置：下垂位置（准备甩起）
pendulum_body.position = (
    slider_body.position.x + pendulum_length * math.sin(initial_angle),
    slider_body.position.y - pendulum_length * math.cos(initial_angle)
)
pendulum_shape = pymunk.Circle(pendulum_body, pendulum_radius)
pendulum_shape.friction = 0.3
pendulum_shape.color = (0, 0, 255, 255)
space.add(pendulum_body, pendulum_shape)

# 创建连接
pin_joint = pymunk.PinJoint(slider_body, pendulum_body, (0, 0), (0, 0))
pin_joint.distance = pendulum_length
space.add(pin_joint)


# ========== 绘制函数（使用相机坐标） ==========
def draw_infinite_rail(surface, camera):
    """绘制无限导轨（只绘制可见部分）"""
    # 计算可见范围内的导轨
    left_edge = camera.x - camera.width // 2 - 100
    right_edge = camera.x + camera.width // 2 + 100

    # 绘制主导轨线
    start_screen = camera.world_to_screen(left_edge, rail_y)
    end_screen = camera.world_to_screen(right_edge, rail_y)
    pygame.draw.line(surface, (100, 100, 100), start_screen, end_screen, 5)

    # 绘制刻度标记（每100像素一个）
    mark_spacing = 100
    first_mark = int(left_edge / mark_spacing) * mark_spacing

    for x in range(first_mark, int(right_edge) + mark_spacing, mark_spacing):
        mark_screen_x, mark_screen_y = camera.world_to_screen(x, rail_y)
        # 主刻度
        if x % 500 == 0:
            pygame.draw.line(surface, (50, 50, 50),
                             (mark_screen_x, mark_screen_y - 15),
                             (mark_screen_x, mark_screen_y + 15), 2)
            # 显示坐标值
            coord_text = small_font.render(f"{x}", True, (100, 100, 100))
            surface.blit(coord_text, (mark_screen_x - 20, mark_screen_y + 20))
        else:
            # 小刻度
            pygame.draw.line(surface, (150, 150, 150),
                             (mark_screen_x, mark_screen_y - 5),
                             (mark_screen_x, mark_screen_y + 5), 1)

    # 绘制原点标记
    origin_screen_x, origin_screen_y = camera.world_to_screen(0, rail_y)
    if abs(origin_screen_x - camera.width // 2) < camera.width // 2:  # 如果原点在屏幕内
        pygame.draw.circle(surface, (255, 0, 0), (origin_screen_x, origin_screen_y), 5)
        origin_text = small_font.render("原点" if chinese_isfont else "Origin", True, (255, 0, 0))
        surface.blit(origin_text, (origin_screen_x - 20, origin_screen_y - 30))


def draw_objects(surface, camera):
    """绘制所有物理对象"""
    # 获取屏幕坐标
    slider_screen = camera.world_to_screen(slider_body.position.x, slider_body.position.y)
    pendulum_screen = camera.world_to_screen(pendulum_body.position.x, pendulum_body.position.y)

    # 绘制连接杆
    pygame.draw.line(surface, (50, 50, 50), slider_screen, pendulum_screen, 3)

    # 绘制滑块
    pygame.draw.circle(surface, (255, 100, 100), slider_screen, slider_radius)
    pygame.draw.circle(surface, (255, 50, 50), slider_screen, slider_radius, 3)

    # 绘制摆球
    pygame.draw.circle(surface, (100, 100, 255), pendulum_screen, pendulum_radius)
    pygame.draw.circle(surface, (50, 50, 255), pendulum_screen, pendulum_radius, 3)


def draw_trail(surface, camera, trail_points):
    """绘制轨迹"""
    if len(trail_points) > 1:
        screen_points = []
        for world_point in trail_points:
            screen_point = camera.world_to_screen(world_point[0], world_point[1])
            screen_points.append(screen_point)

        for i in range(1, len(screen_points)):
            alpha = int(255 * (i / len(screen_points)))
            color = (0, 100, 200)
            pygame.draw.line(surface, color, screen_points[i - 1], screen_points[i], 2)


def draw_mpc_prediction(surface, camera, mpc_controller, current_world_x):
    """绘制MPC预测轨迹（使用相机坐标）"""
    if len(mpc_controller.prediction_history) > 0:
        for i in range(min(10, len(mpc_controller.prediction_history))):
            state = mpc_controller.prediction_history[i]

            # state[0]是相对于目标位置的偏移（米），需要转换为像素并加上目标位置
            # 注意：state[0]是相对于目标位置的偏移，不是相对于当前位置
            x_offset_meters = state[0]  # 相对于目标位置的偏移（米）
            x_offset_pixels = x_offset_meters * 1000  # 转换为像素

            # 预测的滑块位置（世界坐标） = 目标位置 + 偏移
            pred_slider_x = mpc_controller.target_x + x_offset_pixels

            # 预测的摆球角度
            theta = state[2]

            # 计算预测的摆球位置（世界坐标）
            pred_world_x = pred_slider_x + pendulum_length * math.sin(theta)
            pred_world_y = rail_y - pendulum_length * math.cos(theta)

            # 转换到屏幕坐标
            pred_screen = camera.world_to_screen(pred_world_x, pred_world_y)

            # 绘制预测轨迹点
            radius = max(2, 8 - i // 2)  # 渐变的半径
            color = (255, 100 + i * 10, 100 + i * 10)  # 渐变的颜色
            pygame.draw.circle(surface, color, pred_screen, radius)

            # 绘制连接线（可选）
            if i > 0:
                prev_state = mpc_controller.prediction_history[i - 1]
                prev_x_offset_pixels = prev_state[0] * 1000
                prev_slider_x = mpc_controller.target_x + prev_x_offset_pixels
                prev_theta = prev_state[2]
                prev_world_x = prev_slider_x + pendulum_length * math.sin(prev_theta)
                prev_world_y = rail_y - pendulum_length * math.cos(prev_theta)
                prev_screen = camera.world_to_screen(prev_world_x, prev_world_y)
                pygame.draw.line(surface, (255, 200, 200), prev_screen, pred_screen, 1)


class EnergySwingUpController:
    """基于能量的甩起控制器"""

    def __init__(self, M, m, l, g):
        """
        初始化甩起控制器
        M: 滑块质量
        m: 摆球质量
        l: 摆杆长度
        g: 重力加速度
        """
        self.M = M
        self.m = m
        self.l = l
        self.g = g

        # 目标能量（摆在垂直向上位置时的势能）
        self.E_target = m * g * l

        # 控制增益
        self.k_energy = 8.0  # 能量控制增益
        self.k_damping = 0.5  # 速度阻尼增益

        # 切换阈值
        self.switch_angle = math.radians(60)  # 当角度小于30度时切换到稳定控制

        # 状态标志
        self.is_active = False
        self.swing_up_complete = False

        # 控制历史记录
        self.energy_history = []
        self.output_history = []
        self.max_history = 300

    def calculate_energy(self, theta, theta_dot):
        """
        计算摆的总能量
        theta: 摆角（弧度，垂直向上为0）
        theta_dot: 角速度（弧度/秒）
        """
        # 动能
        KE = 0.5 * self.m * (self.l * theta_dot) ** 2

        # 势能（以摆的最低点为零势能参考点）
        PE = self.m * self.g * self.l * (1 + math.cos(theta))

        # 总能量
        total_energy = KE + PE

        return total_energy

    def update(self, state):
        """
        根据状态计算甩起控制输出
        state: [x, x_dot, theta, theta_dot]
        """
        x, x_dot, theta, theta_dot = state

        # 计算当前能量
        current_energy = self.calculate_energy(theta, theta_dot)

        # 能量误差
        energy_error = current_energy - self.E_target

        # 基于能量的控制策略
        if abs(theta) < self.switch_angle:
            # 接近垂直位置，准备切换
            self.swing_up_complete = True
            control_output = 0
        else:
            # 甩起控制
            sign = np.sign(theta_dot * math.cos(theta))

            # 基本能量控制
            control_output = self.k_energy * sign * energy_error

            # 添加速度阻尼，防止滑块速度过大
            control_output -= self.k_damping * x_dot

            # 注意：移除了位置限制，因为现在是无限导轨

        # 转换单位（从牛顿到像素空间的力）
        control_output = control_output * 1000

        # 更新历史记录
        self.energy_history.append(current_energy)
        self.output_history.append(control_output)

        if len(self.energy_history) > self.max_history:
            self.energy_history.pop(0)
            self.output_history.pop(0)

        return control_output

    def should_switch(self, theta):
        """判断是否应该切换到稳定控制"""
        return abs(theta) < self.switch_angle and self.swing_up_complete

    def reset(self):
        """重置控制器状态"""
        self.swing_up_complete = False
        self.energy_history.clear()
        self.output_history.clear()

    def adjust_gain(self, param_type, delta):
        """调整控制增益"""
        if param_type == 0:  # 能量增益
            self.k_energy = max(0.1, self.k_energy + delta)
        elif param_type == 1:  # 阻尼增益
            self.k_damping = max(0.0, self.k_damping + delta)
        elif param_type == 2:  # 切换角度阈值
            self.switch_angle = max(math.radians(10), min(math.radians(60), self.switch_angle + delta))


class PIDController:
    """PID控制器类"""

    def __init__(self, kp=0.0, ki=0.0, kd=0.0):
        self.kp = kp  # 比例系数
        self.ki = ki  # 积分系数
        self.kd = kd  # 微分系数

        self.integral = 0.0  # 积分项
        self.prev_error = 0.0  # 上一次误差

        # 积分项限制（防止积分饱和）
        self.integral_limit = 1000.0

        # 控制历史记录（用于绘图）
        self.error_history = []
        self.output_history = []
        self.max_history = 300

    def update(self, error, dt):
        """更新PID控制器并返回控制输出"""
        # 计算积分项
        self.integral += error * dt
        # 限制积分项
        self.integral = max(-self.integral_limit, min(self.integral_limit, self.integral))

        # 计算微分项
        if dt > 0:
            derivative = (error - self.prev_error) / dt
        else:
            derivative = 0

        # PID输出
        output = self.kp * error + self.ki * self.integral + self.kd * derivative

        # 更新历史记录
        self.error_history.append(error)
        self.output_history.append(output)
        if len(self.error_history) > self.max_history:
            self.error_history.pop(0)
            self.output_history.pop(0)

        self.prev_error = error
        return output

    def reset(self):
        """重置控制器状态"""
        self.integral = 0.0
        self.prev_error = 0.0
        self.error_history.clear()
        self.output_history.clear()


class LQRController:
    """LQR控制器类"""

    def __init__(self, M, m, l, g, b):
        """
        初始化LQR控制器
        M: 滑块质量
        m: 摆球质量
        l: 摆杆长度
        g: 重力加速度
        b: 阻尼系数
        """
        self.M = M
        self.m = m
        self.l = l
        self.g = g
        self.b = b
        self.target_x = 0  # 目标位置（世界坐标）

        # 计算线性化系统矩阵
        self.compute_system_matrices()

        # 设置默认的Q和R矩阵
        self.Q = np.diag([900.0, 10.0, 100.0, 100.0])  # 状态权重矩阵
        self.R = np.array([[0.01]])  # 控制输入权重矩阵

        # 计算LQR增益
        self.compute_lqr_gain()

        # 控制历史记录
        self.state_history = []
        self.output_history = []
        self.max_history = 300

    def compute_system_matrices(self):
        """计算线性化的状态空间矩阵 A 和 B"""
        # 在平衡点（θ=0）附近线性化
        # 状态向量: [x, x_dot, theta, theta_dot]
        # 控制输入: u (力)

        # A矩阵
        self.A = np.array([
            [0, 1, 0, 0],
            [0, -self.b / self.M, -self.m * self.g / self.M, 0],
            [0, 0, 0, 1],
            [0, self.b / (self.M * self.l), (self.M + self.m) * self.g / (self.M * self.l), 0]
        ])

        # B矩阵
        self.B = np.array([
            [0],
            [1 / self.M],
            [0],
            [-1 / (self.M * self.l)]
        ])

    def compute_lqr_gain(self):
        """计算LQR增益矩阵K"""
        try:
            # 求解代数Riccati方程
            P = linalg.solve_continuous_are(self.A, self.B, self.Q, self.R)

            # 计算增益矩阵 K = R^(-1) * B^T * P
            self.K = np.linalg.inv(self.R) @ self.B.T @ P

            # 将增益矩阵转换为一维数组以便使用
            self.K = self.K.flatten()

            return True
        except:
            print("无法计算LQR增益，使用默认值")
            self.K = np.array([10.0, 20.0, 300.0, 100.0])
            return False

    def set_target_position(self, world_x):
        """设置目标位置（世界坐标）"""
        self.target_x = world_x

    def update(self, state, current_world_x):
        """
        根据状态计算控制输出
        state: [x, x_dot, theta, theta_dot] （x是相对位置）
        current_world_x: 当前滑块的世界坐标
        """
        # 状态反馈控制律: u = -K * x
        state_array = np.array(state)

        # 计算相对于目标位置的偏移
        state_array[0] = (current_world_x - self.target_x) / 1000.0  # 转换为米

        # 计算控制输出
        control_output = -np.dot(self.K, state_array)

        # 转换单位（从牛顿到像素空间的力）
        control_output = control_output * 1000  # 放大因子

        # 更新历史记录
        self.state_history.append(state_array)
        self.output_history.append(control_output)
        if len(self.state_history) > self.max_history:
            self.state_history.pop(0)
            self.output_history.pop(0)

        return control_output

    def reset(self):
        """重置控制器状态"""
        self.state_history.clear()
        self.output_history.clear()

    def adjust_q_weight(self, index, delta):
        """调整Q矩阵的权重"""
        self.Q[index, index] = max(0.01, self.Q[index, index] + delta)
        self.compute_lqr_gain()

    def adjust_r_weight(self, delta):
        """调整R矩阵的权重"""
        self.R[0, 0] = max(0.001, self.R[0, 0] + delta)
        self.compute_lqr_gain()


class MPCController:
    """MPC控制器类"""

    def __init__(self, M, m, l, g, b, dt=0.01):
        """
        初始化MPC控制器
        M: 滑块质量
        m: 摆球质量
        l: 摆杆长度
        g: 重力加速度
        b: 阻尼系数
        dt: 采样时间
        """
        self.M = M
        self.m = m
        self.l = l
        self.g = g
        self.b = b
        self.dt = dt
        self.target_x = 0  # 目标位置（世界坐标）

        # MPC参数
        self.N = 20  # 预测时域
        self.nu = 1  # 控制输入维度
        self.nx = 4  # 状态维度

        # 权重矩阵
        self.Q = np.diag([900.0, 1.0, 100.0, 10.0])  # 状态权重
        self.R = np.array([[0.1]])  # 控制权重
        self.Qf = self.Q * 10  # 终端状态权重

        # 控制约束
        self.u_max = 5.0  # 最大控制力 (N)
        self.u_min = -5.0  # 最小控制力 (N)

        # 初始化线性化模型
        self.compute_discrete_model()

        # 控制历史记录
        self.state_history = []
        self.output_history = []
        self.prediction_history = []
        self.max_history = 300

        # 优化器的初始猜测
        self.u_prev = np.zeros(self.N)

    def compute_discrete_model(self):
        """计算离散化的线性模型"""
        # 连续时间线性化模型 (在平衡点附近)
        A_cont = np.array([
            [0, 1, 0, 0],
            [0, -self.b / self.M, -self.m * self.g / self.M, 0],
            [0, 0, 0, 1],
            [0, self.b / (self.M * self.l), (self.M + self.m) * self.g / (self.M * self.l), 0]
        ])

        B_cont = np.array([
            [0],
            [1 / self.M],
            [0],
            [-1 / (self.M * self.l)]
        ])

        # 使用前向欧拉法离散化
        self.Ad = np.eye(4) + A_cont * self.dt
        self.Bd = B_cont * self.dt

    def set_target_position(self, world_x):
        """设置目标位置（世界坐标）"""
        self.target_x = world_x

    def predict_trajectory(self, x0, u_sequence):
        """预测给定控制序列下的状态轨迹"""
        trajectory = [x0]
        x = x0.copy()

        for i in range(self.N):
            # 线性预测模型
            x_next = self.Ad @ x + self.Bd.flatten() * u_sequence[i]
            trajectory.append(x_next)
            x = x_next

        return np.array(trajectory)

    def cost_function(self, u_sequence, x0):
        """计算MPC代价函数"""
        # 预测轨迹
        trajectory = self.predict_trajectory(x0, u_sequence)

        # 计算代价
        cost = 0

        # 状态代价
        for i in range(self.N):
            x = trajectory[i + 1]
            cost += x.T @ self.Q @ x

        # 终端代价
        cost += trajectory[-1].T @ self.Qf @ trajectory[-1]

        # 控制代价
        for i in range(self.N):
            cost += self.R[0, 0] * u_sequence[i] ** 2

        # 控制变化率惩罚（使控制更平滑）
        for i in range(1, self.N):
            cost += 0.01 * (u_sequence[i] - u_sequence[i - 1]) ** 2

        return cost

    def update(self, state, current_world_x):
        """
        根据当前状态计算MPC控制输出
        state: [x, x_dot, theta, theta_dot]
        current_world_x: 当前滑块的世界坐标
        """
        # 转换状态
        x0 = np.array(state)
        x0[0] = (current_world_x - self.target_x) / 1000.0  # 相对于目标的偏移

        # 优化控制序列
        bounds = [(self.u_min, self.u_max)] * self.N

        # 使用上一次的解作为初始猜测（温启动）
        initial_guess = np.roll(self.u_prev, -1)
        initial_guess[-1] = 0

        # 优化
        result = minimize(
            fun=lambda u: self.cost_function(u, x0),
            x0=initial_guess,
            method='L-BFGS-B',
            bounds=bounds,
            options={'maxiter': 50, 'ftol': 1e-4}
        )

        # 获取优化后的控制序列
        u_optimal = result.x

        # 只使用第一个控制输入（滚动时域）
        control_output = u_optimal[0]

        # 保存优化结果用于下次温启动
        self.u_prev = u_optimal

        # 转换单位（从牛顿到像素空间的力）
        control_output = control_output * 1000

        # 保存预测轨迹（用于可视化）
        predicted_trajectory = self.predict_trajectory(x0, u_optimal)

        # 更新历史记录
        self.state_history.append(x0)
        self.output_history.append(control_output)
        self.prediction_history = predicted_trajectory

        if len(self.state_history) > self.max_history:
            self.state_history.pop(0)
            self.output_history.pop(0)

        return control_output

    def reset(self):
        """重置控制器状态"""
        self.state_history.clear()
        self.output_history.clear()
        self.prediction_history = []
        self.u_prev = np.zeros(self.N)

    def adjust_q_weight(self, index, delta):
        """调整Q矩阵的权重"""
        self.Q[index, index] = max(0.01, self.Q[index, index] + delta)

    def adjust_r_weight(self, delta):
        """调整R矩阵的权重"""
        self.R[0, 0] = max(0.001, self.R[0, 0] + delta)

    def adjust_horizon(self, delta):
        """调整预测时域"""
        new_N = max(5, min(50, self.N + delta))
        if new_N != self.N:
            self.N = new_N
            self.u_prev = np.zeros(self.N)


def get_pendulum_angle():
    """获取摆杆角度（弧度），垂直向上为0"""
    dx = pendulum_body.position.x - slider_body.position.x
    dy = pendulum_body.position.y - slider_body.position.y
    angle = math.atan2(dx, -dy)
    return angle


def get_pendulum_angular_velocity():
    """估算摆杆角速度"""
    # 使用摆球和滑块的相对速度来估算角速度
    relative_vel_x = pendulum_body.velocity.x - slider_body.velocity.x
    relative_vel_y = pendulum_body.velocity.y - slider_body.velocity.y

    # 获取摆杆方向的单位向量（垂直于摆杆）
    angle = get_pendulum_angle()
    tangent_x = math.cos(angle)
    tangent_y = math.sin(angle)

    # 计算切向速度
    tangent_vel = relative_vel_x * tangent_x + relative_vel_y * tangent_y

    # 角速度 = 切向速度 / 半径
    angular_velocity = tangent_vel / pendulum_length

    return angular_velocity


def get_system_state():
    """获取系统的完整状态向量 [x, x_dot, theta, theta_dot]"""
    # 位置（相对于目标位置，转换为米）
    x = 0  # 这个会在控制器中根据目标位置计算

    # 速度（转换为米/秒）
    x_dot = slider_body.velocity.x / 1000.0

    # 角度（弧度）
    theta = get_pendulum_angle()

    # 角速度（弧度/秒）
    theta_dot = get_pendulum_angular_velocity()

    return [x, x_dot, theta, theta_dot]


def draw_plot(surface, data, pos, size, title, color=(0, 100, 200)):
    """绘制数据图表"""
    x, y, w, h = pos[0], pos[1], size[0], size[1]

    # 绘制背景和边框
    pygame.draw.rect(surface, (250, 250, 250), (x, y, w, h))
    pygame.draw.rect(surface, (100, 100, 100), (x, y, w, h), 2)

    # 绘制标题
    title_surf = small_font.render(title, True, (50, 50, 50))
    surface.blit(title_surf, (x + 5, y - 20))

    if len(data) > 1:
        # 归一化数据
        data_array = np.array(data)
        if np.ptp(data_array) > 0:
            normalized = (data_array - np.min(data_array)) / np.ptp(data_array)
        else:
            normalized = np.ones_like(data_array) * 0.5

        # 绘制数据线
        points = []
        for i, val in enumerate(normalized):
            px = x + int(i * w / len(normalized))
            py = y + h - int(val * h)
            points.append((px, py))

        if len(points) > 1:
            pygame.draw.lines(surface, color, False, points, 2)

    # 绘制零线
    pygame.draw.line(surface, (200, 200, 200), (x, y + h // 2), (x + w, y + h // 2), 1)


# 参数管理器类
class ParameterManager:
    """统一管理所有可调参数"""

    def __init__(self):
        self.current_index = 0
        self.param_groups = {}

    def get_params_for_mode(self, control_mode, is_swing_up_phase, auto_swing_up):
        """根据当前模式获取可调参数列表"""
        params = []


        if control_mode == "PID":
            params.extend([
                ("PID", "Kp", 0),
                ("PID", "Ki", 1),
                ("PID", "Kd", 2)
            ])
        elif control_mode == "LQR":
            params.extend([
                ("LQR", "Q_x", 0),
                ("LQR", "Q_x^", 1),
                ("LQR", "Q_θ", 2),
                ("LQR", "Q_θ^", 3),
                ("LQR", "R", 4)
            ])
        elif control_mode == "MPC":
            params.extend([
                ("MPC", "Q_x", 0),
                ("MPC", "Q_x^", 1),
                ("MPC", "Q_θ", 2),
                ("MPC", "Q_θ^", 3),
                ("MPC", "R", 4),
                ("MPC", "Horizon", 5)
            ])

        # 非手动模式下，总是包含swing-up参数
        if control_mode != "MANUAL" and auto_swing_up:
            params.extend([
                ("Swing-up", "K_energy", 0),
                ("Swing-up", "K_damp", 1),
                ("Swing-up", "Switch angle", 2)
            ])

        return params

    def get_total_params(self, control_mode, is_swing_up_phase, auto_swing_up):
        """获取当前模式下的参数总数"""
        return len(self.get_params_for_mode(control_mode, is_swing_up_phase, auto_swing_up))

    def cycle_parameter(self, control_mode, is_swing_up_phase, auto_swing_up):
        """切换到下一个参数"""
        total = self.get_total_params(control_mode, is_swing_up_phase, auto_swing_up)
        if total > 0:
            self.current_index = (self.current_index + 1) % total

    def get_current_param_info(self, control_mode, is_swing_up_phase, auto_swing_up):
        """获取当前选中参数的信息"""
        params = self.get_params_for_mode(control_mode, is_swing_up_phase, auto_swing_up)
        if params and self.current_index < len(params):
            return params[self.current_index]
        return None

    def reset_index(self):
        """重置参数索引"""
        self.current_index = 0


# 创建控制器
# 甩起控制器
swing_up_controller = EnergySwingUpController(M, m, l, g)

# PID控制器
angle_pid = PIDController(kp=init_kp, ki=init_ki, kd=init_kd)  # 角度控制
position_pid = PIDController(kp=2.0, ki=0.1, kd=0.5)  # 位置控制

# LQR控制器
lqr_controller = LQRController(M, m, l, g, b)

# MPC控制器
mpc_controller = MPCController(M, m, l, g, b, dt=1 / 60.0)

# 参数管理器
param_manager = ParameterManager()



while running:
    for event in pygame.event.get():
        if event.type == pygame.QUIT:
            running = False
        elif event.type == pygame.KEYDOWN:
            if event.key == pygame.K_u:
                # 重置系统到下垂位置（准备甩起）
                slider_body.position = target_position, rail_y
                slider_body.velocity = (0, 0)
                initial_angle = math.radians(180)  # 下垂位置
                pendulum_body.position = (
                    slider_body.position.x + pendulum_length * math.sin(initial_angle),
                    slider_body.position.y - pendulum_length * math.cos(initial_angle)
                )
                pendulum_body.velocity = (0, 0)
                trail_points.clear()

                # 重置所有控制器
                swing_up_controller.reset()
                angle_pid.reset()
                position_pid.reset()
                lqr_controller.reset()
                mpc_controller.reset()

                # 重新进入甩起阶段
                is_swing_up_phase = True

            elif event.key == pygame.K_SPACE:
                # 切换到垂直位置（跳过甩起）
                slider_body.position = target_position, rail_y
                slider_body.velocity = (0, 0)
                initial_angle = math.radians(5)  # 接近垂直
                pendulum_body.position = (
                    slider_body.position.x + pendulum_length * math.sin(initial_angle),
                    slider_body.position.y - pendulum_length * math.cos(initial_angle)
                )
                pendulum_body.velocity = (0, 0)
                trail_points.clear()

                # 重置控制器并直接进入稳定阶段
                swing_up_controller.reset()
                angle_pid.reset()
                position_pid.reset()
                lqr_controller.reset()
                mpc_controller.reset()
                is_swing_up_phase = False

            elif event.key == pygame.K_o:
                # 回到原点
                target_position = 0
                lqr_controller.set_target_position(target_position)
                mpc_controller.set_target_position(target_position)

            elif event.key == pygame.K_g:
                # 跳转到指定位置（示例：向右移动500像素）
                target_position += 500
                lqr_controller.set_target_position(target_position)
                mpc_controller.set_target_position(target_position)

            elif event.key == pygame.K_h:
                # 跳转到指定位置（示例：向左移动500像素）
                target_position -= 500
                lqr_controller.set_target_position(target_position)
                mpc_controller.set_target_position(target_position)

            elif event.key == pygame.K_f:
                # 切换相机跟随模式
                follow_mode = "INSTANT" if follow_mode == "SMOOTH" else "SMOOTH"
                camera.smoothing = 1.0 if follow_mode == "INSTANT" else 0.1

            elif event.key == pygame.K_s:
                # 切换自动甩起开关
                auto_swing_up = not auto_swing_up
                if not auto_swing_up:
                    is_swing_up_phase = False
                param_manager.reset_index()

            elif event.key == pygame.K_m:
                # 切换控制模式
                modes = ["MANUAL", "PID", "LQR", "MPC"]
                current_index = modes.index(control_mode)
                control_mode = modes[(current_index + 1) % len(modes)]
                angle_pid.reset()
                position_pid.reset()
                lqr_controller.reset()
                mpc_controller.reset()
                param_manager.reset_index()

            elif event.key == pygame.K_p:
                # 切换位置控制（仅PID模式）
                if control_mode == "PID":
                    use_position_control = not use_position_control
                    position_pid.reset()

            elif event.key == pygame.K_TAB:
                # 切换选中的参数
                param_manager.cycle_parameter(control_mode, is_swing_up_phase, auto_swing_up)

            elif event.key == pygame.K_r:
                # 随机扰动
                impulse_x = np.random.uniform(-500, 500)
                impulse_y = np.random.uniform(-300, 0)
                pendulum_body.apply_impulse_at_world_point((impulse_x, impulse_y), pendulum_body.position)
            elif event.key == pygame.K_k:
                if target_position > -9900:
                    target_position -= 100
                    lqr_controller.set_target_position(target_position)
            elif event.key == pygame.K_l:
                if target_position < 9900:
                    target_position += 100
                    lqr_controller.set_target_position(target_position)

    # 参数调节（使用上下箭头键）
    keys = pygame.key.get_pressed()

    # 获取当前选中的参数信息
    current_param = param_manager.get_current_param_info(control_mode, is_swing_up_phase, auto_swing_up)

    if current_param and control_mode != "MANUAL":
        group, name, param_idx = current_param

        if keys[pygame.K_UP] or keys[pygame.K_DOWN]:
            delta_multiplier = 1 if keys[pygame.K_UP] else -1

            if group == "Swing-up":
                if name == "K_energy":
                    swing_up_controller.adjust_gain(0, delta_multiplier * param_adjust_speeds["Swing-up"]["K_energy"])
                elif name == "K_damp":
                    swing_up_controller.adjust_gain(1, delta_multiplier * param_adjust_speeds["Swing-up"]["K_damp"])
                elif name == "Switch angle":
                    swing_up_controller.adjust_gain(2,
                                                    delta_multiplier * param_adjust_speeds["Swing-up"]["Switch angle"])

            elif group == "PID":
                if name == "Kp":
                    angle_pid.kp = max(0, angle_pid.kp + delta_multiplier * param_adjust_speeds["PID"]["Kp"])
                elif name == "Ki":
                    angle_pid.ki = max(0, angle_pid.ki + delta_multiplier * param_adjust_speeds["PID"]["Ki"])
                elif name == "Kd":
                    angle_pid.kd = max(0, angle_pid.kd + delta_multiplier * param_adjust_speeds["PID"]["Kd"])

            elif group == "LQR":
                if name.startswith("Q_"):
                    lqr_controller.adjust_q_weight(param_idx, delta_multiplier * param_adjust_speeds["LQR"]["Q"])
                elif name == "R":
                    lqr_controller.adjust_r_weight(delta_multiplier * param_adjust_speeds["LQR"]["R"])

            elif group == "MPC":
                if name.startswith("Q_"):
                    mpc_controller.adjust_q_weight(param_idx, delta_multiplier * param_adjust_speeds["MPC"]["Q"])
                elif name == "R":
                    mpc_controller.adjust_r_weight(delta_multiplier * param_adjust_speeds["MPC"]["R"])
                elif name == "Horizon":
                    mpc_controller.adjust_horizon(delta_multiplier * param_adjust_speeds["MPC"]["Horizon"])

    # 获取系统状态
    state = get_system_state()
    angle = state[2]
    current_world_x = slider_body.position.x

    # 控制逻辑
    if control_mode != "MANUAL":
        # 自动判断是否需要甩起
        if auto_swing_up:
            # 检查是否需要进入甩起阶段
            if abs(angle) > math.radians(90) and not is_swing_up_phase:
                is_swing_up_phase = True
                swing_up_controller.reset()

            # 检查是否可以切换到稳定控制
            if is_swing_up_phase and swing_up_controller.should_switch(angle):
                is_swing_up_phase = False
                # 重置稳定控制器
                if control_mode == "PID":
                    angle_pid.reset()
                    position_pid.reset()
                elif control_mode == "LQR":
                    lqr_controller.reset()
                elif control_mode == "MPC":
                    mpc_controller.reset()

    # 应用控制
    if control_mode == "MANUAL":
        # 手动控制
        if keys[pygame.K_LEFT]:
            slider_body.apply_force_at_world_point((-1000, 0), slider_body.position)
        if keys[pygame.K_RIGHT]:
            slider_body.apply_force_at_world_point((1000, 0), slider_body.position)

    elif is_swing_up_phase and auto_swing_up:
        # 甩起控制
        control_output = swing_up_controller.update(state)

        # 限制控制输出
        max_force = 2000
        control_output = max(-max_force, min(max_force, control_output))

        # 应用控制力
        slider_body.apply_force_at_world_point((control_output, 0), slider_body.position)

    elif control_mode == "PID":
        # PID稳定控制
        angle = get_pendulum_angle()
        angular_velocity = get_pendulum_angular_velocity()

        # 角度误差（目标是0，即垂直向上）
        angle_error = angle

        # PID控制计算
        control_output = angle_pid.update(angle_error, dt)

        # 添加角速度项以提高稳定性
        control_output += angular_velocity * 50  # 角速度阻尼

        # 位置控制（可选，帮助滑块回到目标位置）
        if use_position_control:
            position_error = (slider_body.position.x - target_position) / 100.0
            position_output = position_pid.update(position_error, dt)
            control_output -= position_output

        # 限制控制输出
        max_force = 2000
        control_output = max(-max_force, min(max_force, control_output))

        # 应用控制力
        slider_body.apply_force_at_world_point((control_output, 0), slider_body.position)

    elif control_mode == "LQR":
        # LQR控制计算
        control_output = lqr_controller.update(state, current_world_x)

        # 限制控制输出
        max_force = 2000
        control_output = max(-max_force, min(max_force, control_output))

        # 应用控制力
        slider_body.apply_force_at_world_point((control_output, 0), slider_body.position)

    elif control_mode == "MPC":
        # MPC控制计算
        control_output = mpc_controller.update(state, current_world_x)

        # 限制控制输出
        max_force = 2000
        control_output = max(-max_force, min(max_force, control_output))

        # 应用控制力
        slider_body.apply_force_at_world_point((control_output, 0), slider_body.position)

    # 更新相机位置
    camera.update(slider_body.position.x, rail_y)

    # 清屏
    screen.fill((240, 240, 240))

    # 绘制无限导轨
    draw_infinite_rail(screen, camera)

    # 如果是MPC模式且不在甩起阶段，绘制预测轨迹
    if control_mode == "MPC":
        draw_mpc_prediction(screen, camera, mpc_controller,slider_body.position.x)

    # 添加轨迹点（使用世界坐标）
    trail_points.append((pendulum_body.position.x, pendulum_body.position.y))
    if len(trail_points) > max_trail_points:
        trail_points.pop(0)

    # 绘制轨迹
    draw_trail(screen, camera, trail_points)

    # 绘制所有物理对象
    draw_objects(screen, camera)

    # 绘制目标位置标记
    if abs(target_position - current_world_x) > 10 and (control_mode == "MPC" or control_mode == "LQR"):  # 如果不在目标位置附近或不在这个模式
        target_screen_x, target_screen_y = camera.world_to_screen(target_position, rail_y)
        if 0 <= target_screen_x <= WIDTH:  # 如果在屏幕内
            pygame.draw.line(screen, (0, 200, 0),
                             (target_screen_x, target_screen_y - 30),
                             (target_screen_x, target_screen_y + 30), 3)
            target_text = small_font.render("目标位置" if chinese_isfont else "Target", True, (0, 200, 0))
            screen.blit(target_text, (target_screen_x - 20, target_screen_y - 50))

    # ========== UI面板（固定在屏幕上） ==========
    # 绘制控制信息面板
    panel_x = 750
    panel_y = 20

    # 控制模式
    mode_colors = {
        "MANUAL": (150, 0, 0),
        "PID": (0, 150, 0),
        "LQR": (0, 0, 150),
        "MPC": (150, 0, 150)
    }
    mode_names = {
        "MANUAL": "手动模式",
        "PID": "PID",
        "LQR": "LQR",
        "MPC": "MPC"
    }
    mode_color = mode_colors.get(control_mode, (50, 50, 50))
    mode_text = font.render(f"模式: {mode_names[control_mode]}" if chinese_isfont else f"Mode: {control_mode}", True, mode_color)
    screen.blit(mode_text, (panel_x, panel_y))



    # 自动甩起状态
    swing_status_color = (0, 150, 0) if auto_swing_up else (100, 100, 100)
    swing_status_text = small_font.render(
        f"自动甩起: {'开启' if auto_swing_up else '关闭'}" if chinese_isfont else f"Auto Swing-up: {'ON' if auto_swing_up else 'OFF'}",
        True, swing_status_color)
    screen.blit(swing_status_text, (panel_x, panel_y + 25))

    # 相机模式
    camera_mode_text = small_font.render(
        f"相机: {'平滑' if follow_mode == 'SMOOTH' else '瞬时'}" if chinese_isfont else f"Camera: {follow_mode}",
        True, (70, 70, 70)
    )
    screen.blit(camera_mode_text, (panel_x, panel_y + 45))

    # 显示所有可调参数
    param_y = panel_y + 70

    # 获取所有可调参数
    all_params = param_manager.get_params_for_mode(control_mode, is_swing_up_phase, auto_swing_up)

    # 在初始化部分添加公式定义
    formula_texts = {
        "MANUAL": "Manual Control Mode",
        "PID": "u(t) = Kpe(t) + Ki∫e(t)dt + Kpde(t)/dt",
        "LQR": "u = -Kx, where K is solved by Riccati equation",
        "MPC": "min Σ(x^T Qx + u^T Ru), subject to x(k+1) = Ax(k) + Bu(k)"
    }
    # 对应的中文公式
    chinese_formula_texts = {
        "MANUAL": "手动控制模式",
        "PID": "u(t) = Kpe(t) + Ki∫e(t)dt + Kpde(t)/dt",
        "LQR": "u = -Kx, 其中K通过Riccati方程求解",
        "MPC": "最小化 Σ(x^T Qx + u^T Ru), 约束条件: x(k+1) = Ax(k) + Bu(k)"
    }

    # 添加公式显示
    formula_text = chinese_formula_texts[control_mode] if chinese_isfont else formula_texts[control_mode]

    # 公式太长分成多行显示
    formula_lines = []
    max_chars_per_line = 20  # 每行最大字符数

    formula_title = small_font.render("公式:" if chinese_isfont else "Formula:", True, (70, 70, 70))
    screen.blit(formula_title, (10, int(HEIGHT/2)+5))

    if len(formula_text) > max_chars_per_line:
        words = formula_text.split(' ')
        current_line = ""
        for word in words:
            if len(current_line) + len(word) + 1 <= max_chars_per_line:
                current_line += word + " "
            else:
                formula_lines.append(current_line)
                current_line = word + " "
        if current_line:
            formula_lines.append(current_line)
    else:
        formula_lines = [formula_text]

    # 绘制公式
    for i, line in enumerate(formula_lines):
        formula_surface = small_font.render(line, True, (100, 100, 100))
        screen.blit(formula_surface, (10, int(HEIGHT/2) + 25 + i * 20))

    if control_mode != "MANUAL" and all_params:
        # 显示参数标题
        params_title = small_font.render("可调参数:" if chinese_isfont else "Adjustable Parameters:", True, (70, 70, 70))
        screen.blit(params_title, (panel_x, param_y))
        param_y += 25

        # 显示每个参数
        for i, (group, name, param_idx) in enumerate(all_params):
            # 判断是否为当前选中的参数
            is_selected = (i == param_manager.current_index)
            color = (0, 0, 200) if is_selected else (100, 100, 100)

            # 获取参数值
            value_str = ""
            if group == "PID":
                if name == "Kp":
                    value_str = f"{angle_pid.kp:.1f}"
                elif name == "Ki":
                    value_str = f"{angle_pid.ki:.1f}"
                elif name == "Kd":
                    value_str = f"{angle_pid.kd:.1f}"
            elif group == "LQR":
                if name.startswith("Q_"):
                    value_str = f"{lqr_controller.Q[param_idx, param_idx]:.2f}"
                elif name == "R":
                    value_str = f"{lqr_controller.R[0, 0]:.3f}"
            elif group == "MPC":
                if name.startswith("Q_"):
                    value_str = f"{mpc_controller.Q[param_idx, param_idx]:.1f}"
                elif name == "R":
                    value_str = f"{mpc_controller.R[0, 0]:.2f}"
                elif name == "Horizon":
                    value_str = f"{mpc_controller.N}"
            if group == "Swing-up":
                if name == "K_energy":
                    value_str = f"{swing_up_controller.k_energy:.1f}"
                elif name == "K_damp":
                    value_str = f"{swing_up_controller.k_damping:.2f}"
                elif name == "Switch angle":
                    value_str = f"{math.degrees(swing_up_controller.switch_angle):.0f}°"

            # 需要根据语言选择显示不同的组名
            group_names = {
                "Swing-up": "甩起控制" if chinese_isfont else "Swing-up",
                "PID": "PID控制" if chinese_isfont else "PID",
                "LQR": "LQR控制" if chinese_isfont else "LQR",
                "MPC": "MPC控制" if chinese_isfont else "MPC"
            }
            param_names = {
                "K_energy": "能量增益" if chinese_isfont else "K_energy",
                "K_damp": "阻尼增益" if chinese_isfont else "K_damp",
                "Switch angle": "切换角度" if chinese_isfont else "Switch angle",
                "Kp": "比例系数" if chinese_isfont else "Kp",
                "Ki": "积分系数" if chinese_isfont else "Ki",
                "Kd": "微分系数" if chinese_isfont else "Kd",
                "Q_x": "位置权重" if chinese_isfont else "Q_x",
                "Q_x^": "速度权重" if chinese_isfont else "Q_x^",
                "Q_θ": "角度权重" if chinese_isfont else "Q_θ",
                "Q_θ^": "角速度权重" if chinese_isfont else "Q_θ^",
                "R": "控制权重" if chinese_isfont else "R",
                "Horizon": "预测时域" if chinese_isfont else "Horizon"
            }
            # 显示参数名和值
            prefix = "-> " if is_selected else "  "
            param_text = small_font.render(f"{prefix}{group_names[group]}.{param_names[name]}: {value_str}", True, color)
            screen.blit(param_text, (panel_x, param_y))
            param_y += 20

    # 控制阶段指示
    if is_swing_up_phase and auto_swing_up:
        phase_text = font.render("甩起阶段" if chinese_isfont else "SWING-UP", True, (255, 100, 0))
        screen.blit(phase_text, (panel_x, param_y))
        panel_y += 30

    # 系统状态
    angle_deg = math.degrees(get_pendulum_angle())
    angular_vel = get_pendulum_angular_velocity()

    state_y = 390
    angle_text = font.render(f"角度: {angle_deg:.1f}°" if chinese_isfont else f"Angle: {angle_deg:.1f}°", True, (50, 50, 50))
    angular_vel_text = font.render(f"角速度: {angular_vel:.2f} rad/s" if chinese_isfont else f"ω: {angular_vel:.2f} rad/s", True, (50, 50, 50))
    position_text = small_font.render(f"位置: {int(current_world_x)}" if chinese_isfont else f"Position: {int(current_world_x)}", True, (50, 50, 50))


    screen.blit(angle_text, (panel_x, state_y))
    screen.blit(angular_vel_text, (panel_x, state_y + 25))
    screen.blit(position_text, (panel_x, state_y + 50))


    # 绘制控制图表
    chart_y = 520
    if is_swing_up_phase and auto_swing_up and len(swing_up_controller.energy_history) > 0:
        # 能量历史图
        draw_plot(screen, swing_up_controller.energy_history, (panel_x, chart_y),
                  (200, 60), "系统能量" if chinese_isfont else "System Energy", (255, 100, 0))
        # 控制输出历史图
        draw_plot(screen, swing_up_controller.output_history, (panel_x, chart_y + 90),
                  (200, 60), "控制输出" if chinese_isfont else "Control Output", (255, 150, 50))
    elif control_mode == "PID" and len(angle_pid.error_history) > 0:
        # 误差历史图
        draw_plot(screen, angle_pid.error_history, (panel_x, chart_y),
                  (200, 60), "角度误差" if chinese_isfont else "Angle Error", (200, 50, 50))
        # 控制输出历史图
        draw_plot(screen, angle_pid.output_history, (panel_x, chart_y + 90),
                  (200, 60), "控制输出" if chinese_isfont else "Control Output", (50, 150, 50))
    elif control_mode == "LQR" and len(lqr_controller.state_history) > 0:
        # 角度历史图
        angles = [s[2] for s in lqr_controller.state_history]
        draw_plot(screen, angles, (panel_x, chart_y), (200, 60), "角度(弧度)" if chinese_isfont else "Angle (rad)", (200, 50, 50))
        # 控制输出历史图
        draw_plot(screen, lqr_controller.output_history, (panel_x, chart_y + 90),
                  (200, 60), "控制输出" if chinese_isfont else "Control Output", (50, 150, 50))
    elif control_mode == "MPC" and len(mpc_controller.state_history) > 0:
        # 角度历史图
        angles = [s[2] for s in mpc_controller.state_history]
        draw_plot(screen, angles, (panel_x, chart_y), (200, 60), "角度(弧度)" if chinese_isfont else "Angle (rad)", (200, 50, 50))
        # 控制输出历史图
        draw_plot(screen, mpc_controller.output_history, (panel_x, chart_y + 90),
                  (200, 60), "控制输出" if chinese_isfont else "Control Output", (150, 50, 150))

    # 控制说明
    instructions = [
        "控制说明:",
        "U - 重置(下垂状态)",
        "SPACE - 重置(直立状态)",
        "S - 切换自动甩起",
        "M - 切换控制模式",
        "TAB - 选择参数",
        "↑/↓ - 调整参数",
        "R - 随机扰动",
        "F - 切换相机模式",
    ] if chinese_isfont else [
        "Controls:",
        "U - Reset (Down)",
        "SPACE - Reset (Up)",
        "S - Toggle Auto Swing",
        "M - Toggle Mode",
        "TAB - Select Parameter",
        "↑/↓ - Adjust Parameter",
        "R - Random Disturbance",
        "F - Toggle Camera Mode",
    ]

    if control_mode == "MPC" or control_mode == "LQR":
        instructions.append("K/L - 目标位置增减100" if chinese_isfont else "K/L - Target +/- 100")
        target_text = small_font.render(
            f"目标: {int(target_position)}" if chinese_isfont else f"Target: {int(target_position)}", True, (0, 150, 0))
        screen.blit(target_text, (panel_x, state_y + 70))

    if control_mode == "PID":
        instructions.append("P - 切换位置控制" if chinese_isfont else "P - Toggle Position Ctrl")
    elif control_mode == "MANUAL":
        instructions.append("←/→ - 手动控制" if chinese_isfont else "←/→ - Manual Control")
    elif control_mode == "MPC":
        instructions.append("(显示预测轨迹)" if chinese_isfont else "(Shows prediction)")

    for i, text in enumerate(instructions):
        text_surface = small_font.render(text, True, (70, 70, 70))
        screen.blit(text_surface, (10, 10 + i * 22))

    # 显示系统能量
    slider_ke = 0.5 * slider_mass * slider_body.velocity.length ** 2
    pendulum_ke = 0.5 * pendulum_mass * pendulum_body.velocity.length ** 2
    pendulum_pe = pendulum_mass * space.gravity[1] * (HEIGHT - pendulum_body.position.y)
    total_energy = slider_ke + pendulum_ke + pendulum_pe

    energy_text = f"总能量: {total_energy:.1f}" if chinese_isfont else f"Total Energy: {total_energy:.1f}"
    energy_surface = font.render(energy_text, True, (50, 50, 50))
    screen.blit(energy_surface, (10, 300))

    # 更新物理引擎
    space.step(dt)

    # 限制滑块速度（防止失控）
    max_velocity = 2000  # 增加最大速度
    if abs(slider_body.velocity.x) > max_velocity:
        slider_body.velocity = (
            max_velocity if slider_body.velocity.x > 0 else -max_velocity,
            slider_body.velocity.y
        )

    # 更新显示
    pygame.display.flip()
    clock.tick(60)

pygame.quit()