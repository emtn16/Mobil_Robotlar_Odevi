import time
import math
import numpy as np
import matplotlib.pyplot as plt
import tkinter as tk
from tkinter import ttk, messagebox
import pygame
from scipy.ndimage import binary_dilation
import gc
from datetime import datetime

from environment import Environment
from planner import Planner
from visualizer import Visualizer
from robot import Robot
from sensor_fusion import EKF

# --- ROCKET LEAGUE TEMA RENKLERİ VE FONT AYARLARI ---
RL_BG = "#1A1A24"
RL_BLUE = "#00C8FF"
RL_ORANGE = "#FF6400"
RL_TEXT = "#FFFFFF"

FONT_TITLE = ("Calibri", 26, "bold")
FONT_HEADER = ("Calibri", 16, "bold")
FONT_MAIN = ("Calibri", 13)
FONT_BTN = ("Calibri", 12, "bold")


# --- YARDIMCI FONKSİYONLAR ---
def is_night_time():
    hour = datetime.now().hour
    return hour >= 19 or hour < 6


def calculate_path_length(path):
    if not path or len(path) < 2: return 0.0
    return sum(math.hypot(path[i][0] - path[i - 1][0], path[i][1] - path[i - 1][1]) for i in range(1, len(path)))


def interpolate_path(path, step=2.0):
    if not path or len(path) < 2: return path
    dense_path = [path[0]]
    for i in range(1, len(path)):
        p1, p2 = path[i - 1], path[i]
        dist = math.hypot(p2[0] - p1[0], p2[1] - p1[1])
        if dist > step:
            num_pts = int(dist / step)
            for j in range(1, num_pts + 1):
                nx = p1[0] + (p2[0] - p1[0]) * (j / num_pts)
                ny = p1[1] + (p2[1] - p1[1]) * (j / num_pts)
                dense_path.append((nx, ny))
        dense_path.append(p2)
    return dense_path


def execute_planner(planner, choice):
    start_time = time.time()
    try:
        if choice == '1':
            path, title = planner.plan_with_a_star(), "Klasik A* Algoritması"
        elif choice == '3':
            path, title = planner.plan_with_d_star_lite(), "D* Lite (Dinamik)"
        elif choice == '4':
            path, title = planner.plan_with_prm(), "PRM (Yol Haritası)"
        elif choice == '5':
            path, title = planner.plan_with_rrt(), "Klasik RRT Keşfi"
        elif choice == '6':
            path, title = planner.plan_with_rrt_star(), "RRT* Optimal"
        elif choice == '9':
            rough = planner.plan_with_a_star()
            if not rough: return None, None, None
            path, title = planner.smooth_path(rough, smoothing_factor=40.0), "B-Spline Pürüzsüz Yörünge"
        else:
            return None, None, None

        if not path or len(path) < 2: return None, title, None

        path = interpolate_path(path)
        metrics = {'time': time.time() - start_time, 'length': calculate_path_length(path)}
        return path, title, metrics
    except Exception as e:
        print(f"Planlama hatası ({choice}): {e}")
        return None, None, None


def plot_simulation_results(grid, path, h_true, h_odom, h_ekf, h_time, h_error, start_pos, goal_pos, title=""):
    plt.close('all')
    fig = plt.figure(figsize=(15, 6))
    fig.suptitle(f"Akademik Çıktı: {title} - EKF Sensör Füzyonu", fontsize=16)
    ax1 = plt.subplot(1, 2, 1)
    ax1.imshow(grid, cmap='binary', origin='upper')
    ax1.plot(start_pos[0], start_pos[1], 'go', markersize=8, label='Başlangıç')
    ax1.plot(goal_pos[0], goal_pos[1], 'ro', markersize=8, label='Hedef')
    if h_true and len(h_true) > 0:
        tx, ty = zip(*h_true);
        ox, oy = zip(*h_odom);
        ex, ey = zip(*h_ekf);
        px, py = zip(*path)
        ax1.plot(px, py, 'k--', linewidth=1, label='Planlanan Rota')
        ax1.plot(ox, oy, 'y-', linewidth=1.5, alpha=0.7, label='Ham Enkoder')
        ax1.plot(ex, ey, 'b-', linewidth=2, label='EKF Tahmini')
        ax1.plot(tx, ty, 'g-', linewidth=2, label='Gerçek Konum')
    ax1.legend(loc='best')
    ax2 = plt.subplot(1, 2, 2)
    if h_error and len(h_error) > 0:
        ax2.plot(h_time, h_error, 'r-', linewidth=1.5)
        ax2.axhline(y=np.mean(h_error), color='b', linestyle='--', label=f'MAE: {np.mean(h_error):.2f}m')
    ax2.set_title("Zaman Boyunca Konum Hatası");
    ax2.grid(True);
    ax2.legend()
    plt.tight_layout()


def plot_comparison_results_with_error(grid, p1, p2, err1, err2, start_pos, goal_pos, t1, t2):
    plt.close('all')
    fig = plt.figure(figsize=(16, 6))
    fig.suptitle(f"Kıyaslama: {t1} vs {t2} - Hata Analizi", fontsize=16)
    ax1 = plt.subplot(1, 2, 1)
    ax1.imshow(grid, cmap='binary', origin='upper')
    ax1.plot(start_pos[0], start_pos[1], 'go', markersize=8);
    ax1.plot(goal_pos[0], goal_pos[1], 'ro', markersize=8)
    if p1: ax1.plot(*zip(*p1), color=RL_BLUE, linewidth=2, label=f'Mavi: {t1}')
    if p2: ax1.plot(*zip(*p2), color=RL_ORANGE, linewidth=2, label=f'Turuncu: {t2}')
    ax1.legend(loc='best')
    ax2 = plt.subplot(1, 2, 2)
    if err1 and len(err1[4]) > 0: ax2.plot(err1[3], err1[4], color=RL_BLUE, label=f'{t1} Hatası')
    if err2 and len(err2[4]) > 0: ax2.plot(err2[3], err2[4], color=RL_ORANGE, label=f'{t2} Hatası')
    ax2.grid(True);
    ax2.legend(loc='best');
    plt.tight_layout()


# --- ARAÇ ÇİZİMİ ---
def draw_vehicle(screen, vehicle_type, rx, ry, rth, color_override=None):
    if is_night_time():
        s = pygame.Surface((100, 100), pygame.SRCALPHA)
        pygame.draw.circle(s, (0, 200, 255, 60), (50, 50), 35)
        screen.blit(s, (int(rx - 50), int(ry - 50)))
        pygame.draw.circle(screen, (255, 255, 255), (int(rx), int(ry)), 2)

    try:
        primary_color = color_override if color_override else (255, 200, 0)
        if "Drone" in vehicle_type:
            pygame.draw.circle(screen, (200, 200, 200), (int(rx), int(ry)), 6)
            for o in [math.pi / 4, 3 * math.pi / 4, 5 * math.pi / 4, 7 * math.pi / 4]:
                pygame.draw.circle(screen, (50, 255, 50),
                                   (int(rx + 12 * math.cos(rth + o)), int(ry + 12 * math.sin(rth + o))), 5)
        elif "Mecanum" in vehicle_type:
            body = [(-10, -8), (10, -8), (10, 8), (-10, 8)]
            rot = [(px * math.cos(rth) - py * math.sin(rth) + rx, px * math.sin(rth) + py * math.cos(rth) + ry) for
                   px, py in body]
            pygame.draw.polygon(screen, primary_color, rot)
            for wx, wy in [(-8, -9), (8, -9), (-8, 9), (8, 9)]:
                pygame.draw.circle(screen, (255, 100, 100), (
                int(wx * math.cos(rth) - wy * math.sin(rth) + rx), int(wx * math.sin(rth) + wy * math.cos(rth) + ry)),
                                   3)
        elif "Üçlü Omni" in vehicle_type:
            pygame.draw.circle(screen, primary_color, (int(rx), int(ry)), 10)
            for o in [0, 2 * math.pi / 3, 4 * math.pi / 3]:
                cx = rx + 10 * math.cos(rth + o)
                cy = ry + 10 * math.sin(rth + o)
                pygame.draw.rect(screen, (50, 50, 50), (int(cx - 2), int(cy - 2), 4, 4))
        elif "Diferansiyel" in vehicle_type:
            pygame.draw.circle(screen, primary_color, (int(rx), int(ry)), 9)
            for wy in [-9, 9]:
                pygame.draw.circle(screen, (40, 40, 40), (int(-wy * math.sin(rth) + rx), int(wy * math.cos(rth) + ry)),
                                   4)
            pygame.draw.circle(screen, (200, 200, 200), (int(rx + 8 * math.cos(rth)), int(ry + 8 * math.sin(rth))), 2)
        elif "4WD" in vehicle_type:
            body = [(-12, -6), (12, -6), (12, 6), (-12, 6)]
            rot = [(px * math.cos(rth) - py * math.sin(rth) + rx, px * math.sin(rth) + py * math.cos(rth) + ry) for
                   px, py in body]
            pygame.draw.polygon(screen, primary_color, rot)
            for wx, wy in [(-8, -8), (8, -8), (-8, 8), (8, 8)]:
                pygame.draw.circle(screen, (20, 20, 20), (
                int(wx * math.cos(rth) - wy * math.sin(rth) + rx), int(wx * math.sin(rth) + wy * math.cos(rth) + ry)),
                                   4)
        elif "4WS" in vehicle_type:
            car = [(-12, -5), (12, -5), (14, 0), (12, 5), (-12, 5)]
            rot = [(px * math.cos(rth) - py * math.sin(rth) + rx, px * math.sin(rth) + py * math.cos(rth) + ry) for
                   px, py in car]
            pygame.draw.polygon(screen, primary_color, rot)
        elif "Omni" in vehicle_type:
            pygame.draw.circle(screen, primary_color, (int(rx), int(ry)), 10)
            pygame.draw.line(screen, (255, 255, 255), (rx, ry), (rx + 15 * math.cos(rth), ry + 15 * math.sin(rth)), 2)
        elif "F1" in vehicle_type:
            pts = [([(-14, -3), (12, -2), (18, -1), (18, 1), (12, 2), (-14, 3)], primary_color),
                   ([(-16, -5), (-13, -5), (-13, 5), (-16, 5)], (20, 20, 20))]
            for p, c in pts:
                rot = [(x * math.cos(rth) - y * math.sin(rth) + rx, x * math.sin(rth) + y * math.cos(rth) + ry) for x, y
                       in p]
                if len(rot) >= 3: pygame.draw.polygon(screen, c, rot)
        elif "Kamyon" in vehicle_type:
            pts = [([(-22, -6), (5, -6), (5, 6), (-22, 6)], primary_color),
                   ([(7, -5), (14, -5), (14, 5), (7, 5)], (200, 200, 200))]
            for p, c in pts:
                rot = [(x * math.cos(rth) - y * math.sin(rth) + rx, x * math.sin(rth) + y * math.cos(rth) + ry) for x, y
                       in p]
                if len(rot) >= 3: pygame.draw.polygon(screen, c, rot)
        elif "Bisiklet" in vehicle_type:
            bike = [(-12, -1), (12, -1), (12, 1), (-12, 1)]
            rot = [(x * math.cos(rth) - y * math.sin(rth) + rx, x * math.sin(rth) + y * math.cos(rth) + ry) for x, y in
                   bike]
            if len(rot) >= 3: pygame.draw.polygon(screen, primary_color, rot)
            pygame.draw.circle(screen, (50, 50, 50), (int(rx + 10 * math.cos(rth)), int(ry + 10 * math.sin(rth))), 3)
            pygame.draw.circle(screen, (50, 50, 50), (int(rx - 10 * math.cos(rth)), int(ry - 10 * math.sin(rth))), 3)
        else:
            car = [(-10, -5), (10, -5), (15, 0), (10, 5), (-10, 5)]
            rot = [(x * math.cos(rth) - y * math.sin(rth) + rx, x * math.sin(rth) + y * math.cos(rth) + ry) for x, y in
                   car]
            if len(rot) >= 3:
                pygame.draw.polygon(screen, primary_color, rot);
                pygame.draw.polygon(screen, (255, 255, 255), rot, 2)
    except Exception:
        pass


# --- YARIŞ & SİMÜLASYON MOTORU (PYGAME) ---
def run_pygame_viewer(env, mode, viz, p1, t1, m1, p2=None, t2=None, m2=None, v_subtype="Octane"):
    if pygame.get_init(): pygame.quit()
    pygame.init()

    height, width = env.grid.shape
    screen = pygame.display.set_mode((width, height))
    pygame.display.set_caption(f"ROCKET LEAGUE ARENA: {v_subtype} (W/A/S/D ile Aracı Saptırın)")
    clock = pygame.time.Clock()

    rgb_array = np.zeros((width, height, 3), dtype=np.uint8)
    grid_t = env.grid.T
    rgb_array[grid_t == 0] = (20, 20, 30)
    rgb_array[grid_t == 1] = (0, 255, 255)
    bg_surface = pygame.surfarray.make_surface(rgb_array)

    speed_mult = 2.5
    if "F1" in v_subtype:
        speed_mult = 4.0
    elif "Drone" in v_subtype:
        speed_mult = 3.5
    elif "Kamyon" in v_subtype:
        speed_mult = 1.2
    elif "Bisiklet" in v_subtype:
        speed_mult = 1.8
    if mode == "C": speed_mult *= 1.5

    dt = 0.1
    err_data1, err_data2 = None, None

    if p1 and len(p1) > 1:
        st1 = math.atan2(p1[1][1] - p1[0][1], p1[1][0] - p1[0][0])
        rob1 = Robot(p1[0][0], p1[0][1], st1, dt=dt)
        ekf1 = EKF(p1[0][0], p1[0][1], st1, dt=dt)
        t_idx1, c_time1 = 1, 0.0
        ht1, ho1, he1, htime1, herr1 = [], [], [], [], []
        pose1 = (p1[0][0], p1[0][1], st1)
    else:
        p1 = None

    if mode == "B" and p2 and len(p2) > 1:
        st2 = math.atan2(p2[1][1] - p2[0][1], p2[1][0] - p2[0][0])
        rob2 = Robot(p2[0][0], p2[0][1], st2, dt=dt)
        ekf2 = EKF(p2[0][0], p2[0][1], st2, dt=dt)
        t_idx2, c_time2 = 1, 0.0
        ht2, ho2, he2, htime2, herr2 = [], [], [], [], []
        pose2 = (p2[0][0], p2[0][1], st2)
    else:
        p2 = None

    font_alert = pygame.font.SysFont("Calibri", 22, bold=True)

    running = True
    while running:
        done1 = not p1 or t_idx1 >= len(p1)
        done2 = not p2 or t_idx2 >= len(p2)
        if done1 and done2: break

        for event in pygame.event.get():
            if event.type == pygame.QUIT: running = False

        keys = pygame.key.get_pressed()
        dist_x, dist_y = 0.0, 0.0
        disturbance_active = False
        dist_force = 1.5

        if keys[pygame.K_LEFT] or keys[pygame.K_a]: dist_x -= dist_force; disturbance_active = True
        if keys[pygame.K_RIGHT] or keys[pygame.K_d]: dist_x += dist_force; disturbance_active = True
        if keys[pygame.K_UP] or keys[pygame.K_w]: dist_y -= dist_force; disturbance_active = True
        if keys[pygame.K_DOWN] or keys[pygame.K_s]: dist_y += dist_force; disturbance_active = True

        screen.blit(bg_surface, (0, 0))
        pygame.draw.circle(screen, (0, 255, 100), env.start_pos, 8)
        pygame.draw.circle(screen, (255, 0, 100), env.goal_pos, 8)

        if p1:
            if not done1:
                tx, ty = p1[t_idx1]
                v, s = rob1.pure_pursuit_control(tx, ty)
                rx, ry, rth = rob1.move(v * speed_mult, s)
                if disturbance_active: rx += dist_x; ry += dist_y; rob1.x, rob1.y = rx, ry
                if math.hypot(tx - rx, ty - ry) < 3.0: t_idx1 += 1

                ekf1.predict(v * speed_mult, s)
                z = np.array(
                    [rx + np.random.normal(0, 0.2), ry + np.random.normal(0, 0.2), rth + np.random.normal(0, 0.05)])
                ex, ey, eth = ekf1.update(z)

                ht1.append((rx, ry));
                ho1.append((rob1.odom_x, rob1.odom_y));
                he1.append((ex, ey))
                herr1.append(math.hypot(rx - ex, ry - ey));
                htime1.append(c_time1)
                c_time1 += dt
                pose1 = (rx, ry, rth)
            if len(ht1) > 2: pygame.draw.lines(screen, (0, 200, 255), False, ht1, 3)
            draw_vehicle(screen, v_subtype, pose1[0], pose1[1], pose1[2], color_override=(0, 200, 255))

        if p2:
            if not done2:
                tx2, ty2 = p2[t_idx2]
                v2, s2 = rob2.pure_pursuit_control(tx2, ty2)
                rx2, ry2, rth2 = rob2.move(v2 * speed_mult, s2)
                if disturbance_active: rx2 += dist_x; ry2 += dist_y; rob2.x, rob2.y = rx2, ry2
                if math.hypot(tx2 - rx2, ty2 - ry2) < 3.0: t_idx2 += 1

                ekf2.predict(v2 * speed_mult, s2)
                z2 = np.array(
                    [rx2 + np.random.normal(0, 0.2), ry2 + np.random.normal(0, 0.2), rth2 + np.random.normal(0, 0.05)])
                ex2, ey2, eth2 = ekf2.update(z2)

                ht2.append((rx2, ry2));
                ho2.append((rob2.odom_x, rob2.odom_y));
                he2.append((ex2, ey2))
                herr2.append(math.hypot(rx2 - ex2, ry2 - ey2));
                htime2.append(c_time2)
                c_time2 += dt
                pose2 = (rx2, ry2, rth2)
            if len(ht2) > 2: pygame.draw.lines(screen, (255, 100, 0), False, ht2, 3)
            draw_vehicle(screen, v_subtype, pose2[0], pose2[1], pose2[2], color_override=(255, 100, 0))

        if disturbance_active:
            screen.blit(font_alert.render("⚠️ HARİCİ BOZUCU KUVVET (KAYMA) AKTİF!", True, (255, 50, 50)), (20, 20))

        pygame.display.flip()
        clock.tick(60)

    if p1: err_data1 = (ht1, ho1, he1, htime1, herr1)
    if p2: err_data2 = (ht2, ho2, he2, htime2, herr2)

    font_large = pygame.font.SysFont("Calibri", 32, bold=True)
    font_btn = pygame.font.SysFont("Calibri", 20, bold=True)
    btn_out = pygame.Rect(width // 2 - 220, height // 2, 220, 50)
    btn_close = pygame.Rect(width // 2 + 20, height // 2, 150, 50)

    overlay = pygame.Surface((width, height));
    overlay.set_alpha(180);
    overlay.fill((0, 0, 0))
    screen.blit(overlay, (0, 0))
    text_win = font_large.render("YARIŞ VE SİMÜLASYON TAMAMLANDI", True, (255, 255, 255))
    screen.blit(text_win, (width // 2 - text_win.get_width() // 2, height // 2 - 80))

    waiting, show_outputs = True, False
    while waiting:
        m_pos = pygame.mouse.get_pos()
        pygame.draw.rect(screen, (0, 200, 255) if btn_out.collidepoint(m_pos) else (0, 140, 200), btn_out,
                         border_radius=8)
        pygame.draw.rect(screen, (255, 100, 0) if btn_close.collidepoint(m_pos) else (200, 50, 0), btn_close,
                         border_radius=8)
        screen.blit(font_btn.render("Ödev Çıktılarını Gör", True, (255, 255, 255)), (btn_out.x + 20, btn_out.y + 12))
        screen.blit(font_btn.render("Kapat", True, (255, 255, 255)), (btn_close.x + 45, btn_close.y + 12))
        pygame.display.flip()
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                waiting = False
            elif event.type == pygame.MOUSEBUTTONDOWN:
                if btn_out.collidepoint(event.pos):
                    show_outputs = True; waiting = False
                elif btn_close.collidepoint(event.pos):
                    waiting = False
        clock.tick(15)
    pygame.quit()

    if show_outputs:
        plt.close('all')
        if mode == "A" or mode == "C":
            plot_simulation_results(env.grid, p1, err_data1[0], err_data1[1], err_data1[2], err_data1[3], err_data1[4],
                                    env.start_pos, env.goal_pos, t1)
            viz.show_single(t1, p1, m1)
        elif mode == "B":
            plot_comparison_results_with_error(env.grid, p1, p2, err_data1, err_data2, env.start_pos, env.goal_pos, t1,
                                               t2)
            viz.show_comparison(t1, p1, m1, t2, p2, m2)
        plt.show()


# --- ROCKET LEAGUE GUI (AŞAMALI WIZARD) ---
class OtonomGUI:
    def __init__(self, root):
        self.root = root
        self.root.title("RL AUTO-ARENA")
        self.root.geometry("800x650")
        self.root.configure(bg=RL_BG)
        self.root.protocol("WM_DELETE_WINDOW", self.on_closing)

        try:
            self.env = Environment('labirent_resim.png')
            self.env.load_map()

            # --- ARACIN KAPLADIĞI ALAN KADAR GÜVENLİK PAYI (16 PİKSEL) ---
            margin = 16
            safe_grid = binary_dilation(self.env.grid, iterations=margin).astype(np.int8)
            h, w = safe_grid.shape
            safe_grid[0:margin, :] = 1;
            safe_grid[-margin:, :] = 1;
            safe_grid[:, 0:margin] = 1;
            safe_grid[:, -margin:] = 1

            sx, sy = int(self.env.start_pos[0]), int(self.env.start_pos[1])
            gx, gy = int(self.env.goal_pos[0]), int(self.env.goal_pos[1])
            for dy in range(-margin, margin + 1):
                for dx in range(-margin, margin + 1):
                    if 0 <= sy + dy < h and 0 <= sx + dx < w: safe_grid[sy + dy, sx + dx] = 0
                    if 0 <= gy + dy < h and 0 <= gx + dx < w: safe_grid[gy + dy, gx + dx] = 0

            self.planner = Planner(safe_grid, self.env.start_pos, self.env.goal_pos)
            self.viz = Visualizer(self.env.grid, self.env.start_pos, self.env.goal_pos)
        except Exception as e:
            messagebox.showerror("Hata", f"Harita yüklenemedi. Detay: {e}")
            root.destroy();
            return

        self.all_algs = {
            "1. Klasik A*": '1', "2. D* Lite (Dinamik)": '3',
            "3. PRM (Yol Haritası)": '4', "4. Klasik RRT": '5', "5. RRT* Optimal": '6',
            "6. B-Spline Smooth": '9'
        }

        self.vehicle_type = tk.StringVar(value="holo")
        self.selected_mode = tk.StringVar(value="")
        self.vehicle_subtype = tk.StringVar(value="🤖 Omni-Robot")

        self.container = tk.Frame(self.root, bg=RL_BG)
        self.container.pack(fill="both", expand=True)

        self.frames = {}
        for F in (StartScreen, VehicleScreen, ModeScreen, ActionScreen):
            frame = F(parent=self.container, controller=self)
            self.frames[F.__name__] = frame
            frame.grid(row=0, column=0, sticky="nsew")

        self.show_frame("StartScreen")

    def show_frame(self, page_name):
        frame = self.frames[page_name]
        frame.tkraise()
        if hasattr(frame, "update_options"): frame.update_options()

    def on_closing(self):
        if pygame.get_init(): pygame.quit()
        plt.close('all')
        self.root.destroy()
        import sys;
        sys.exit()


class StartScreen(tk.Frame):
    def __init__(self, parent, controller):
        tk.Frame.__init__(self, parent, bg=RL_BG)
        tk.Label(self, text="ROCKET LEAGUE\nOTONOM ARENA", font=FONT_TITLE, bg=RL_BG, fg=RL_BLUE).pack(pady=80)

        tk.Button(self, text="SİSTEMİ BAŞLAT", font=FONT_BTN, bg=RL_ORANGE, fg=RL_TEXT, bd=0, cursor="hand2",
                  command=lambda: controller.show_frame("VehicleScreen"), width=20, height=2).pack(pady=15)
        # --- YENİ EKLENEN ÇIKIŞ BUTONU ---
        tk.Button(self, text="SİSTEMDEN ÇIK", font=FONT_BTN, bg="#E53935", fg=RL_TEXT, bd=0, cursor="hand2",
                  command=controller.on_closing, width=20, height=2).pack(pady=15)


class VehicleScreen(tk.Frame):
    def __init__(self, parent, controller):
        tk.Frame.__init__(self, parent, bg=RL_BG)
        self.controller = controller
        tk.Label(self, text="ADIM 1: GARAJ - Araç Seçimi", font=FONT_HEADER, bg=RL_BG, fg=RL_TEXT).pack(pady=15)

        tk.Label(self, text="▶ Holonomik Araçlar (Çok Yönlü & Uçan)", font=FONT_MAIN, bg=RL_BG, fg=RL_BLUE).pack(
            anchor="w", padx=40, pady=(5, 5))
        f_holo = tk.Frame(self, bg=RL_BG)
        f_holo.pack(fill="x", padx=40)

        holo_vehicles = ["🤖 Omni-Robot", "⚙️ Mecanum", "🔺 Üçlü Omni", "🚁 Drone"]
        self.buttons = {}
        r, c = 0, 0
        for v in holo_vehicles:
            btn = tk.Button(f_holo, text=v, font=FONT_BTN, bg="#333", fg=RL_TEXT, bd=0, pady=8, width=16,
                            cursor="hand2", command=lambda x=v: self.select_vehicle("holo", x))
            btn.grid(row=r, column=c, padx=5, pady=5)
            self.buttons[v] = btn
            c += 1
            if c > 2: c = 0; r += 1

        tk.Label(self, text="▶ Non-Holonomik Araçlar (Araç Fiziği)", font=FONT_MAIN, bg=RL_BG, fg=RL_ORANGE).pack(
            anchor="w", padx=40, pady=(15, 5))
        f_non = tk.Frame(self, bg=RL_BG)
        f_non.pack(fill="x", padx=40)

        non_holo_vehicles = ["🚙 Octane", "🛞 Diferansiyel", "🚜 4WD Skid", "🚘 4WS", "🏎️ F1 Aracı", "🚛 Kamyon",
                             "🚲 Bisiklet"]
        r, c = 0, 0
        for v in non_holo_vehicles:
            btn = tk.Button(f_non, text=v, font=FONT_BTN, bg="#333", fg=RL_TEXT, bd=0, pady=8, width=16, cursor="hand2",
                            command=lambda x=v: self.select_vehicle("non_holo", x))
            btn.grid(row=r, column=c, padx=5, pady=5)
            self.buttons[v] = btn
            c += 1
            if c > 2: c = 0; r += 1

        nav_frame = tk.Frame(self, bg=RL_BG)
        nav_frame.pack(side="bottom", fill="x", pady=20)
        tk.Button(nav_frame, text="← Geri", font=FONT_BTN, bg="#444", fg=RL_TEXT, bd=0,
                  command=lambda: controller.show_frame("StartScreen"), width=12, pady=5).pack(side="left", padx=40)
        tk.Button(nav_frame, text="İleri →", font=FONT_BTN, bg=RL_BLUE, fg=RL_TEXT, bd=0,
                  command=lambda: controller.show_frame("ModeScreen"), width=12, pady=5).pack(side="right", padx=40)

        self.select_vehicle("holo", holo_vehicles[0])

    def select_vehicle(self, v_type, v_name):
        self.controller.vehicle_type.set(v_type)
        self.controller.vehicle_subtype.set(v_name)
        for name, btn in self.buttons.items():
            btn.config(
                bg=RL_BLUE if (name == v_name and v_type == "holo") else RL_ORANGE if (name == v_name) else "#333",
                fg="#000" if name == v_name else RL_TEXT)


class ModeScreen(tk.Frame):
    def __init__(self, parent, controller):
        tk.Frame.__init__(self, parent, bg=RL_BG)
        tk.Label(self, text="ADIM 2: Maç Formatını Seçin", font=FONT_HEADER, bg=RL_BG, fg=RL_TEXT).pack(pady=40)

        def select_mode(mode):
            controller.selected_mode.set(mode)
            controller.show_frame("ActionScreen")

        btn_args = {"font": FONT_BTN, "bd": 0, "fg": RL_TEXT, "width": 35, "pady": 12, "cursor": "hand2"}
        tk.Button(self, text="A. Tekli Antrenman (1 Araç Koşar)", bg="#333", command=lambda: select_mode("A"),
                  **btn_args).pack(pady=8)
        tk.Button(self, text="B. Kıyaslama Modu (2 Araç Yarışır)", bg="#444", command=lambda: select_mode("B"),
                  **btn_args).pack(pady=8)
        tk.Button(self, text="C. Performans Turu (Hızlı Sürüş)", bg=RL_ORANGE, command=lambda: select_mode("C"),
                  **btn_args).pack(pady=20)

        btn_frame = tk.Frame(self, bg=RL_BG)
        btn_frame.pack(side="bottom", fill="x", pady=20)
        tk.Button(btn_frame, text="← Geri", font=FONT_BTN, bg="#444", fg=RL_TEXT, bd=0,
                  command=lambda: controller.show_frame("VehicleScreen"), width=12, pady=5).pack(side="left", padx=40)


class ActionScreen(tk.Frame):
    def __init__(self, parent, controller):
        tk.Frame.__init__(self, parent, bg=RL_BG)
        self.controller = controller

        self.lbl_title = tk.Label(self, text="ADIM 3: Taktik (Algoritma) Seçimi", font=FONT_HEADER, bg=RL_BG,
                                  fg=RL_TEXT)
        self.lbl_title.pack(pady=30)

        self.combo1 = ttk.Combobox(self, state="readonly", width=40, font=FONT_MAIN)
        self.combo1.pack(pady=10)

        self.lbl_vs = tk.Label(self, text="Rakip Taktik (Sadece Kıyaslama İçin):", font=FONT_MAIN, bg=RL_BG, fg=RL_TEXT)
        self.combo2 = ttk.Combobox(self, state="readonly", width=40, font=FONT_MAIN)

        self.btn_run = tk.Button(self, text="🚀 SAHAYA ÇIK", font=FONT_BTN, bg=RL_BLUE, fg=RL_TEXT, bd=0, width=25,
                                 pady=12, command=self.execute)
        self.btn_run.pack(pady=40)

        # --- BURAYA EKLENDİ ---
        btn_frame = tk.Frame(self, bg=RL_BG)
        btn_frame.pack(side="bottom", fill="x", pady=20)

        # Geri Butonu
        tk.Button(btn_frame, text="← Geri", font=FONT_BTN, bg="#444", fg=RL_TEXT, bd=0,
                  command=lambda: controller.show_frame("ModeScreen"), width=12, pady=5).pack(side="left", padx=40)

        # YENİ ÇIKIŞ BUTONU
        tk.Button(btn_frame, text="SİSTEMDEN ÇIK", font=FONT_BTN, bg="#E53935", fg=RL_TEXT, bd=0,
                  command=controller.on_closing, width=15, pady=5).pack(side="right", padx=40)

    def update_options(self):
        algs = self.controller.all_algs
        alg_list = list(algs.keys())

        self.combo1['values'] = alg_list
        if alg_list: self.combo1.current(len(alg_list) - 1)

        mode = self.controller.selected_mode.get()
        if mode == "B":
            self.lbl_vs.pack(pady=(15, 0))
            self.combo2.pack(pady=5)
            self.combo2['values'] = alg_list
            if len(alg_list) > 1:
                self.combo2.current(0)
            else:
                self.combo2.current(len(alg_list) - 1)
        else:
            self.lbl_vs.pack_forget()
            self.combo2.pack_forget()

    def execute(self):
        try:
            v_subtype = self.controller.vehicle_subtype.get()
            mode = self.controller.selected_mode.get()
            algs = self.controller.all_algs

            if not self.combo1.get(): return

            self.btn_run.config(text="⏳ HESAPLANIYOR... BEKLEYİN", bg="#555", state="disabled")
            self.controller.root.update()

            plt.close('all')
            gc.collect()

            if "Drone" in v_subtype:
                empty_grid = np.zeros_like(self.controller.env.grid)
                active_planner = Planner(empty_grid, self.controller.env.start_pos, self.controller.env.goal_pos)
            else:
                active_planner = self.controller.planner

            if mode == "A":
                p1, t1, m1 = execute_planner(active_planner, algs[self.combo1.get()])
                if p1:
                    run_pygame_viewer(self.controller.env, mode, self.controller.viz, p1, t1, m1, v_subtype=v_subtype)
                else:
                    messagebox.showwarning("Uyarı", "Yol bulunamadı! (Hedef kapalı olabilir)")

            elif mode == "B":
                if not self.combo2.get(): return
                p1, t1, m1 = execute_planner(active_planner, algs[self.combo1.get()])
                p2, t2, m2 = execute_planner(active_planner, algs[self.combo2.get()])
                if p1 and p2:
                    run_pygame_viewer(self.controller.env, mode, self.controller.viz, p1, t1, m1, p2, t2, m2,
                                      v_subtype=v_subtype)
                else:
                    messagebox.showwarning("Uyarı", "Algoritmaların biri rotayı hesaplayamadı!")

            elif mode == "C":
                p1, t1, m1 = execute_planner(active_planner, algs[self.combo1.get()])
                if p1:
                    run_pygame_viewer(self.controller.env, mode, self.controller.viz, p1, t1, m1, v_subtype=v_subtype)
                else:
                    messagebox.showwarning("Uyarı", "Yol bulunamadı, simülasyon başlatılamıyor!")

        except Exception as e:
            messagebox.showerror("Hata", f"Beklenmeyen bir hata oluştu:\n{str(e)}")
            if pygame.get_init(): pygame.quit()
        finally:
            self.btn_run.config(text="🚀 SAHAYA ÇIK", bg=RL_BLUE, state="normal")
            self.controller.root.update()


if __name__ == "__main__":
    root = tk.Tk()
    style = ttk.Style()
    style.theme_use('clam')
    app = OtonomGUI(root)
    root.mainloop()