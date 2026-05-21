import numpy as np
import heapq
import math
import random
from scipy.interpolate import splprep, splev


class Planner:
    def __init__(self, grid_map, start_pos, goal_pos, robot_radius=1):
        self.grid_map = grid_map
        self.start = tuple(start_pos)
        self.goal = tuple(goal_pos)
        self.robot_radius = robot_radius
        self.height, self.width = grid_map.shape

    def is_collision(self, x, y):
        """Basit noktasal çarpışma kontrolü."""
        x, y = int(x), int(y)
        if x < 0 or x >= self.width or y < 0 or y >= self.height:
            return True
        if self.grid_map[y, x] == 1:
            return True
        return False

    def check_line_collision(self, x0, y0, x1, y1):
        """
        Bresenham Algoritması: İki nokta arasına çizilen düz bir çizginin
        haritadaki herhangi bir engele çarpıp çarpmadığını kontrol eder.
        Özellikle PRM, Theta* ve RRT* için kritik öneme sahiptir.
        """
        dx = abs(x1 - x0)
        dy = abs(y1 - y0)
        x, y = x0, y0
        n = 1 + dx + dy
        x_inc = 1 if x1 > x0 else -1
        y_inc = 1 if y1 > y0 else -1
        error = dx - dy
        dx *= 2
        dy *= 2

        for _ in range(n):
            if self.is_collision(x, y):
                return True
            if error > 0:
                x += x_inc
                error -= dy
            else:
                y += y_inc
                error += dx
        return False

    def heuristic(self, a, b):
        """A* tabanlı algoritmalar için Öklid (Kuş uçuşu) mesafesi."""
        return math.hypot(b[0] - a[0], b[1] - a[1])

    # =========================================================================
    # 1. HOLONOMİK (HER YÖNE HAREKET EDEBİLEN) PLANLAMA ALGORİTMALARI
    # =========================================================================

    def plan_with_a_star(self):
        """Klasik A* Algoritması (A-Star)"""
        neighbors = [(0, 1), (0, -1), (1, 0), (-1, 0), (1, 1), (1, -1), (-1, 1), (-1, -1)]
        close_set = set()
        came_from = {}
        gscore = {self.start: 0}
        fscore = {self.start: self.heuristic(self.start, self.goal)}
        oheap = [(fscore[self.start], self.start)]

        while oheap:
            current = heapq.heappop(oheap)[1]

            if current == self.goal:
                path = []
                while current in came_from:
                    path.append(current)
                    current = came_from[current]
                path.append(self.start)
                return path[::-1]

            close_set.add(current)
            for i, j in neighbors:
                neighbor = current[0] + i, current[1] + j
                if self.is_collision(neighbor[0], neighbor[1]):
                    continue

                tentative_g_score = gscore[current] + math.sqrt(i ** 2 + j ** 2)

                if neighbor in close_set and tentative_g_score >= gscore.get(neighbor, 0):
                    continue

                if tentative_g_score < gscore.get(neighbor, float('inf')):
                    came_from[neighbor] = current
                    gscore[neighbor] = tentative_g_score
                    fscore[neighbor] = tentative_g_score + self.heuristic(neighbor, self.goal)
                    heapq.heappush(oheap, (fscore[neighbor], neighbor))
        return []

    def plan_with_theta_star(self):
        """Theta* Algoritması: A*'ın kısıtlı açılarını (Any-Angle) aşan versiyonu."""
        neighbors = [(0, 1), (0, -1), (1, 0), (-1, 0), (1, 1), (1, -1), (-1, 1), (-1, -1)]
        close_set = set()
        came_from = {}
        gscore = {self.start: 0}
        fscore = {self.start: self.heuristic(self.start, self.goal)}
        oheap = [(fscore[self.start], self.start)]
        came_from[self.start] = self.start

        while oheap:
            current = heapq.heappop(oheap)[1]

            if current == self.goal:
                path = []
                while current != self.start:
                    path.append(current)
                    current = came_from[current]
                path.append(self.start)
                return path[::-1]

            close_set.add(current)
            for i, j in neighbors:
                neighbor = current[0] + i, current[1] + j
                if self.is_collision(neighbor[0], neighbor[1]):
                    continue

                parent = came_from[current]

                # Theta* Farkı: Ata düğüm (parent) ile komşu arasında doğrudan görüş var mı?
                if not self.check_line_collision(parent[0], parent[1], neighbor[0], neighbor[1]):
                    cost = self.heuristic(parent, neighbor)
                    if gscore[parent] + cost < gscore.get(neighbor, float('inf')):
                        gscore[neighbor] = gscore[parent] + cost
                        came_from[neighbor] = parent
                        fscore[neighbor] = gscore[neighbor] + self.heuristic(neighbor, self.goal)
                        heapq.heappush(oheap, (fscore[neighbor], neighbor))
                else:
                    cost = math.sqrt(i ** 2 + j ** 2)
                    if gscore[current] + cost < gscore.get(neighbor, float('inf')):
                        gscore[neighbor] = gscore[current] + cost
                        came_from[neighbor] = current
                        fscore[neighbor] = gscore[neighbor] + self.heuristic(neighbor, self.goal)
                        heapq.heappush(oheap, (fscore[neighbor], neighbor))
        return []

    def plan_with_d_star_lite(self):
        """D* Lite (Statik Ortam Kurulumu)"""
        # Sabit haritalar için D* Lite temelde hedeften başlayıp başlangıca gelen bir A* aramasıdır.
        temp_start = self.start
        temp_goal = self.goal
        self.start = temp_goal
        self.goal = temp_start

        path = self.plan_with_a_star()

        self.start = temp_start
        self.goal = temp_goal
        return path[::-1] if path else []

    def plan_with_prm(self, sample_size=500, k_neighbors=10):
        """PRM (Probabilistic Roadmap): Haritada rastgele ağ oluşturma yöntemi."""
        samples = [self.start, self.goal]

        # 1. Aşama: Rastgele geçerli (engele çarpmayan) noktalar üret
        for _ in range(sample_size):
            rx = random.randint(0, self.width - 1)
            ry = random.randint(0, self.height - 1)
            if not self.is_collision(rx, ry):
                samples.append((rx, ry))

        # 2. Aşama: Noktalar arası graf (bağlantı ağı) oluştur
        graph = {i: [] for i in range(len(samples))}
        for i, p1 in enumerate(samples):
            distances = []
            for j, p2 in enumerate(samples):
                if i != j:
                    distances.append((j, math.hypot(p1[0] - p2[0], p1[1] - p2[1])))
            distances.sort(key=lambda x: x[1])

            # En yakın K komşuyu bağla
            for j, dist in distances[:k_neighbors]:
                if not self.check_line_collision(int(p1[0]), int(p1[1]), int(samples[j][0]), int(samples[j][1])):
                    graph[i].append((j, dist))

        # 3. Aşama: Oluşturulan graf üzerinde Dijkstra/A* ile en kısa yolu bul
        oheap = [(0, 0)]
        close_set = set()
        came_from = {}
        gscore = {0: 0}

        while oheap:
            current_g, current_node = heapq.heappop(oheap)

            if current_node == 1:  # İndeks 1 'hedef' noktasıdır
                path = []
                while current_node in came_from:
                    path.append(samples[current_node])
                    current_node = came_from[current_node]
                path.append(self.start)
                return path[::-1]

            close_set.add(current_node)
            for neighbor, cost in graph[current_node]:
                if neighbor in close_set:
                    continue
                tentative_g = gscore[current_node] + cost
                if tentative_g < gscore.get(neighbor, float('inf')):
                    came_from[neighbor] = current_node
                    gscore[neighbor] = tentative_g
                    heapq.heappush(oheap, (tentative_g, neighbor))
        return []

    def plan_with_rrt(self, max_iter=3000, step_size=4.0):
        """Klasik RRT (Rapidly-exploring Random Tree)"""

        class Node:
            def __init__(self, x, y):
                self.x = x
                self.y = y
                self.parent = None

        nodes = [Node(self.start[0], self.start[1])]

        for _ in range(max_iter):
            # Hedef tabanlı örnekleme (%10 ihtimalle direkt hedefe bak)
            if random.random() > 0.1:
                rnd = [random.randint(0, self.width - 1), random.randint(0, self.height - 1)]
            else:
                rnd = list(self.goal)

            nearest = min(nodes, key=lambda n: math.hypot(n.x - rnd[0], n.y - rnd[1]))
            theta = math.atan2(rnd[1] - nearest.y, rnd[0] - nearest.x)

            nx = nearest.x + step_size * math.cos(theta)
            ny = nearest.y + step_size * math.sin(theta)

            if not self.check_line_collision(int(nearest.x), int(nearest.y), int(nx), int(ny)):
                new_node = Node(nx, ny)
                new_node.parent = nearest
                nodes.append(new_node)

                # Hedefe ulaştık mı?
                if math.hypot(nx - self.goal[0], ny - self.goal[1]) <= step_size:
                    path = [[self.goal[0], self.goal[1]]]
                    current = new_node
                    while current:
                        path.append([current.x, current.y])
                        current = current.parent
                    return path[::-1]
        return []

    def plan_with_rrt_star(self, max_iter=3000, step_size=4.0, search_radius=10.0):
        """RRT* Algoritması: Bulunan yolu geriye dönük optimize eden RRT versiyonu."""

        class Node:
            def __init__(self, x, y):
                self.x = x
                self.y = y
                self.parent = None
                self.cost = 0.0

        nodes = [Node(self.start[0], self.start[1])]

        for _ in range(max_iter):
            if random.random() > 0.1:
                rnd = [random.randint(0, self.width - 1), random.randint(0, self.height - 1)]
            else:
                rnd = list(self.goal)

            nearest = min(nodes, key=lambda n: math.hypot(n.x - rnd[0], n.y - rnd[1]))
            theta = math.atan2(rnd[1] - nearest.y, rnd[0] - nearest.x)

            nx = nearest.x + step_size * math.cos(theta)
            ny = nearest.y + step_size * math.sin(theta)

            if not self.check_line_collision(int(nearest.x), int(nearest.y), int(nx), int(ny)):
                new_node = Node(nx, ny)

                # RRT* Farkı: Rewiring (Yeniden Bağlama - En düşük maliyetli ata düğümü bulma)
                near_inds = [i for i, n in enumerate(nodes) if math.hypot(n.x - nx, n.y - ny) <= search_radius]
                min_cost = nearest.cost + math.hypot(nearest.x - nx, nearest.y - ny)
                min_ind = nodes.index(nearest)

                for i in near_inds:
                    n = nodes[i]
                    if not self.check_line_collision(int(n.x), int(n.y), int(nx), int(ny)):
                        cost = n.cost + math.hypot(n.x - nx, n.y - ny)
                        if cost < min_cost:
                            min_cost = cost
                            min_ind = i

                new_node.cost = min_cost
                new_node.parent = nodes[min_ind]
                nodes.append(new_node)

                if math.hypot(nx - self.goal[0], ny - self.goal[1]) <= step_size:
                    path = [[self.goal[0], self.goal[1]]]
                    current = new_node
                    while current:
                        path.append([current.x, current.y])
                        current = current.parent
                    return path[::-1]
        return []

    # =========================================================================
    # 2. NON-HOLONOMİK (ARAÇ TİPİ) PLANLAMA ALGORİTMALARI
    # =========================================================================

    def plan_with_hybrid_a_star(self):
        """Hybrid A*: Aracın dönüş kinematiğini kısıtlayarak arama yapar."""
        L = 2.0  # Araç dingil mesafesi
        steering_angles = [-math.pi / 4, 0, math.pi / 4]  # Sol, Düz, Sağ direksiyon limitleri
        step_len = 3.0

        # Durum uzayı (x, y, theta)
        start_state = (self.start[0], self.start[1], 0.0)
        close_set = set()
        came_from = {}
        gscore = {start_state: 0}
        oheap = [(self.heuristic(self.start, self.goal), start_state)]

        while oheap:
            current = heapq.heappop(oheap)[1]

            # Hedefe yeterince yaklaştıysa aramayı bitir
            if self.heuristic((current[0], current[1]), self.goal) < step_len * 2:
                path = [self.goal]
                while current in came_from:
                    path.append((current[0], current[1]))
                    current = came_from[current]
                path.append(self.start)
                return path[::-1]

            # Sonsuz hesaplamayı önlemek için açısal durumları yuvarlayarak kaydet
            state_key = (int(current[0]), int(current[1]), int(math.degrees(current[2]) // 10))
            if state_key in close_set:
                continue
            close_set.add(state_key)

            for delta in steering_angles:
                # Basit Kinematik Model
                nx = current[0] + step_len * math.cos(current[2])
                ny = current[1] + step_len * math.sin(current[2])
                ntheta = current[2] + (step_len / L) * math.tan(delta)

                if not self.check_line_collision(int(current[0]), int(current[1]), int(nx), int(ny)):
                    neighbor = (nx, ny, ntheta)
                    cost = step_len + (abs(delta) * 1.5)  # Direksiyon kırma eylemine ceza puanı ver
                    tentative_g = gscore[current] + cost

                    if tentative_g < gscore.get(neighbor, float('inf')):
                        came_from[neighbor] = current
                        gscore[neighbor] = tentative_g
                        fscore = tentative_g + self.heuristic((nx, ny), self.goal)
                        heapq.heappush(oheap, (fscore, neighbor))
        return []

    def plan_with_state_lattice(self):
        """State Lattice Planner: Önceden hesaplanmış kinematik eğrileri kullanır."""
        # Üniversite simülasyonları için genelde Hybrid A* ile aynı hareket modeliyle çağrılır.
        return self.plan_with_hybrid_a_star()

    def smooth_path(self, rough_path, smoothing_factor=50.0):
        """ÖZGÜNLÜK MODÜLÜ: RRT veya A* yollarını araba için pürüzsüzleştirir (B-Spline)."""
        if not rough_path or len(rough_path) < 4:
            return rough_path

        # 1. Aynı veya birbirine çok yakın noktaları temizle
        unique_path = []
        for p in rough_path:
            if not unique_path or (int(p[0]) != int(unique_path[-1][0]) or int(p[1]) != int(unique_path[-1][1])):
                unique_path.append(p)

        # 2. NOKTA SEYRELTME (DOWNSAMPLING) - Hayat kurtaran kısım
        # Yolu daha az sayıda "ana kontrol noktasına" indirgiyoruz ki B-Spline nefes alabilsin.
        step = max(1, len(unique_path) // 20)  # Tüm yolu ortalama 20-25 noktaya böl
        sampled_path = unique_path[::step]

        # Hedef noktasını kaybetmemek için son noktayı mutlaka ekle
        if sampled_path[-1] != unique_path[-1]:
            sampled_path.append(unique_path[-1])

        if len(sampled_path) < 4:
            return unique_path

        try:
            x = [p[0] for p in sampled_path]
            y = [p[1] for p in sampled_path]

            # 3. 's' KATSAYISINI KULLANMA
            # s=0 noktaların tam üstünden geçmektir (titreme yapar).
            # s büyüdükçe (örn: 50.0 - 100.0) yol harika bir otobana dönüşür.
            tck, u = splprep([x, y], s=smoothing_factor)

            # Pürüzsüz yolu çok sayıda noktayla (yüksek çözünürlükte) yeniden çiz
            u_new = np.linspace(0, 1, num=len(rough_path) * 2)
            x_smooth, y_smooth = splev(u_new, tck)

            return list(zip(x_smooth, y_smooth))
        except Exception as e:
            print(f"Yumuşatma hatası: {e}")
            return rough_path