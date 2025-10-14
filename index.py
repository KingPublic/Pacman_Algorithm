# Implementasi A* (pathfinding), MDP (value iteration), dan RL (Q-learning) untuk domain bergaya Pac-Man.
# Catatan:
# - Implementasi disederhanakan supaya mudah dibaca & diintegrasikan ke laporan.
# - A*: grid dengan dinding (#), jalan (.), start, goal.
# - MDP: kebijakan mode ghost (CHASE vs SCATTER) berdasar jarak & status power-pellet.
# - RL: Q-learning untuk ghost mengejar Pac-Man pada grid kecil; reward +10 jika menangkap, -1 per langkah.

from collections import deque
import heapq
import math
import random
from typing import List, Tuple, Dict, Optional

# =========================================================
# =============== 1) A* PATHFINDING =======================
# =========================================================

Grid = List[str]
Pos = Tuple[int, int]

def manhattan(a: Pos, b: Pos) -> int:
    return abs(a[0]-b[0]) + abs(a[1]-b[1])

def neighbors(grid: Grid, p: Pos) -> List[Pos]:
    H, W = len(grid), len(grid[0])
    res = []
    for dx, dy in [(1,0),(-1,0),(0,1),(0,-1)]:
        nx, ny = p[0]+dx, p[1]+dy
        if 0 <= nx < H and 0 <= ny < W and grid[nx][ny] != '#':
            res.append((nx, ny))
    return res

def astar(grid: Grid, start: Pos, goal: Pos) -> Optional[List[Pos]]:
    """Return path from start to goal or None if no path. Uses 4-neighborhood."""
    open_heap = []
    g = {start: 0}
    parent: Dict[Pos, Optional[Pos]] = {start: None}
    heapq.heappush(open_heap, (manhattan(start, goal), start))
    closed = set()
    while open_heap:
        _, u = heapq.heappop(open_heap)
        if u in closed:
            continue
        if u == goal:
            # reconstruct
            path = []
            cur = u
            while cur is not None:
                path.append(cur)
                cur = parent[cur]
            return path[::-1]
        closed.add(u)
        for v in neighbors(grid, u):
            if v in closed:
                continue
            tentative = g[u] + 1
            if tentative < g.get(v, math.inf):
                g[v] = tentative
                parent[v] = u
                f = tentative + manhattan(v, goal)
                heapq.heappush(open_heap, (f, v))
    return None

# =========================================================
# =============== 2) MDP VALUE-ITERATION ==================
# =========================================================
# Formulasi ringkas:
# State = (distance_bin, pellet_active) dengan:
# distance_bin ∈ {NEAR, MID, FAR}; pellet_active ∈ {0,1}
# Action ∈ {CHASE, SCATTER}
# Transition: sebagian stokastik untuk merefleksikan ketidakpastian game.
# Reward: +5 jika aksi "benar" secara taktis, -3 jika "salah", -1 netral.
# "Benar" (heuristik): jika pellet_active=1, SCATTER lebih aman.
#                      jika pellet_active=0: NEAR→CHASE, MID→CHASE, FAR→SCATTER (untuk regroup).

NEAR, MID, FAR = 0, 1, 2
CHASE, SCATTER = 0, 1
STATE_SPACE = [(d, p) for d in (NEAR, MID, FAR) for p in (0,1)]
ACTIONS = [CHASE, SCATTER]

def mdp_transitions(s: Tuple[int,int], a: int) -> List[Tuple[Tuple[int,int], float]]:
    d, pellet = s
    # Probabilistik sederhana atas perubahan jarak & status pellet:
    # - pellet bisa habis dengan p=0.2 jika aktif; bisa aktif dengan p=0.05 jika tidak.
    # - jarak cenderung mengecil saat CHASE, membesar saat SCATTER.
    # - tambahkan sedikit noise agar tidak deterministik.
    pellet_next = pellet
    r = random.random()
    if pellet == 1 and r < 0.2:
        pellet_next = 0
    elif pellet == 0 and r < 0.05:
        pellet_next = 1

    def shift_distance(d, delta):
        nd = d + delta
        return max(NEAR, min(FAR, nd))

    outcomes = []
    if a == CHASE:
        # dominan mendekat
        probs = [(shift_distance(d, -1), 0.6),
                 (d, 0.3),
                 (shift_distance(d, +1), 0.1)]
    else:
        # dominan menjauh
        probs = [(shift_distance(d, +1), 0.6),
                 (d, 0.3),
                 (shift_distance(d, -1), 0.1)]
    # gabungkan dengan perubahan pellet
    for nd, p in probs:
        outcomes.append(((nd, pellet_next), p))
    # normalisasi (harusnya sudah 1.0)
    return outcomes

def mdp_reward(s: Tuple[int,int], a: int) -> float:
    d, pellet = s
    # Heuristik "benar"
    prefer_scatter = (pellet == 1) or (d == FAR)
    prefer_chase = (pellet == 0) and (d in (NEAR, MID))
    if a == SCATTER and prefer_scatter:
        return 5.0
    if a == CHASE and prefer_chase:
        return 5.0
    # kalau tidak ideal, penalti
    return -3.0

def value_iteration(gamma=0.9, theta=1e-4, max_iter=1000):
    V = {s: 0.0 for s in STATE_SPACE}
    for _ in range(max_iter):
        delta = 0.0
        for s in STATE_SPACE:
            best = -1e9
            for a in ACTIONS:
                val = mdp_reward(s, a)
                for s2, p in mdp_transitions(s, a):
                    val += gamma * p * V[s2]
                best = max(best, val)
            delta = max(delta, abs(best - V[s]))
            V[s] = best
        if delta < theta:
            break
    # derive policy
    pi: Dict[Tuple[int,int], int] = {}
    for s in STATE_SPACE:
        best_a, best_val = None, -1e9
        for a in ACTIONS:
            val = mdp_reward(s, a)
            for s2, p in mdp_transitions(s, a):
                val += gamma * p * V[s2]
            if val > best_val:
                best_val = val
                best_a = a
        pi[s] = best_a
    return V, pi

# =========================================================
# =============== 3) RL: Q-LEARNING =======================
# =========================================================
# Lingkungan grid kecil (7x7). Ghost (agent) ingin menangkap Pac-Man.
# Pac-Man bergerak deterministik sederhana: menuju "pellets corner" (0,0) atau diam (mode uji).
# State: (gx, gy, px, py) dibinning kasar supaya Q-table tidak raksasa.
# Aksi agent: {UP, DOWN, LEFT, RIGHT, STAY}
# Reward: +10 saat menangkap (gx==px & gy==py), -1 per langkah.

A_UP, A_DOWN, A_LEFT, A_RIGHT, A_STAY = 0, 1, 2, 3, 4
A_LIST = [A_UP, A_DOWN, A_LEFT, A_RIGHT, A_STAY]

class MiniChaseEnv:
    def __init__(self, size=7, pacman_moves=True):
        self.N = size
        self.pacman_moves = pacman_moves
        self.reset()

    def reset(self, ghost_pos=None, pac_pos=None):
        if ghost_pos is None:
            self.gx, self.gy = random.randrange(self.N), random.randrange(self.N)
        else:
            self.gx, self.gy = ghost_pos
        if pac_pos is None:
            self.px, self.py = random.randrange(self.N), random.randrange(self.N)
        else:
            self.px, self.py = pac_pos
        return self.state()

    def state(self):
        # binning ringan: pakai posisi langsung (ukuran kecil)
        return (self.gx, self.gy, self.px, self.py)

    def step(self, a: int):
        # agent (ghost) bergerak
        if a == A_UP:    self.gx = max(0, self.gx-1)
        if a == A_DOWN:  self.gx = min(self.N-1, self.gx+1)
        if a == A_LEFT:  self.gy = max(0, self.gy-1)
        if a == A_RIGHT: self.gy = min(self.N-1, self.gy+1)
        # pacman bergerak: menuju (0,0) dengan greedy manhattan (opsional)
        if self.pacman_moves:
            dx = -1 if self.px > 0 else 0
            dy = -1 if self.py > 0 else 0
            self.px = max(0, self.px + dx)
            self.py = max(0, self.py + dy)
        # reward & done
        done = (self.gx == self.px and self.gy == self.py)
        reward = 10.0 if done else -1.0
        return self.state(), reward, done

def q_learning(env: MiniChaseEnv, episodes=800, alpha=0.5, gamma=0.95, eps=0.2, max_steps=60):
    Q: Dict[Tuple[int,int,int,int], List[float]] = {}
    def getQ(s):
        if s not in Q:
            Q[s] = [0.0]*len(A_LIST)
        return Q[s]

    for _ in range(episodes):
        s = env.reset()
        for _ in range(max_steps):
            qvals = getQ(s)
            if random.random() < eps:
                a = random.choice(A_LIST)
            else:
                a = max(range(len(A_LIST)), key=lambda i: qvals[i])
            s2, r, done = env.step(a)
            q_next = getQ(s2)
            qvals[a] = qvals[a] + alpha * (r + gamma * max(q_next) - qvals[a])
            s = s2
            if done:
                break
    return Q

def greedy_policy(Q, s):
    qvals = Q.get(s, [0.0]*len(A_LIST))
    return max(range(len(A_LIST)), key=lambda i: qvals[i])

# =========================================================
# =============== 4) UTIL & PENGUJIAN =====================
# =========================================================

def print_grid(grid: Grid, path: Optional[List[Pos]] = None):
    mark = set(path or [])
    for i, row in enumerate(grid):
        s = ""
        for j, ch in enumerate(row):
            if (i,j) in mark and ch == '.':
                s += '*'
            else:
                s += ch
        print(s)

def test_astar_cases():
    print("\n=== TEST A* (5 kasus) ===")
    cases = []
    # Kasus 1: koridor lurus
    grid1 = [
        "S....G",
    ]
    cases.append((grid1, (0,0), (0,5), "koridor lurus"))

    # Kasus 2: halangan sederhana
    grid2 = [
        "S.#..G",
        ".....#",
        "#####."
    ]
    cases.append((grid2, (0,0), (0,5), "halangan sederhana"))

    # Kasus 3: kotak 5x7 labirin ringan
    grid3 = [
        "S..#..G",
        ".##.#..",
        ".#..#..",
        ".#..#..",
        "......."
    ]
    cases.append((grid3, (0,0), (0,6), "labirin ringan"))

    # Kasus 4: tidak ada jalur
    grid4 = [
        "S#.#G",
        "#####",
        "....."
    ]
    cases.append((grid4, (0,0), (0,4), "buntu (no path)"))

    # Kasus 5: start==goal
    grid5 = [
        "S"
    ]
    cases.append((grid5, (0,0), (0,0), "start==goal"))

    for idx, (grid, s, g, name) in enumerate(cases, 1):
        # ganti S/G jadi jalan saat komputasi
        grid_use = []
        for i, row in enumerate(grid):
            row2 = list(row)
            for j, ch in enumerate(row2):
                if (i,j) == s or (i,j) == g:
                    row2[j] = '.'
            grid_use.append("".join(row2))
        path = astar(grid_use, s, g)
        print(f"\nKasus {idx}: {name}")
        if path is None:
            print("  Path: TIDAK DITEMUKAN")
        else:
            print(f"  Panjang path: {len(path)-1}")
            # render
            gvis = [list(r) for r in grid]
            for (x,y) in path:
                if (x,y) != s and (x,y) != g and gvis[x][y] == '.':
                    gvis[x][y] = '*'
            gvis[s[0]][s[1]] = 'S'
            gvis[g[0]][g[1]] = 'G'
            print_grid(["".join(r) for r in gvis], None)

def test_mdp_cases():
    print("\n=== TEST MDP (Value Iteration) ===")
    random.seed(0)  # untuk reprodusibilitas transition sampling
    V, pi = value_iteration(gamma=0.9, theta=1e-4, max_iter=500)
    def a2str(a): return "CHASE" if a == CHASE else "SCATTER"
    # 5 skenario
    scenarios = [
        ((NEAR, 0), "Dekat & pellet OFF"),
        ((MID, 0),  "Sedang & pellet OFF"),
        ((FAR, 0),  "Jauh & pellet OFF"),
        ((NEAR, 1), "Dekat & pellet ON"),
        ((FAR, 1),  "Jauh & pellet ON"),
    ]
    for s, desc in scenarios:
        print(f"State: {desc:22s} => Policy: {a2str(pi[s])} (V={V[s]:.2f})")

def run_episode(env: MiniChaseEnv, Q, max_steps=60):
    s = env.reset()
    steps = 0
    for _ in range(max_steps):
        a = greedy_policy(Q, s)
        s, r, done = env.step(a)
        steps += 1
        if done:
            return True, steps
    return False, steps

def test_rl_cases():
    print("\n=== TEST RL (Q-learning, 5 skenario evaluasi) ===")
    random.seed(42)
    # Latih policy di environment default (Pac-Man bergerak menuju (0,0))
    env_train = MiniChaseEnv(size=7, pacman_moves=True)
    Q = q_learning(env_train, episodes=1000, alpha=0.5, gamma=0.95, eps=0.2, max_steps=60)

    tests = [
        {"name": "Skenario-1: posisi acak (default)", "ghost": None, "pac": None, "pac_moves": True},
        {"name": "Skenario-2: Ghost(6,6) vs Pac(0,0)", "ghost": (6,6), "pac": (0,0), "pac_moves": True},
        {"name": "Skenario-3: Ghost(3,3) vs Pac(6,6)", "ghost": (3,3), "pac": (6,6), "pac_moves": True},
        {"name": "Skenario-4: Pac diam di (5,5)",       "ghost": (0,0), "pac": (5,5), "pac_moves": False},
        {"name": "Skenario-5: Start dekat",             "ghost": (1,1), "pac": (1,2), "pac_moves": True},
    ]

    for t in tests:
        env_eval = MiniChaseEnv(size=7, pacman_moves=t["pac_moves"])
        # jalankan 10 episode untuk stabilkan rata-rata
        success = 0
        steps_list = []
        for _ in range(10):
            s0 = env_eval.reset(ghost_pos=t["ghost"], pac_pos=t["pac"])
            ok, steps = run_episode(env_eval, Q, max_steps=60)
            success += 1 if ok else 0
            steps_list.append(steps)
        avg_steps = sum(steps_list)/len(steps_list)
        print(f"{t['name']}: success {success}/10, rata-rata langkah {avg_steps:.1f}")

# =========================================================
# =============== 5) MAIN =================================
# =========================================================

if __name__ == "__main__":
    # Pengujian A* (5 kasus), MDP (5 skenario), RL (5 skenario)
    test_astar_cases()
    test_mdp_cases()
    test_rl_cases()
