import sys
import math
import heapq
import random
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.widgets import CheckButtons 
from mpl_toolkits.mplot3d.art3d import Poly3DCollection
from shapely.geometry import Polygon, LineString, Point
import networkx as nx
import osmnx as ox

# --- CONFIGURATION ---
LOCATION = (29.7604, -95.3698) # Houston
#LOCATION = (34.0522, -118.2437)
RADIUS = 600                   
MIN_Z = 5                   
MAX_Z = 125.0                  
NUM_ENEMIES = 5
ENEMY_RANGE = 600.0            # Reduced slightly to ensure shadow paths exist
BUILDING_BUFFER = 5.0          
SIMPLIFY_TOLERANCE = 8.0       
MAX_LINK_DIST = 800.0   
ANTENNA_H = 30 

try:
    from shapely.strtree import STRTree
except ImportError:
    from shapely.strtree import STRtree as STRTree

# ==============================================================================
# 1. CLASSES & MATH
# ==============================================================================

class Enemy:
    def __init__(self, pos, r):
        self.pos = np.array(pos)
        self.range = r

class CityMap:
    def __init__(self):
        print(f"1. Loading Map Data for {LOCATION}...")
        self.gdf = ox.features_from_point(LOCATION, tags={"building": True}, dist=RADIUS).to_crs(epsg=32615)
        minx, miny, maxx, maxy = self.gdf.total_bounds
        self.cx, self.cy = (minx+maxx)/2, (miny+maxy)/2
        self.buildings = [] 
        self.obstacles_2d = []
        for _, row in self.gdf.iterrows():
            poly = row.geometry
            if poly.geom_type == 'MultiPolygon': poly = max(poly.geoms, key=lambda a: a.area)
            if poly.geom_type != 'Polygon': continue
            trans_poly = Polygon([(p[0]-self.cx, p[1]-self.cy) for p in poly.exterior.coords])
            h = 15.0 
            if 'height' in row and str(row['height']).replace('.','',1).isdigit(): h = float(row['height'])
            elif 'building:levels' in row and str(row['building:levels']).isdigit(): h = float(row['building:levels']) * 3.5 
            safe_poly = trans_poly.buffer(BUILDING_BUFFER, join_style=2)
            self.buildings.append({'poly': safe_poly, 'height': h, 'bbox': safe_poly.bounds})
            self.obstacles_2d.append(safe_poly)
        self.tree = STRTree(self.obstacles_2d)

    def check_los_3d(self, p1, p2):
        line_2d = LineString([(p1[0], p1[1]), (p2[0], p2[1])])
        candidates = self.tree.query(line_2d)
        total_dist_2d = math.dist((p1[0], p1[1]), (p2[0], p2[1]))
        if total_dist_2d < 1e-3: return True
        for idx in candidates:
            b = self.buildings[idx]
            if p1[2] > b['height'] and p2[2] > b['height']: continue
            if line_2d.intersects(b['poly']):
                intersection = line_2d.intersection(b['poly'])
                parts = [intersection] if not hasattr(intersection, 'geoms') else list(intersection.geoms)
                for part in parts:
                    if part.geom_type not in ['LineString', 'Point']: continue
                    for coord in (part.coords if part.geom_type == 'LineString' else [part.coords[0]]):
                        d_at_pt = math.dist((p1[0], p1[1]), coord)
                        z_at_pt = p1[2] + (d_at_pt / total_dist_2d) * (p2[2] - p1[2])
                        if z_at_pt < b['height']: return False 
        return True

def get_3d_vector(p1, p2):
    return np.array([p2[0] - p1[0], p2[1] - p1[1], p2[2] - p1[2]])

def calculate_3d_angle(v1, v2):
    mag1, mag2 = np.linalg.norm(v1), np.linalg.norm(v2)
    if mag1 == 0 or mag2 == 0: return 0
    return math.degrees(math.acos(np.clip(np.dot(v1, v2) / (mag1 * mag2), -1.0, 1.0)))

# ==============================================================================
# 2. SOLVERS
# ==============================================================================

def solve_min_link_path(nodes, s_idx, e_idx, shadow_indices, city):
    """Path 1: Min hops using hidden relays. Beams are allowed to be seen."""
    G = nx.Graph()
    
    # We MUST include start and goal indices in the graph, 
    active_indices = list(set(shadow_indices) | {s_idx, e_idx})
    
    for idx in active_indices:
        G.add_node(idx)
        
    for i in active_indices:
        for j in active_indices:
            if i >= j: continue
            
            dist = math.dist(nodes[i], nodes[j])
            # Check ONLY for building collisions. 
            # Signals can pass through enemy range.
            if dist < MAX_LINK_DIST and city.check_los_3d(nodes[i], nodes[j]):
                # 1.0 weight for Min-Link logic
                G.add_edge(i, j, weight=1.0 + (dist/1e7))
                
    try:
        path_idxs = nx.astar_path(G, s_idx, e_idx, weight='weight')
        return [nodes[i] for i in path_idxs]
    except (nx.NetworkXNoPath, nx.NodeNotFound):
        return None

def solve_max_min_stealth_path(nodes, s_idx, e_idx, shadow_indices, enemies, city):
    """Path 2: Max-Min logic for safest beam orientation between hidden relays."""
    active_indices = list(set(shadow_indices) | {s_idx, e_idx})
    
    # Priority Queue: (-bottleneck_angle, current_idx, path_list)
    pq = [(-180.0, s_idx, [s_idx])]
    best_bottleneck = {s_idx: 180.0}
    final_path = None
    max_angle = 0

    while pq:
        neg_bottleneck, u_idx, path = heapq.heappop(pq)
        curr_bottleneck = -neg_bottleneck
        
        if u_idx == e_idx:
            final_path = [nodes[i] for i in path]
            max_angle = curr_bottleneck
            break

        p1 = nodes[u_idx]
        for v_idx in active_indices:
            if v_idx in path: continue
            p2 = nodes[v_idx]
            
            dist = math.dist(p1, p2)
            # Physical LOS check only (Buildings block, Enemies don't)
            if dist < MAX_LINK_DIST and city.check_los_3d(p1, p2):
                v_12, v_21 = get_3d_vector(p1, p2), get_3d_vector(p2, p1)
                
                edge_min = 180.0
                for e in enemies:
                    ang1 = calculate_3d_angle(v_12, get_3d_vector(p1, e.pos))
                    ang2 = calculate_3d_angle(v_21, get_3d_vector(p2, e.pos))
                    edge_min = min(edge_min, ang1, ang2)
                
                new_bottleneck = min(curr_bottleneck, edge_min)
                if new_bottleneck > best_bottleneck.get(v_idx, -1.0):
                    best_bottleneck[v_idx] = new_bottleneck
                    heapq.heappush(pq, (-new_bottleneck, v_idx, path + [v_idx]))
                    
    return final_path, max_angle

# ==============================================================================
# 3. VISUALIZATION & MAIN
# ==============================================================================

def visualize_combined(city, enemies, path_min, path_mm, foxholes, nodes, start_pt, end_pt, b_angle):
    fig = plt.figure(figsize=(16, 10))
    ax = fig.add_subplot(1, 1, 1, projection='3d')
    
    # Title with degrees and reference to 3D space
    ax.set_title(f"Shadow Relay Analysis | Max-Min Bottleneck: {b_angle:.2f}°", fontsize=14)
    
    # Axis labels in Meters
    ax.set_xlabel("X (meters)")
    ax.set_ylabel("Y (meters)")
    ax.set_zlabel("Altitude (meters)")

    # --- 1. BUILDINGS ---
    for b in city.buildings:
        x, y = b['poly'].exterior.coords.xy
        h = b['height']
        for i in range(len(x)-1):
            ax.add_collection3d(Poly3DCollection([[(x[i],y[i],0),(x[i+1],y[i+1],0),(x[i+1],y[i+1],h),(x[i],y[i],h)]], 
                                                 facecolors='gray', alpha=0.3, edgecolors='k', linewidths=0.1))
        ax.add_collection3d(Poly3DCollection([list(zip(x,y,[h]*len(x)))], facecolors='dimgray', alpha=0.5))

    # --- 2. ENEMIES & RANGE SPHERES ---
    for i, e in enumerate(enemies):
        lbl = f"Enemy Sensors ({ENEMY_RANGE}m range)" if i == 0 else None
        ax.scatter(*e.pos, c='red', s=100, marker='X', edgecolors='white', zorder=5, label=lbl)
        u, v = np.mgrid[0:2*np.pi:20j, 0:np.pi:10j]
        xs, ys = e.pos[0] + ENEMY_RANGE*np.cos(u)*np.sin(v), e.pos[1] + ENEMY_RANGE*np.sin(u)*np.sin(v)
        zs = np.maximum(e.pos[2] + ENEMY_RANGE * np.cos(v), 0)
        ax.plot_wireframe(xs, ys, zs, color="red", alpha=0.15, linewidth=0.5)

    # --- 3. NODES ---
    fx, fy, fz = zip(*foxholes)
    scat_fox = ax.scatter(fx, fy, fz, c='black', s=10, alpha=0.2, label='Shadow Nodes')

    # --- 4. PATHS ---
    p_min_els, p_mm_els = [], []
    if path_min:
        px, py, pz = zip(*path_min)
        p_min_els.append(ax.plot(px, py, pz, color='green', linewidth=3, label='Shadow Relay (Min-Hops)', zorder=10)[0])
        p_min_els.append(ax.scatter(px, py, pz, c='green', marker='o', s=80, edgecolors='k'))
    if path_mm:
        px, py, pz = zip(*path_mm)
        p_mm_els.append(ax.plot(px, py, pz, color='gold', linewidth=5, label='Stealth Relay (Max-Min)', zorder=12)[0])
        p_mm_els.append(ax.scatter(px, py, pz, c='white', marker='*', s=150, edgecolors='black'))

    # DYNAMIC ZOOM CALCULATION
    all_active = [start_pt, end_pt] + [e.pos for e in enemies]
    ax_b, ay_b = [p[0] for p in all_active], [p[1] for p in all_active]
    margin = 100
    span = max(max(ax_b)-min(ax_b), max(ay_b)-min(ay_b)) + margin
    mid_x, mid_y = (max(ax_b)+min(ax_b))/2, (max(ay_b)+min(ay_b))/2
    
    # 200m set as Z-limit for visualization clarity
    ax.set_box_aspect((span, span, 200)) 
    ax.set_xlim(mid_x-span/2, mid_x+span/2); ax.set_ylim(mid_y-span/2, mid_y+span/2); ax.set_zlim(0, 200)
    
    ax.scatter(*start_pt, c='green', s=250, label='START', marker='P', edgecolors='white')
    ax.scatter(*end_pt, c='blue', s=250, label='GOAL', marker='H', edgecolors='white')
    ax.legend(loc='upper left', frameon=True, fontsize=9)

    # GUI Toggles
    ax_check = plt.axes([0.01, 0.4, 0.15, 0.25])
    check = CheckButtons(ax_check, ['Shadow Nodes', 'Min-Link', 'Stealth'], [True, True, True])
    def toggle(label):
        if label == 'Shadow Nodes': scat_fox.set_visible(not scat_fox.get_visible())
        elif label == 'Min-Link': [e.set_visible(not e.get_visible()) for e in p_min_els]
        elif label == 'Stealth': [e.set_visible(not e.get_visible()) for e in p_mm_els]
        plt.draw()
    check.on_clicked(toggle)
    plt.show()

if __name__ == "__main__":
    city = CityMap()
    plt.figure(figsize=(7,7))
    plt.title("Pick Start then Goal")
    for obs in city.obstacles_2d: plt.fill(*obs.exterior.coords.xy, color='gray', alpha=0.5)
    pts = plt.ginput(2, timeout=-1); plt.close()
    
    # Antenna height 2m
    start = (pts[0][0], pts[0][1], ANTENNA_H)
    end = (pts[1][0], pts[1][1], ANTENNA_H)

    # 1. ENEMY SPAWNER: Ensuring they CANNOT see Start or Goal
    enemies = []
    print("2. Spawning Enemies in locations hidden from Start/Goal...")
    attempts = 0
    while len(enemies) < NUM_ENEMIES and attempts < 2000:
        attempts += 1
        ex = random.uniform(-RADIUS*0.8, RADIUS*0.8)
        ey = random.uniform(-RADIUS*0.8, RADIUS*0.8)
        ez = random.uniform(10, 40)
        candidate_pos = (ex, ey, ez)
        
        # Check if inside building
        if any(b['poly'].contains(Point(ex, ey)) for b in city.buildings): continue
        
        # STRICT REQUIREMENT: Enemy must NOT have LOS to Start AND must NOT have LOS to Goal
        if not city.check_los_3d(candidate_pos, start) and not city.check_los_3d(candidate_pos, end):
            enemies.append(Enemy(candidate_pos, ENEMY_RANGE))

    # 2. NODE GENERATION
    nodes = [start, end]
    for b in city.buildings:
        if b['height'] < MIN_Z: continue
        simple = b['poly'].simplify(SIMPLIFY_TOLERANCE)
        for x, y in simple.exterior.coords[:-1]:
            nodes.append((x, y, MIN_Z)); nodes.append((x, y, MAX_Z))

    s_idx, e_idx = 0, 1
    
    # 3. SHADOW MASKING: Only nodes 100% hidden from the spawned enemies
    shadow_indices = []
    for i, n in enumerate(nodes):
        is_exposed = False
        for e in enemies:
            if city.check_los_3d(n, e.pos):
                is_exposed = True
                break
        if not is_exposed:
            shadow_indices.append(i)

    # Final Check: Are Start and Goal actually safe?
    if s_idx not in shadow_indices or e_idx not in shadow_indices:
        print("\n[!] WARNING: Start or Goal is exposed to an enemy. No safe path possible.")
    else:
        path_min = solve_min_link_path(nodes, s_idx, e_idx, shadow_indices, city)
        path_mm, b_angle = solve_max_min_stealth_path(nodes, s_idx, e_idx, shadow_indices, enemies, city)
        
        # Mission Results Printout...
        visualize_combined(city, enemies, path_min, path_mm, [nodes[i] for i in shadow_indices], nodes, start, end, b_angle)