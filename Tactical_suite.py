import srtm
import networkx as nx
import heapq
from lxml import etree
from shapely.geometry import Point, Polygon, LineString
import numpy as np
import math
import sys
import os
import time

# --- CONFIGURATION ---
KML_FILE_PATH = "MapTest3.kml" 
OUTPUT_KML_NAME = "Tactical_Final_Solution_v6.kml"
MAX_HOP_METERS = 3000   
GRID_RES = 0.001    

# --- ANTENNA HEIGHTS ---
RELAY_H = 2.0      # Your Drones/Relays height
ENEMY_H = 2.0     # Enemy sensors height (e.g., on a tower)
# -----------------------

print("--- INITIALIZING INTEGRATED TACTICAL SYSTEM ---")
elevation_data = srtm.get_data()

def get_elevation(p):
    return elevation_data.get_elevation(p.y, p.x) or 0

def get_3d_vector(p1, p2, h1, h2):
    """Calculates 3D vector between two points with specific antenna heights."""
    dx = (p2.x - p1.x) * 111320 * math.cos(math.radians(p1.y))
    dy = (p2.y - p1.y) * 111000
    dz = (get_elevation(p2) + h2) - (get_elevation(p1) + h1)
    return np.array([dx, dy, dz])

def calculate_3d_angle(v1, v2):
    mag1, mag2 = np.linalg.norm(v1), np.linalg.norm(v2)
    if mag1 == 0 or mag2 == 0: return 0
    return math.degrees(math.acos(np.clip(np.dot(v1, v2) / (mag1 * mag2), -1.0, 1.0)))

def calculate_2d_bearing(p1, p2):
    lat1, lon1 = math.radians(p1.y), math.radians(p1.x)
    lat2, lon2 = math.radians(p2.y), math.radians(p2.x)
    dLon = lon2 - lon1
    y = math.sin(dLon) * math.cos(lat2)
    x = math.cos(lat1) * math.sin(lat2) - math.sin(lat1) * math.cos(lat2) * math.cos(dLon)
    return (math.degrees(math.atan2(y, x)) + 360) % 360

def is_actually_hidden(relay_pt, threat_pt, h_relay, h_threat):
    """Checks LOS between a relay and a threat using their respective heights."""
    line = LineString([relay_pt, threat_pt])
    dist_deg = relay_pt.distance(threat_pt)
    samples = max(20, int((dist_deg * 111000) / 25))
    z1 = get_elevation(relay_pt) + h_relay
    z2 = get_elevation(threat_pt) + h_threat
    for i in range(1, samples):
        pt = line.interpolate(i/samples, normalized=True)
        if get_elevation(pt) > (z1 + (z2 - z1) * (i/samples)): return True 
    return False

def extract_kml_data(file_path):
    with open(file_path, 'rb') as f:
        tree = etree.parse(f)
    root = tree.getroot()
    ns = {"kml": root.tag.split('}')[0].strip('{')}
    data = {'poly': None, 'start': None, 'goal': None, 'threats': []}
    for pm in tree.xpath("//kml:Placemark", namespaces=ns):
        name = (pm.find("kml:name", namespaces=ns).text or "").lower()
        if pm.find(".//kml:Polygon", namespaces=ns) is not None:
            coord_text = pm.find(".//kml:coordinates", namespaces=ns).text.strip()
            coords = [tuple(map(float, c.split(',')[:2])) for c in coord_text.split()]
            data['poly'] = Polygon(coords)
        elif pm.find(".//kml:Point", namespaces=ns) is not None:
            coord_text = pm.find(".//kml:coordinates", namespaces=ns).text.strip()
            c = list(map(float, coord_text.split(',')[:2]))
            p = Point(c[0], c[1])
            if "start" in name: data['start'] = p
            elif "goal" in name: data['goal'] = p
            elif "see" in name or "threat" in name or "enemy" in name: data['threats'].append(p)
    return data

def main():
    kml_data = extract_kml_data(KML_FILE_PATH)
    poly, start, goal, threats = kml_data['poly'], kml_data['start'], kml_data['goal'], kml_data['threats']
    
    print("\n[SELECT MISSION MODE]")
    print("1: Standard BFS (Shortest Path) -> Red")
    print("2: User-Supplied Separation Constraint -> Yellow")
    print("3: Max-Min Bottleneck Optimization (Best Safety) -> Blue")
    mode = input("Enter Mode (1/2/3): ")

    path_color = "ff0000ff" if mode == '1' else ("ff00ffff" if mode == '2' else "ffff0000")
    t_start = time.perf_counter()

    # 1. Shadow Analysis
    candidate_nodes = [start, goal]
    minx, miny, maxx, maxy = poly.bounds
    x_steps = np.arange(minx, maxx, GRID_RES)
    y_steps = np.arange(miny, maxy, GRID_RES)
    
    print(f"\n[1/3] Analyzing shadows for {len(threats)} threats (Enemy H={ENEMY_H}m)...")
    for i, x in enumerate(x_steps):
        sys.stdout.write(f"\r      Progress: {int((i/len(x_steps))*100)}%")
        for y in y_steps:
            p = Point(x, y)
            if poly.contains(p) and all(is_actually_hidden(p, t, RELAY_H, ENEMY_H) for t in threats):
                candidate_nodes.append(p)
    
    orig_count = len(x_steps) * len(y_steps)
    safe_count = len(candidate_nodes) 
    t_masking = time.perf_counter() - t_start

    # 2. Solver Selection
    t_solve_start = time.perf_counter()
    print(f"\n[2/3] Processing Pathfinding (Mode {mode}, Relay H={RELAY_H}m)...")
    
    final_path = None
    max_achieved_angle = 0.0
    G = nx.Graph()
    adj = {i: [] for i in range(len(candidate_nodes))}
    for i in range(len(candidate_nodes)):
        for j in range(i + 1, len(candidate_nodes)):
            if candidate_nodes[i].distance(candidate_nodes[j]) * 111000 <= MAX_HOP_METERS:
                if not is_actually_hidden(candidate_nodes[i], candidate_nodes[j], RELAY_H, RELAY_H):
                    adj[i].append(j)
                    adj[j].append(i)

    if mode in ['1', '2']:
        for u, neighbors in adj.items():
            for v in neighbors: G.add_edge(u, v)
        if mode == '1':
            try: final_path = nx.shortest_path(G, 0, 1)
            except nx.NetworkXNoPath: pass
        else:
            threshold = float(input("Enter Min Separation Threshold (degrees): "))
            temp_G = G.copy()
            bad_edges = []
            for u, v in G.edges():
                v_beam = get_3d_vector(candidate_nodes[u], candidate_nodes[v], RELAY_H, RELAY_H)
                if any(calculate_3d_angle(v_beam, get_3d_vector(candidate_nodes[u], t, RELAY_H, ENEMY_H)) < threshold for t in threats):
                    bad_edges.append((u, v))
            temp_G.remove_edges_from(bad_edges)
            try: final_path = nx.shortest_path(temp_G, 0, 1)
            except nx.NetworkXNoPath: pass

    elif mode == '3':
        pq = [(-180.0, 0, [0])]
        best_bottleneck = {0: 180.0}
        while pq:
            neg_bottleneck, u, path = heapq.heappop(pq)
            current_bottleneck = -neg_bottleneck
            if u == 1:
                final_path, max_achieved_angle = path, current_bottleneck
                break
            for v in adj[u]:
                if v in path: continue
                v_beam = get_3d_vector(candidate_nodes[u], candidate_nodes[v], RELAY_H, RELAY_H)
                edge_min = 180.0
                for t in threats:
                    v_null = get_3d_vector(candidate_nodes[u], t, RELAY_H, ENEMY_H)
                    edge_min = min(edge_min, calculate_3d_angle(v_beam, v_null))
                new_path_bottleneck = min(current_bottleneck, edge_min)
                if new_path_bottleneck > best_bottleneck.get(v, 0):
                    best_bottleneck[v] = new_path_bottleneck
                    heapq.heappush(pq, (-new_path_bottleneck, v, path + [v]))

    t_solve = time.perf_counter() - t_solve_start

    # 3. Export
    if final_path:
        path_pts = [candidate_nodes[idx] for idx in final_path]
        total_dist = sum(path_pts[i].distance(path_pts[i+1])*111000 for i in range(len(path_pts)-1))
        
        # Fixed: Changed ANTENNA_H to RELAY_H
        line_coords = " ".join([f"{p.x},{p.y},{get_elevation(p) + RELAY_H}" for p in path_pts])
        content = f"<Placemark><name>Relay Path</name><Style><LineStyle><color>{path_color}</color><width>4</width></LineStyle></Style><LineString><altitudeMode>absolute</altitudeMode><coordinates>{line_coords}</coordinates></LineString></Placemark>"
        
        for i, p in enumerate(path_pts):
            lbl = "START" if i==0 else ("GOAL" if i==len(path_pts)-1 else f"Relay {i}")
            details = f"<b>Position:</b> {p.y:.5f}, {p.x:.5f}<br/>"
            
            neighbors = []
            if i > 0: neighbors.append(("PREVIOUS", path_pts[i-1]))
            if i < len(path_pts)-1: neighbors.append(("NEXT", path_pts[i+1]))

            for n_lbl, n_pt in neighbors:
                v_beam = get_3d_vector(p, n_pt, RELAY_H, RELAY_H)
                bearing = calculate_2d_bearing(p, n_pt)
                details += f"<br/><b>BEAM TO {n_lbl}:</b> {bearing:.1f}°<br/>"
                
                for j, t in enumerate(threats):
                    # Fixed: Uses ENEMY_H for the null calculation
                    v_null = get_3d_vector(p, t, RELAY_H, ENEMY_H)
                    sep_3d = calculate_3d_angle(v_beam, v_null)
                    details += f"&nbsp;&nbsp;Null Sep (Enemy {j+1}): <b>{sep_3d:.1f}°</b><br/>"

            content += f"<Placemark><name>{lbl}</name><description><![CDATA[{details}]]></description><Point><altitudeMode>absolute</altitudeMode><coordinates>{p.x},{p.y},{get_elevation(p)+RELAY_H}</coordinates></Point></Placemark>"
        
        for i, t in enumerate(threats):
            # Fixed: Enemy points placed at ENEMY_H
            content += f"<Placemark><name>ENEMY {i+1}</name><Style><IconStyle><color>ff0000ff</color></IconStyle></Style><Point><altitudeMode>absolute</altitudeMode><coordinates>{t.x},{t.y},{get_elevation(t)+ENEMY_H}</coordinates></Point></Placemark>"

        with open(OUTPUT_KML_NAME, "w") as f:
            f.write(f'<?xml version=\"1.0\" encoding=\"UTF-8\"?><kml xmlns=\"http://www.opengis.net/kml/2.2\"><Document>{content}</Document></kml>')
        print(f"\nSUCCESS: Saved to {OUTPUT_KML_NAME}")
    else:
        print("\nFAILURE: No path found.")

if __name__ == "__main__":
    main()