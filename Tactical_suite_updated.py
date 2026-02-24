import srtm
import networkx as nx
import heapq
from lxml import etree
from shapely.geometry import Point, Polygon, LineString
import numpy as np
import math
import sys
import time

# --- CONFIGURATION ---
KML_FILE_PATH = "MapTest3.kml" 
OUTPUT_KML_NAME = "Tactical_Integrated_Solution_Uniform_constraint30.kml"
MAX_HOP_METERS = 3000   
GRID_RES = 0.001        
ANTENNA_H = 2.0         

print("--- INITIALIZING SYSTEM ---")
elevation_data = srtm.get_data()

def get_elevation(p):
    return elevation_data.get_elevation(p.y, p.x) or 0

def get_3d_vector(p1, p2):
    dx = (p2.x - p1.x) * 111320 * math.cos(math.radians(p1.y))
    dy = (p2.y - p1.y) * 111000
    dz = (get_elevation(p2) + ANTENNA_H) - (get_elevation(p1) + ANTENNA_H)
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

def is_actually_hidden(relay_pt, threat_pt):
    line = LineString([relay_pt, threat_pt])
    dist_deg = relay_pt.distance(threat_pt)
    samples = max(20, int((dist_deg * 111000) / 25))
    z1, z2 = get_elevation(relay_pt) + ANTENNA_H, get_elevation(threat_pt) + ANTENNA_H
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
            elif any(x in name for x in ["can't see", "threat", "enemy"]): data['threats'].append(p)
    return data

def main():
    kml_data = extract_kml_data(KML_FILE_PATH)
    poly, start, goal, threats = kml_data['poly'], kml_data['start'], kml_data['goal'], kml_data['threats']
    
    print("\n[SELECT MISSION MODE]\n1: BFS (Shortest Path)\n2: Constraint (Threshold Angle)\n3: Max-Min (Widest Bottleneck)")
    mode = input("Enter Mode: ")
    
    # KML Colors (AABBGGRR)
    if mode == '1': path_color = "ff0000ff" # Red
    elif mode == '2': path_color = "ff00ffff" # Yellow
    elif mode == '3': path_color = "ffff0000" # Blue
    else: path_color = "ffffffff"

    t_masking_start = time.perf_counter()
    candidate_nodes = [start, goal]
    minx, miny, maxx, maxy = poly.bounds
    x_steps = np.arange(minx, maxx, GRID_RES)
    y_steps = np.arange(miny, maxy, GRID_RES)
    orig_count = len(x_steps) * len(y_steps)

    print(f"[1/3] Shadow Analysis (Uniform for all modes)...")
    for x in x_steps:
        for y in y_steps:
            p = Point(x, y)
            if poly.contains(p):
                # EVERY mode now enforces shadow masking here
                if all(is_actually_hidden(p, t) for t in threats):
                    candidate_nodes.append(p)
    
    safe_count = len(candidate_nodes)
    t_masking = time.perf_counter() - t_masking_start

    t_solve_start = time.perf_counter()
    G = nx.Graph()
    adj = {i: [] for i in range(len(candidate_nodes))}
    for i in range(len(candidate_nodes)):
        for j in range(i + 1, len(candidate_nodes)):
            if candidate_nodes[i].distance(candidate_nodes[j]) * 111000 <= MAX_HOP_METERS:
                if not is_actually_hidden(candidate_nodes[i], candidate_nodes[j]):
                    adj[i].append(j); adj[j].append(i)
                    G.add_edge(i, j)

    final_path = None
    max_achieved_angle = 0.0

    if mode == '1':
        try: final_path = nx.shortest_path(G, 0, 1)
        except: pass
    elif mode == '2':
        threshold = float(input("Enter Min Separation (°): "))
        G_con = G.copy()
        for u, v in G.edges():
            v12, v21 = get_3d_vector(candidate_nodes[u], candidate_nodes[v]), get_3d_vector(candidate_nodes[v], candidate_nodes[u])
            if any(calculate_3d_angle(v12, get_3d_vector(candidate_nodes[u], t)) < threshold for t in threats) or \
               any(calculate_3d_angle(v21, get_3d_vector(candidate_nodes[v], t)) < threshold for t in threats):
                G_con.remove_edge(u, v)
        try: final_path = nx.shortest_path(G_con, 0, 1)
        except: pass
    elif mode == '3':
        pq = [(-180.0, 0, [0])]
        best_bots = {0: 180.0}
        while pq:
            curr_bot, u, path = heapq.heappop(pq)
            curr_bot = -curr_bot
            if u == 1: 
                final_path = path; max_achieved_angle = curr_bot; break
            for v in adj[u]:
                if v in path: continue
                v12, v21 = get_3d_vector(candidate_nodes[u], candidate_nodes[v]), get_3d_vector(candidate_nodes[v], candidate_nodes[u])
                edge_min = min([calculate_3d_angle(v12, get_3d_vector(candidate_nodes[u], t)) for t in threats] + 
                               [calculate_3d_angle(v21, get_3d_vector(candidate_nodes[v], t)) for t in threats] + [180.0])
                new_bot = min(curr_bot, edge_min)
                if new_bot > best_bots.get(v, -1):
                    best_bots[v] = new_bot
                    heapq.heappush(pq, (-new_bot, v, path + [v]))

    t_solve = time.perf_counter() - t_solve_start

    if final_path:
        path_pts = [candidate_nodes[idx] for idx in final_path]
        total_dist = sum(path_pts[i].distance(path_pts[i+1])*111000 for i in range(len(path_pts)-1))
        
        # --- FINAL METRICS REPORT ---
        print("\n" + "="*45)
        print("MISSION PERFORMANCE METRICS")
        print(f"Nodes Generated (Original):  {orig_count}")
        print(f"Nodes in Safe Volume:        {safe_count}")
        print(f"Edges in Initial Graph:      {G.number_of_edges()}")
        print("-" * 45)
        print(f"Safety Bottleneck Angle:     {max_achieved_angle:.2f}°")
        print(f"Total Relay Hops:            {len(final_path)-1}")
        print(f"Total Drones Required:       {len(final_path)}")
        print(f"Total Path Distance:         {total_dist:.1f}m")
        print(f"Average Hop Length:          {total_dist/(len(final_path)-1):.1f}m")
        print("-" * 45)
        print(f"Compute Time (Masking):      {t_masking:.4f}s")
        print(f"Compute Time (Search):       {t_solve:.4f}s")
        print(f"Total Execution Time:        {t_masking + t_solve:.4f}s")
        print("="*45)

        line_coords = " ".join([f"{p.x},{p.y},{get_elevation(p)+ANTENNA_H}" for p in path_pts])
        content = f"<Placemark><name>Path</name><Style><LineStyle><color>{path_color}</color><width>4</width></LineStyle></Style><LineString><altitudeMode>absolute</altitudeMode><coordinates>{line_coords}</coordinates></LineString></Placemark>"
        
        for i, t in enumerate(threats):
            content += f"<Placemark><name>ENEMY {i+1}</name><Style><IconStyle><color>ff0000ff</color></IconStyle></Style><Point><coordinates>{t.x},{t.y},{get_elevation(t)}</coordinates></Point></Placemark>"

        for i, p in enumerate(path_pts):
            lbl = "START" if i==0 else ("GOAL" if i==len(path_pts)-1 else f"Relay {i}")
            details = f"<b>Position:</b> {p.y:.5f}, {p.x:.5f}<br/>"
            neighbors = []
            if i > 0: neighbors.append(("PREVIOUS", path_pts[i-1]))
            if i < len(path_pts)-1: neighbors.append(("NEXT", path_pts[i+1]))

            for n_lbl, n_pt in neighbors:
                v_beam = get_3d_vector(p, n_pt)
                bearing = calculate_2d_bearing(p, n_pt)
                details += f"<br/><b>BEAM TO {n_lbl}:</b> {bearing:.1f}°<br/>"
                for j, t in enumerate(threats):
                    v_null = get_3d_vector(p, t)
                    sep_3d = calculate_3d_angle(v_beam, v_null)
                    details += f"&nbsp;&nbsp;Null Sep (Enemy {j+1}): <b>{sep_3d:.1f}°</b><br/>"

            content += f"<Placemark><name>{lbl}</name><description><![CDATA[{details}]]></description><Point><altitudeMode>absolute</altitudeMode><coordinates>{p.x},{p.y},{get_elevation(p)+ANTENNA_H}</coordinates></Point></Placemark>"

        with open(OUTPUT_KML_NAME, "w") as f:
            f.write(f'<?xml version="1.0" encoding="UTF-8"?><kml xmlns="http://www.opengis.net/kml/2.2"><Document>{content}</Document></kml>')
        print(f"SUCCESS: Saved to {OUTPUT_KML_NAME}")
    else:
        print("\nFAILURE: No solution found with current shadow constraints.")

if __name__ == "__main__":
    main()