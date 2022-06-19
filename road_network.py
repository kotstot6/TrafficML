

import networkx as nx
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import os

class Node:

    def __init__(self, node_dict):

        self.id = node_dict['osmid']
        self.location = (float(node_dict['x']), float(node_dict['y']))

class Edge:

    def __init__(self, node1, node2, edge_dict):

        self.id = (node1.id, node2.id)
        self.nodes = (node1, node2)
        self.road = edge_dict['name'] if 'name' in edge_dict else 'No Name'
        self.distance = float(edge_dict['length'])
        self.lanes = self.lanes_to_int(edge_dict['lanes']) if 'lanes' in edge_dict else None
        self.speed_limit = self.speed_to_int(edge_dict['maxspeed']) if 'maxspeed' in edge_dict else None
        self.direction = self.get_direction()
        self.station = None

    def lanes_to_int(self,lanes):
        try:
            return int(lanes)
        except:
            try:
                return int(list(lanes)[0])
            except:
                return None

    def speed_to_int(self, speed):
        try:
            return int(speed.split(' ')[0])
        except:
            try:
                return int(list(speed)[0].split(' ')[0])
            except:
                return None

    def get_direction(self):

        node1, node2 = self.nodes
        (x1, y1), (x2, y2) = node1.location, node2.location

        return ('E' if x2 - x1 >= 0 else 'W', 'N' if y2 - y1 > 0 else 'S')

    def location(self):
        return (np.array(self.nodes[0].location) + np.array(self.nodes[1].location)) / 2

    def arctan(self):
        (x1, y1), (x2, y2) = self.nodes[0].location, self.nodes[1].location
        return np.arctan((y2 - y1) / (x2 - x1)) if x2 != x1 else np.pi/2 if y2 > y1 else -np.pi/2

    def angle(self):
        (x1, y1), (x2, y2) = self.nodes[0].location, self.nodes[1].location
        y_up = y2 - y1 > 0
        atan = np.arctan((y2 - y1) / (x2 - x1)) if x2 != x1 else np.pi / 2
        return atan + np.pi if atan < 0 and y_up else atan - np.pi if atan > 0 and not y_up else atan

    def get_turns(self, edge, type=1):

        angle1, angle2 = self.angle(), edge.angle()

        angle_diff = angle2 - angle1
        angle_diff = -(2*np.pi - angle_diff) if angle_diff > np.pi else 2*np.pi + angle_diff if angle_diff < -np.pi else angle_diff

        turns = []

        id = self.id if type == 1 else edge.id

        if angle_diff >= 0:
            hor_diff = angle_diff - np.pi/2
            turns.append((id, 'L' + str(type), np.abs(hor_diff)))
        else:
            angle_diff = np.abs(angle_diff)
            hor_diff = angle_diff - np.pi/2
            turns.append((id, 'R' + str(type), np.abs(hor_diff)))

        if hor_diff >= 0:
            turns.append((id, 'B', np.pi - angle_diff))
        else:
            turns.append((id, 'F' + str(type), angle_diff))

        return turns

    def speed_div_dist(self, avg_speed=20):
        return self.speed_limit / self.distance if self.speed_limit is not None else avg_speed / self.distance

    def dist_times_lanes(self, avg_lanes=1):
        return self.distance * self.lanes if self.lanes is not None else self.distance * avg_lanes

class Station:

    def __init__(self, id, dir, loc, vol):

        self.id = id + '-' + dir
        self.location = loc
        self.direction = dir
        self.volume = vol



    def is_within(self, edge, ep_x=1e-3, ep_y=1e-3):

        x,y = self.location
        n1, n2 = edge.nodes
        (x1, y1), (x2, y2) = n1.location, n2.location

        return x >= min(x1, x2) - ep_x and x <= max(x1, x2) + ep_x and y >= min(y1, y2) - ep_y and y <= max(y1, y2) + ep_y

    def distance_from(self, edge, MAX=10):

        x,y = self.location
        n1, n2 = edge.nodes
        (x1, y1), (x2, y2) = n1.location, n2.location

        if x1 == x2:
            if y1 == y2:
                if x == x1 and y == y1:
                    return 0
                else:
                    return MAX
            m = (x2 - x1)/(y2 - y1)
            x_hat = m * (y - y1) + x1
            dist = abs(x_hat - x)
        else:
            m = (y2 - y1)/(x2 - x1)
            y_hat = m * (x - x1) + y1
            dist = abs(y_hat - y)

        return dist


class RoadNetwork:

    def __init__(self, graphml_file):

        G = nx.DiGraph(nx.read_graphml(graphml_file))

        self.nx_graph = G

        self.nodes = {node_id : Node(G.nodes.data()[node_id]) for node_id in list(G.nodes)}
        self.compute_node_centralities()

        self.edges = { (n1_id, n2_id) :  Edge(self.nodes[n1_id], self.nodes[n2_id], edge_dict)
                            for n1_id, n2_id, edge_dict in list(G.edges.data())}
        self.compute_edge_centralities()
        self.compute_edge_degrees()

        vol_df = self.make_vol_df()

        self.stations = [
            Station(id, dir, loc, vol)
            for id, dir, loc, vol in zip(vol_df['Station ID'],vol_df['Direction'],
                                        vol_df['Location'], vol_df['Volume'])
        ]
        self.compute_station_edges()

        self.compute_count_map()

    def make_vol_df(self):

        columns = ['RCSTA', 'Federal Direction', 'Latitude', 'Longitude', 'Specific Recorder Placement']
        hours = ['1200am-0100am', '0100am-0200am', '0200am-0300am', '0300am-0400am', '0400am-0500am', '0500am-0600am',
                '0600am-0700am', '0700am-0800am', '0800am-0900am', '0900am-1000am', '1000am-1100am', '1100am-1200pm',
            '1200pm-0100pm', '0100pm-0200pm', '0200pm-0300pm', '0300pm-0400pm', '0400pm-0500pm', '0500pm-0600pm',
                '0600pm-0700pm', '0700pm-0800pm', '0800pm-0900pm', '0900pm-1000pm', '1000pm-1100pm', '1100pm-1200pm']
        direct_map = {'Northbound' : 'N', 'Southbound' : 'S', 'Eastbound' : 'E', 'Westbound' : 'W', 'Combined Total' : 'C'}


        vol_dict = {'Station ID' : [], 'Direction' : [], 'Location' : [], 'Volume' : []}

        for i in range(10):
            df = pd.read_csv('data/sc_network' + str(i) + '.csv')
            for row in zip(*[df[cat] for cat in columns + hours]):

                stat_id, fed_direct, lat, long, plac = row[:5]
                vol = list(row[5:])

                stat_id = str(stat_id)
                if stat_id[0] != '4':
                    continue

                loc = (float(long), float(lat))

                plac_elems = [p for p in plac.split(' ') if p != '']
                plac_direct = plac_elems[2][0]
                orient = 'H' if plac_direct in ['W', 'E'] else 'V'

                direct = direct_map[fed_direct]

                if direct == 'C':

                    if stat_id in vol_dict['Station ID']:
                        continue

                    # split up combined total
                    directs = ['W', 'E'] if orient == 'H' else ['N', 'S']
                    vol_dict['Station ID'] += [stat_id] * 2
                    vol_dict['Direction'] += directs
                    vol_dict['Location'] += [loc] * 2
                    vol_1 = [round(v / 2) for v in vol]
                    vol_2 = [v - v1 for v, v1 in zip(vol, vol_1)]
                    vol_dict['Volume'] += [vol_1, vol_2]
                else:

                    if (stat_id, direct) in zip(vol_dict['Station ID'], vol_dict['Direction']):
                        continue

                    vol_dict['Station ID'].append(stat_id)
                    vol_dict['Direction'].append(direct)
                    vol_dict['Location'].append(loc)
                    vol_dict['Volume'].append(vol)

        return pd.DataFrame(vol_dict)


    def compute_node_centralities(self):

        # Betweeness
        print('Calculating Betweenness...')
        bt_cent_dict = nx.betweenness_centrality(self.nx_graph)

        # Closeness
        print('Calculating Closeness...')
        cl_cent_dict = nx.closeness_centrality(self.nx_graph)

        for node_id in self.nodes:
            self.nodes[node_id].betweenness = bt_cent_dict[node_id]
            self.nodes[node_id].closeness = cl_cent_dict[node_id]


    def compute_edge_centralities(self):

        for n1, n2 in self.edges:
            self.nx_graph.edges[n1, n2]['distance'] = self.edges[(n1, n2)].distance

        # Betweenness
        print('Calculating Edge Betweenness...')
        bt_cent_dict = nx.edge_betweenness_centrality(self.nx_graph)#, weight='distance')

        for n1_id, n2_id in self.edges:
            self.edges[(n1_id, n2_id)].betweenness = bt_cent_dict[(n1_id, n2_id)]

    def compute_edge_degrees(self):

        in_deg_dict = self.nx_graph.in_degree()
        out_deg_dict = self.nx_graph.out_degree()

        for (n1, n2), edge in self.edges.items():
            edge.in_deg = in_deg_dict[n1]
            edge.out_deg = out_deg_dict[n2]


    def compute_edge_adjacency(self):

        adj_dict = {node : list(neighbors.keys()) for node, neighbors in list(self.nx_graph.adjacency())}

        for edge_id, edge in self.edges.items():

            directs = ['F1', 'L1', 'R1', 'F2', 'L2', 'R2', 'B']
            turns_dict = {direct : None for direct in directs}

            out_node_id = edge.nodes[1].id
            out_edges = [self.edges[(out_node_id, neighbor)] for neighbor in adj_dict[out_node_id]]

            in_node_id = edge.nodes[0].id
            in_edges = [self.edges[(neighbor, in_node_id)] for neighbor in adj_dict if in_node_id in adj_dict[neighbor]]

            turns = []

            for out_edge in out_edges:
                turns += edge.get_turns(out_edge, type=2)

            for in_edge in in_edges:
                turns += in_edge.get_turns(edge, type=1)

            turns.sort(key=lambda x: x[2])

            for edge_id, direct, _ in turns:
                if turns_dict[direct] is None and self.edges[edge_id] not in turns_dict.values():
                    turns_dict[direct] = self.edges[edge_id]

            edge.adjacency = turns_dict


    def compute_station_edges(self):

        for station in self.stations:
            possible_edges = []
            for edge in self.edges.values():
                if station.direction in list(edge.direction) and station.is_within(edge):
                    possible_edges.append((edge, station.distance_from(edge)))

            station.edge = min(possible_edges, key=lambda x: x[1])[0] if possible_edges else None

            if station.edge is not None:
                self.edges[station.edge.id].station = station

        self.stations = [station for station in self.stations if station.edge is not None]

    def compute_count_map(self):

        count_dict = {}

        for station in self.stations:
            for count in station.volume:
                c_str = str(count)
                if c_str not in count_dict:
                    count_dict[c_str] = 0
                count_dict[c_str] += 1

        count_map = [0 for i in range(max([int(c_str) for c_str in count_dict]) + 1)]

        total = len(self.stations) * 24
        run_total = 0

        for i in range(len(count_map)):
            count_map[i] = run_total + (count_dict[str(i)] if str(i) in count_dict else 0)
            run_total = count_map[i]
            count_map[i] /= total

        self.count_map = count_map

    def get_color(self, volume, color1=(0,1,0), color2=(1,1,0), color3=(1,0,0)):

        t = self.count_map[min(len(self.count_map)-1, volume)]

        return tuple([
            self.linear(2*t, v1, v2) if t <= 0.5 else self.linear(2*(t - 0.5), v2, v3)
            for v1, v2, v3 in zip(list(color1), list(color2), list(color3))
        ] + [1])

    def linear(self, t, v1, v2):

        return v1 * (1 - t) + v2 * t

    def animate(self, edge_vols, name):

        os.mkdir('animations/' + name)

        for time in range(24):
            print('Drawing time =', time)
            path = 'animations/' + name + '/' + str(time)
            self.draw(time, edge_vols, path=path)

    def draw(self, time, edge_vols, path=None):

        edge_colors = [self.get_color(edge_vols[edge][time]) for edge in self.edges]

        G = nx.DiGraph(self.nx_graph)
        G.add_nodes_from([station.id for station in self.stations])

        node_colors = [(0.5, 0.5, 0.5, 0.5)] * len(self.nodes) + [self.get_color(stat.volume[time]) for stat in self.stations]
        node_pos = {node.id : node.location for node in list(self.nodes.values()) + self.stations}
        node_sizes = [1e-3] * len(self.nodes) + [10] * len(self.stations)

        fig = plt.figure(1,figsize=(12,12))

        nx.draw(G, pos=node_pos, node_size=node_sizes, width=0.5, arrowsize=1e-3, node_color=node_colors, edge_color=edge_colors)

        if path is not None:
            plt.savefig(path, facecolor=fig.get_facecolor())

        plt.show()
