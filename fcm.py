import math
import numpy as np
import networkx as nx
import matplotlib.pyplot as plt
from random import randint

class FCM_class:

    # visualize the graph
    def visualize(graph, desc_main, desc_nodes):
        pos = nx.spring_layout(graph)
        colors = []
        for i in range(len(graph.nodes)):
            colors.append('#%06X' % randint(0, 0xFFFFFF))
        nx.draw(graph, pos, with_labels=True, node_size=900, node_color=colors)
        labels = nx.get_edge_attributes(graph, 'weight')
        nx.draw_networkx_edge_labels(graph, pos, edge_labels=labels)

        legend_labels = []
        i = 0
        for _, desc in desc_nodes.items():
            legend_labels.append(plt.Circle((0, 0), 0.1, color=colors[i], label=f"{i}: {desc}"))
            i+=1
        plt.legend(handles=legend_labels, loc='upper left', title=desc_main)
    
        plt.show()
    

    # create a graph from a matrix
    def fcm_from_matrix_to_graph(WW, Al, depth, iterations):
        "Create a graph based on a given matrix"
        G = nx.DiGraph(depth=depth)
        n = WW.shape[0]

        # nodes
        for k in range(n):
            # "value" is an array representing the activation level through the iterations
            # "link" is the index of the graph to which the node is linked
            G.add_node(k, attr_dict = {"value":[0]*iterations, "link":Al[k][1]})
            G.nodes[k]['attr_dict']['value'][0] = round(Al[k][0], 5)

        # edges
        for i in range(n):
            for j in range(n):
                if (WW[i][j] != 0): G.add_edge(i, j, weight = round(WW[i][j], 5))

        return G
    
    
    # sigmoid function
    def sigmoid(x, lambda_value):    
        return  1/(1+math.exp(-lambda_value*x))
    

    # algorithm from Cyprus
    def cyprus_fcm_alg_graph(graph_list, g_index, start_iter, end_iter, lambda_value):
        G = graph_list[g_index]

        for t in range(start_iter, end_iter):   #for each iteration
            for node in G:  #for each node in the graph

                # contribution of the linked node
                # recursive call of the algorithm for the linked graph
                if G.nodes[node]['attr_dict']['link'] > 0:
                    node_attr_links = int(G.nodes[node]['attr_dict']['link'])
                    graph_list[node_attr_links].nodes[0]['attr_dict']['value'][t-1] = G.nodes[node]['attr_dict']['value'][t-1]
                    graph_list = FCM_class.cyprus_fcm_alg_graph(graph_list, node_attr_links, t, t+1, lambda_value)
                    G.nodes[node]['attr_dict']['value'][t-1] = graph_list[node_attr_links].nodes[0]['attr_dict']['value'][t]

                # contribution of the incoming edges
                b = 0   # B_i^t
                for edge in G.in_edges(node):   #for each incoming edge
                    other, _ = edge
                    w_edge = G[other][node]['weight']
                    other_attr = G.nodes[other]['attr_dict']['value']

                    if t == 1:
                        al_old = 0
                    else:
                        al_old = other_attr[t-2]   # -2 because start_iter is 1 and not 0

                    al_new = other_attr[t-1]    # A_j^t
                    # al_new = A_j^t
                    # al_old = A_j^(t-1)
                    d_al = (al_new - al_old)    # delta A_j^t
                    
                    b += (w_edge * d_al)    # B_i^t = sum(w_ji * delta A_j^t)
                
                al_node = G.nodes[node]['attr_dict']['value'][t-1] # A_i^t
                if al_node != 0:
                    c = (np.log((1-al_node)/al_node))/-lambda_value  # C_i^t = log((1-A_i^t)/A_i^t) / -lambda
                else:
                    c = 0
                
                in_factor = 1.2
                x = in_factor * b + c   # x = 1.2 * B_i^t + C_i^t
                
                if b == 0:
                    final_al = G.nodes[node]['attr_dict']['value'][t-1]
                else:
                    final_al = round(FCM_class.sigmoid(x, lambda_value), 5) # A_i^t = sigmoid(x)
                
                G.nodes[node]['attr_dict']['value'][t] = final_al
            
        graph_list[g_index] = G

        return graph_list
    

    # algorithm from kosko
    def kosko_alg_graph(graph_list, g_index, start_iter, end_iter, lambda_value):
        G = graph_list[g_index]

        for t in range(start_iter,end_iter):    #for each iteration
            for node in G:  #for each node in the graph

                # contribution of the linked node
                # recursive call of the algorithm for the linked graph
                if G.nodes[node]['attr_dict']['link'] > 0:
                    node_attr_links = int(G.nodes[node]['attr_dict']['link'])
                    graph_list[node_attr_links].nodes[0]['attr_dict']['value'][t-1] = G.nodes[node]['attr_dict']['value'][t-1]
                    graph_list = FCM_class.kosko_alg_graph(graph_list, node_attr_links, t, t+1, lambda_value)
                    G.nodes[node]['attr_dict']['value'][t-1] = graph_list[node_attr_links].nodes[0]['attr_dict']['value'][t]

                # contribution of the incoming edges of the node i (B_i^t)
                b = 0
                for edge in G.in_edges(node):   #for each incoming edge
                    # i is the node index
                    # j is the index of the node from which the edge comes
                    other, _ = edge
                    w_edge = G[other][node]['weight']   # edge weight
                    other_attr = G.nodes[other]['attr_dict']['value']   # other node attributes
                    
                    b += w_edge * other_attr[t-1]   # B_i^t = sum(w_ji * A_j^(t-1))
            
                final_al = round(FCM_class.sigmoid(b, lambda_value), 5)  # A_i^t = sigmoid(B_i^t)
                
                G.nodes[node]['attr_dict']['value'][t] = final_al
                
        graph_list[g_index] = G      
        return graph_list
    

    # algorithm from Stylios and Groumpos
    def stylios_groumpos_alg_graph(graph_list, g_index, start_iter, end_iter, lambda_value):
        G = graph_list[g_index]
            
        for t in range(start_iter,end_iter):    #for each iteration
            for node in G:  #for each node in the graph

                # contribution of the linked node
                # recursive call of the algorithm for the linked graph
                if G.nodes[node]['attr_dict']['link'] > 0:
                    node_attr_links = int(G.nodes[node]['attr_dict']['link'])
                    graph_list[node_attr_links].nodes[0]['attr_dict']['value'][t-1] = G.nodes[node]['attr_dict']['value'][t-1]
                    graph_list = FCM_class.stylios_groumpos_alg_graph(graph_list, node_attr_links, t, t+1, lambda_value)
                    G.nodes[node]['attr_dict']['value'][t-1] = graph_list[node_attr_links].nodes[0]['attr_dict']['value'][t]

                # contribution of the incoming edges
                b = 0
                for edge in G.in_edges(node):   #for each incoming edge
                    other, _ = edge
                    w_edge = G[other][node]['weight']
                    other_attr = G.nodes[other]['attr_dict']['value']
                    
                    b += w_edge * other_attr[t-1]   # B_i^t = sum(w_ij * A_j^(t-1))

                al_node = G.nodes[node]['attr_dict']['value'][t-1]

                b = b + al_node
            
                final_al = round(FCM_class.sigmoid(b, lambda_value), 5)
                
                G.nodes[node]['attr_dict']['value'][t] = final_al
                
        graph_list[g_index] = G      
        return graph_list
