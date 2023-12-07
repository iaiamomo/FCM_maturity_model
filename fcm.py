import math
import numpy as np
import networkx as nx
import matplotlib.pyplot as plt

class FCM_class:

    # visualize the graph
    def visualize(graph):
        pos = nx.spring_layout(graph)
        nx.draw(graph, pos, with_labels=True, node_size=900, node_color='skyblue', font_weight='bold')
        labels = nx.get_edge_attributes(graph, 'weight')
        nx.draw_networkx_edge_labels(graph, pos, edge_labels=labels)
        plt.show()
    

    # create a graph from a matrix
    def fcm_from_matrix_to_graph(WW, Al, depth, iterations):
        "Create a graph based on a given matrix"
        G = nx.DiGraph(depth=depth)
        n = WW.shape[0]

        # nodes
        for k in range(n):
            G.add_node(k, attr_dict = {"value":[0]*iterations, "link":Al[k]})
            G.nodes[k]['attr_dict']['value'][0] = round(Al[k], 5)

        # edges
        for i in range(n):
            for j in range(n):
                if (WW[i][j] != 0): G.add_edge(i, j, weight = round(WW[i][j], 5))

        return G
    
    # sigmoid function
    def sigmoid(x, lambda_value):    
        return  1/(1+math.exp(-lambda_value*x))
    

    # algorithm from cyprus
    def aa_fcm_alg_graph(graph_list, start_iter, end_iter, lambda_value):
        
        for i in range(len(graph_list)):
            G = graph_list[i]

            for t in range(start_iter, end_iter):   #for each iteration
                for node in G:  #for each node in the graph
                    node_attributes = G.nodes[node]['attr_dict']['value']

                    # contributo degli archi entranti
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
                        
                        b += (w_edge * d_al)    # B_i^t = sum(w_ij * delta A_j^t)
                    
                    al_node = node_attributes[t-1] # A_i^t
                    if al_node != 0:
                        c = (np.log((1-al_node)/al_node))/-lambda_value  # C_i^t = log((1-A_i^t)/A_i^t) / -lambda
                    else:
                        c = 0
                    
                    in_factor = 1.2
                    x = in_factor * b + c   # x = 1.2 * B_i^t + C_i^t
                    
                    if b == 0:
                        final_al = node_attributes[t-1]
                    else:
                        final_al = round(FCM_class.sigmoid(x, lambda_value), 5) # A_i^t = sigmoid(x)
                    
                    G.nodes[node]['attr_dict']['value'][t] = final_al
                    
            graph_list[i] = G

        return graph_list
    

    # algorithm from kosko
    def kosko_alg_graph(graph_list, start_iter, end_iter, lambda_value):
        for i in range(len(graph_list)):
            G = graph_list[i]
            
            for t in range(start_iter,end_iter):    #for each iteration
                for node in G:  #for each node in the graph                        
                    b = 0
                    for edge in G.in_edges(node):   #for each incoming edge
                        other, _ = edge
                        w_edge = G[other][node]['weight']   # edge weight
                        other_attr = G.nodes[other]['attr_dict']['value']   # other node attributes
                        
                        b += w_edge * other_attr[t-1]   # B_i^t = sum(w_ij * A_j^(t-1))
                
                    final_al = round(FCM_class.sigmoid(b, lambda_value), 5)  # A_i^t = sigmoid(B_i^t)
                    
                    G.nodes[node]['attr_dict']['value'][t] = final_al
                    
            graph_list[i] = G      
        return graph_list
    

    # algorithm from Stylios and Groumpos
    def stylios_groumpos_alg_graph(graph_list, start_iter, end_iter, lambda_value):
        for i in range(len(graph_list)):
            G = graph_list[i]
            
            for t in range(start_iter,end_iter):    #for each iteration
                for node in G:  #for each node in the graph
                    node_attributes = G.nodes[node]['attr_dict']['value']
                        
                    b = 0
                    for edge in G.in_edges(node):   #for each incoming edge
                        other, _ = edge
                        w_edge = G[other][node]['weight']
                        other_attr = G.nodes[other]['attr_dict']['value']
                        
                        b += w_edge * other_attr[t-1]

                    al_node = node_attributes[t-1]

                    b = b + al_node
                
                    final_al = round(FCM_class.sigmoid(b, lambda_value), 5)
                    
                    G.nodes[node]['attr_dict']['value'][t] = final_al
                    
            graph_list[i] = G      
        return graph_list


    # algorithm from TODO
    def pi_fcm_alg_graph(graph_list, start_iter, end_iter, lambda_value):
        
        for i in range(len(graph_list)):
            G = graph_list[i]
            
            for t in range(start_iter, end_iter):   #for each iteration
                for node in G:  #for each node in the graph
                    node_attributes = G.nodes[node]['attr_dict']['value']

                    # contributo degli archi entranti  
                    b = 0   # B_i^t
                    for edge in G.in_edges(node):   #for each incoming edge
                        other, _ = edge
                        w_edge = G[other][node]['weight']
                        
                        al_new = node_attributes[t-1]
                    
                        b += w_edge * (2 * al_new - 1)
                    
                    al_node = node_attributes[t-1]
                    x = b + (2 * al_node - 1)
                    
                    if b == 0:
                        final_al = node_attributes[t-1]
                    else:
                        final_al = round(FCM_class.sigmoid(x, lambda_value), 5)
                    
                    G.node[node]['attr_dict']['value'][t] = final_al
                    
            graph_list[i] = G
        
        return graph_list 
    

    # algorithm from TODO
    def fcm_alg_graph(graph_list, start_iter, end_iter, lambda_value):
        for i in range(len(graph_list)):
            G = graph_list[i]
            
            for t in range(start_iter, end_iter):   #for each iteration
                for node in G:  #for each node in the graph
                    node_attributes = G.nodes[node]['attr_dict']['value']

                    # contributo degli archi entranti
                    b = 0
                    for edge in G.in_edges(node):   #for each incoming edge
                        other, _ = edge
                        w_edge = G[other][node]['weight']
                        if t == 1:
                            al_old = 0
                        else:
                            al_old = node_attributes[t-2]
                        al_new = node_attributes[t-1]
                        d_al = al_new - al_old
                        b += (w_edge * d_al)
                        
                    al_node = node_attributes[t-1]
                    if al_node != 0:
                        c = (np.log((1-al_node)/al_node))/-lambda_value
                    else:
                        c = 0
                    
                    x = (1 * b + 0.5 * c) / 1.5
                    
                    if b == 0: 
                        final_al = node_attributes[t-1]
                    else:
                        final_al = round(FCM_class.sigmoid(x, lambda_value),5)
                        
                    G.nodes[node]['attr_dict']['value'][t] = final_al
            
            graph_list[i] = G      
        
        return graph_list
    

    # algorithm from TODO
    def acfcm_alg_graph(graph_list, start_iter, end_iter, lambda_value):

        for i in range(len(graph_list)):
            G = graph_list[i]
            
            for t in range(start_iter,end_iter):    #for each iteration
                for node in G:  #for each node in the graph
                    node_attributes = G.nodes[node]['attr_dict']['value']
                        
                    b = 0
                    for edge in G.in_edges(node):   #for each incoming edge
                        other, _ = edge
                        w_edge = G[other][node]['weight']

                        if t == 1:
                            al_old = 0
                        else:
                            al_old = node_attributes[t-2]
                        
                        al_new = node_attributes[t-1]
                    
                        d_al = al_new #- al_old
                        
                        b += w_edge * d_al
                    
                    al_node = node_attributes[t-1]
                
                    Si = round(FCM_class.sigmoid(b, lambda_value), 5)
                
                    final_al = (Si + al_node) / 2
                    
                    G.nodes[node]['attr_dict']['value'][t] = final_al
                    
            graph_list[i] = G      
        return graph_list
