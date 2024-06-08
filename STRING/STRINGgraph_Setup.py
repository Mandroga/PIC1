import dgl.data
import torch
import networkx as nx
import matplotlib.pyplot as plt
import json
import numpy as np
import time

# FUNCTIONS --------------------------
def PlotDGLGraph(graph):
    print("Plotting DGLGraph")
    nx_graph = graph.to_networkx()
    has_labels = True
    has_colors = True
    has_edgedata = True
    # Extract node labels (features)
    try: node_labels = {node: f"{node} - {graph.ndata['feat'][node].tolist()}" for node in nx_graph.nodes()}
    except: has_labels = False
    try: node_colors = ['red' if graph.ndata['label'][node].item()==1 else 'blue' for node in nx_graph.nodes]
    except: has_colors = False
    # Extract edge labels (values)
    try:
        edge_labels = {}
        for src, dst in nx_graph.edges():
            edge_ids = graph.edge_ids(src, dst)
            edge_labels[(src, dst)] = graph.edata['w'][edge_ids].item()
    except: has_edgedata = False
    # Plot the graph
    pos = nx.spring_layout(nx_graph)  # Position nodes using the spring layout algorithm
    if has_labels and has_colors and has_edgedata:
        nx.draw(nx_graph, pos, with_labels=True, labels=node_labels, node_color=node_colors, node_size=1500, font_size=12)
        nx.draw_networkx_edge_labels(nx_graph, pos, edge_labels=edge_labels, font_color='red')
    elif has_edgedata and has_labels:
        nx.draw(nx_graph, pos, with_labels=True, labels=node_labels,  node_size=1500,font_size=12)
        nx.draw_networkx_edge_labels(nx_graph, pos, edge_labels=edge_labels, font_color='red')
    else:
        nx.draw(nx_graph, pos, node_size=1500, font_size=12)
    plt.show()

last_percent = 0
t = 0
def EstimateTimePercent(i, maxiter, chunk):
    global t
    global last_percent
    if i % chunk == 0:
        percent = (i / maxiter) * 100

        if i != 0:
            diff = percent - last_percent
            interval = time.time() - t
            estimate_time = int((100 - percent) * interval / diff)
        else:
            estimate_time = -1

        t = time.time()
        last_percent = percent

        print(f"{percent:.3f}% - estimate time: {estimate_time}")

#PATHS
if 1:
    protein_links_path = r'C:\Users\xamuc\Desktop\PIC1\DataSetup\STRING\9606.protein.links.v12.0.txt'
    protein_set_path = r'C:\Users\xamuc\Desktop\PIC1\DataSetup\STRING\protein_set.json'

    protein_to_gene_path = r'C:\Users\xamuc\Desktop\PIC1\DataSetup\Annotations\protein_to_gene.json'

    String_gene_set_path = r'C:\Users\xamuc\Desktop\PIC1\DataSetup\STRING\Sring_gene_set.json'
    GTex_gene_set_path = r'C:\Users\xamuc\Desktop\PIC1\DataSetup\GTex_gene_set.json'
    GTexString_gene_set_path = r'C:\Users\xamuc\Desktop\PIC1\DataSetup\STRING\GTexString_gene_set.json'

    gene_links_path = r'C:\Users\xamuc\Desktop\PIC1\DataSetup\STRING\gene_links.txt'
    gene_links_graph_path = r'C:\Users\xamuc\Desktop\PIC1\DataSetup\STRING\gene_links_graph.dgl'

    familyndata_path = r'C:\Users\xamuc\Desktop\PIC1\DataSetup\STRING\familyndata.json'

#Protein set
if 0:
    protein_set = set()
    with open(protein_links_path, 'r') as f:
        i = 0
        for line in f:
            a = line.split(" ")
            p1 = a[0].split(".")[1]
            p2 = a[1].split(".")[1]
            protein_set.add(p1)
            protein_set.add(p2)
            i += 1
            EstimateTimePercent(i, 13715404, 10000)
    with open(protein_set_path, 'w') as f:
        json.dump({'protein_set':list(protein_set)}, f)

#Protein to Gene analysis and String Gene set || #proteins in graph: 19622, #genes codified: 19150, #different genes: 19139
if 0:
    with open(protein_set_path, 'r') as f:
        protein_set = json.load(f)['protein_set']
    with open(protein_to_gene_path, 'r') as f:
        protein_to_gene = json.load(f)

    String_gene_set = set()
    protein_list = list(protein_to_gene)
    gene_counts = []

    for protein in protein_list:
        String_gene_set.update(protein_to_gene[protein])
        gene_counts += [len(protein_to_gene[protein])]

    gene_counts = sorted(gene_counts)
    String_gene_set = list(String_gene_set)
    print(gene_counts)
    print(f"Number of proteins in graph: {len(protein_set)}, Number of genes codified: {sum(gene_counts)}, Number of genes: {len(String_gene_set)}")
    # Algumas proteinas não tem um gene respetivo, São todas codificadas por um gene e existem proteinas que são codificadas pelo mesmo gene

    with open(String_gene_set_path, 'w') as f:
        json.dump({'String_gene_set':String_gene_set}, f)

# -----

#generate GTex&String gene set
if 0:
    with open(GTex_gene_set_path, 'r') as f:
        GTex_gene_set = json.load(f)['GTex_gene_set']
    with open(String_gene_set_path, 'r') as f:
        String_gene_set = json.load(f)['String_gene_set']

    GTexString_gene_set = list(np.sort(np.array(list(set(String_gene_set) & set(GTex_gene_set)))))
    print(f"GTex gene set: {len(GTex_gene_set)} Sring gene set: {len(String_gene_set)} Gtex and String gene set: {len(GTexString_gene_set)}")

    with open(GTexString_gene_set_path, 'w') as f:
        json.dump({'GTexString_gene_set':GTexString_gene_set}, f)

# Gene links from GTex&String gene set
if 0:
    with open(GTexString_gene_set_path, 'r') as f:
        GTexString_gene_set = json.load(f)['GTexString_gene_set']

    with open(protein_set_path, 'r') as f:
        protein_set = json.load(f)['protein_set']

    with open(protein_to_gene_path, 'r') as f:
        protein_to_gene = json.load(f)

    pg2 = {}
    i = 0
    for protein in protein_set:
        gene = protein_to_gene[protein]
        if gene != [] and gene[0] in GTexString_gene_set:
            pg2[protein] = gene[0]
        EstimateTimePercent(i, len(protein_set),1000)
        i += 1

    print("Generating gene links")
    with open(gene_links_path, 'w') as of:
        of.write("")
    with open(protein_links_path, 'r') as f, open(gene_links_path, 'a') as of:
        bs = 125000
        bi = 0
        while True:
            try:
                lines = np.array([next(f) for _ in range(bs)])
                arrays = np.array([line.split(" ") for line in lines])
                arrays = np.array([(a[0].split(".")[1],a[1].split(".")[1],a[2]) for a in arrays])
                lines = np.array([f"{pg2[a[0]]} {pg2[a[1]]} {a[2]}" for a in arrays if pg2.get(a[0]) != None and pg2.get(a[1]) != None])
                of.writelines(lines)
                EstimateTimePercent(bi*bs, 13715404, bs)
                bi += 1
            except: break

# generate dgl graph
if 0:
    with open(GTexString_gene_set_path, 'r') as f:
        GTexString_gene_set = np.array(json.load(f)['GTexString_gene_set'])

    g = dgl.graph(([], []))
    g.edata['w'] = torch.zeros(g.number_of_edges())

    with open(gene_links_path, 'r') as f:
        bs = 125000
        bi = 0
        while True:
            try:
                src, dst, weights = zip(*[tuple(next(f).strip().split(" ")) for _ in range(bs)])
                start_time = time.time()
                src, dst, weights = np.searchsorted(GTexString_gene_set, src), np.searchsorted(GTexString_gene_set, dst), np.array(weights, dtype=np.float32)
                dst_test = [GTexString_gene_set[v] for v in dst]
                weights = torch.tensor(weights, dtype=torch.float32)

                g.add_edges(src, dst)
                g.add_edges(dst, src)

                g.edata['w'][g.edge_ids(src, dst)] = weights
                g.edata['w'][g.edge_ids(dst, src)] = weights

                EstimateTimePercent(bi*bs, 12966505,bs)
                bi += 1
            except: break

        print("Finished reading...")
        print(g.edata['w'])
        dgl.save_graphs(gene_links_graph_path, [g])
        print("Graph Saved")

#Graph analysis

if 0:
    print("loading data...")
    g = dgl.load_graphs(gene_links_graph_path)[0][0]

    #plot hist link strenght frequencies
    if 0:
        data = np.array(g.edata['w'].tolist())
        data_set = list(set(data))
        max_link = max(data_set) # 999
        min_link = min(data_set) # 0
        nbins = 10
        plt.hist(data, bins=nbins, edgecolor='black')
        plt.title('Link strenght frequencies')
        plt.xlabel('Link Strenght')
        plt.ylabel('Frequency')
        plt.savefig(r'C:\Users\xamuc\Desktop\PIC1\DataSetup\STRING\Linkstrenghtfrequencies.png')
        plt.show()

    #Generate families
    w_threshold = 500
    if 0:
        print("Searching for families...")
        g.ndata['f'] = torch.full((g.num_nodes(),), -1)
        nf = 0

        bs = 10000
        bi = 0
        nb = len(g.edata['w'])//bs
        print(f"Reading batches - {nb}")
        for bi in range(nb):
            I = np.arange(bi*bs, (bi+1*bs))
            #I, W = zip(*[(i,w) for i,w in zip(I, g.edata['w'][I]) if w > 150])
            I = np.array([i for i, w in zip(I, g.edata['w'][I]) if w > w_threshold])
            for i in I:
                src = g.edges()[0][i]
                fsrc = g.ndata['f'][src]
                dst = g.edges()[1][i]
                fdst = g.ndata['f'][dst]
                if fsrc == -1 and fdst == -1:
                    g.ndata['f'][src] = nf
                    g.ndata['f'][dst] = nf
                    nf += 1
                elif fsrc == -1 and fdst != -1:
                    g.ndata['f'][src] = fdst
                elif fsrc != -1 and fdst == -1:
                    g.ndata['f'][dst] = fsrc
                else:
                    g.ndata['f'][g.ndata['f'] == fdst] = fsrc

            EstimateTimePercent(bi, nb, 1)
            bi += 1

        familyndata = g.ndata['f'].tolist()
        family_set = sorted(list(set(familyndata)))[1:]
        for i in range(len(family_set)):
            print(i, family_set[i])
            g.ndata['f'][g.ndata['f'] == family_set[i]] = i
        familyndata = g.ndata['f'].tolist()


        with open(familyndata_path,'w') as f:
            json.dump({'familyndata':familyndata}, f)

    #Family analysis
    if 1:
        with open(familyndata_path, 'r') as f:
            familyndata = np.array(json.load(f)['familyndata'])

        family_set = set(familyndata)
        familyndata = familyndata.astype(np.float32) + 0.1
        print(family_set)
        #family hist
        print(sorted(familyndata))
        if 1:
            bin_edges = list(range(-1, max(family_set)+2))
            plt.hist(familyndata, bins=bin_edges, edgecolor='black', log=True)
            plt.title('Family counts')
            #plt.xticks(list(range(-1,max(family_set)+1)))
            plt.xlabel('Family')
            plt.ylabel('Frequency')
            plt.savefig(fr'C:\Users\xamuc\Desktop\PIC1\DataSetup\STRING\familycounts{w_threshold}.png')
            plt.show()
        #g.ndata['f'] = torch.tensor(familyndata)

