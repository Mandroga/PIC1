import os
import pickle
import numpy as np
import time
import pandas as pd
import dgl
import torch

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
def GenerateDir(dir):
    if not os.path.exists(dir):
        os.makedirs(dir)

def savefile(file, file_path):
    with open(file_path, 'wb') as f:
        pickle.dump(file, f)

def loadfile(file_path):
    with open(file_path, 'rb') as f:
        return pickle.load(f)
    
def DefineFindFunctions(search_for, search_find):
    search_condition = search_for + search_find

    if 'gene_name' in search_condition:
        gene_name_find = lambda lines: np.char.find(lines, "gene_name") + 11
    else:
        gene_name_find = lambda lines: np.array([10] * len(lines))

    if 'gene_id' in search_condition:
        gene_id_find = lambda lines: np.char.find(lines, "gene_id") + 9
    else:
        gene_id_find = lambda lines: np.array([8] * len(lines))

    if 'protein_id' in search_condition:
        protein_id_find = lambda lines: np.char.find(lines, "protein_id") + 12
    else:
        protein_id_find = lambda lines: np.array([11] * len(lines))

    return gene_name_find, gene_id_find, protein_id_find
def DefineCondition(search_for, find_list):
    found_gene_name = lambda line, gnidx, giidx, pidx: (gnidx != 10 and line[gnidx:gnidx + 4] in find_list)
    found_gene_id = lambda line, gnidx, giidx, pidx: (giidx != 8 and line[giidx:giidx + 15] in find_list)
    found_protein_id = lambda line, gnidx, giidx, pidx: (pidx != 11 and line[pidx:pidx + 15] in find_list)

    # Create a list of conditions based on search_for
    conditions = []
    if 'gene_name' in search_for:
        conditions.append(found_gene_name)
    if 'gene_id' in search_for:
        conditions.append(found_gene_id)
    if 'protein_id' in search_for:
        conditions.append(found_protein_id)

    # Define the combined condition function
    condition = lambda line, gnidx, giidx, pidx: any(cond(line, gnidx, giidx, pidx) for cond in conditions)
    return condition

def AnnotationSearch(find_list, search_for, search_find):
    condition = DefineCondition(search_for, find_list)
    gene_name_find, gene_id_find, protein_id_find = DefineFindFunctions(search_for, search_find)

    found_list = set()
    with open(annotations_path, 'r') as f:
        bs = 50000
        bi = 0
        for _ in range(5): next(f)  # skip metadata
        while True:
           try:
                lines = np.array([next(f) for _ in range(bs)])

                gene_name_indexes = gene_name_find(lines)
                gene_id_indexes = gene_id_find(lines)
                protein_id_indexes = protein_id_find(lines)

                found_list.update({(line[gnidx:gnidx + 4], line[giidx:giidx + 15], line[pidx:pidx + 15])
                                    for line,  gnidx, giidx, pidx in zip(lines, gene_name_indexes, gene_id_indexes,  protein_id_indexes)
                                    if condition(line, gnidx, giidx, pidx)})

                EstimateTimePercent(bi * bs, 3420000, bs)
                bi += 1
           except:
                break
        found_list = list(found_list)
        return found_list

#PATHS
if 1:
    project_dir = r'C:\Users\xamuc\Desktop\PIC1\Data'
    input_tissue = 'Whole blood'
    output_tissue = 'Amygdala'

    Annotations_dir = project_dir + r'\Annotations'
    STRING_dir = project_dir + rf'\STRING'
    GTEx_dir = project_dir + rf'\GTEx'
    
    STRING_setup_dir = STRING_dir + rf'\{input_tissue} - {output_tissue}'
    GTEx_setup_dir = GTEx_dir + rf'\{input_tissue} - {output_tissue}'
    
    for dir in [Annotations_dir, STRING_setup_dir, GTEx_setup_dir]: GenerateDir(dir)

    annotations_path = Annotations_dir + r'\Homo_sapiens.GRCh38.111.gtf'

    protein_links_path = STRING_dir + r'\9606.protein.links.v12.0.txt'

    input_gene_tpm_path = GTEx_dir + r'\gene_tpm_whole_blood.gct'
    target_gene_tpm_path = GTEx_dir + r'\gene_tpm_brain_amygdala.gct'

    #------------
    STRING_protein_set_path = STRING_setup_dir + r'\STRING_protein_set.pkl'
    STRING_link_weights_path = STRING_setup_dir + r'\STRING_link_weights.pkl'
    STRING_protein_to_gene_path = STRING_setup_dir + r'\STRING_protein_to_gene.pkl'
    STRING_gene_set_path = STRING_setup_dir + r'\STRING_gene_set.pkl'
    STRING_gene_links_path = STRING_setup_dir + r'\STRING_gene_links.txt'
    STRING_graph_links_path = STRING_setup_dir + r'\STRING_graph_links.txt'
    dgl_graph_path = STRING_setup_dir + r'\dgl_graph.bin'

    GTEx_important_genes_path = GTEx_setup_dir + r'\GTEx_important_genes.pkl'
    GTEx_gene_set_path = GTEx_setup_dir + r'\GTEx_gene_set.pkl'
    GTEx_common_donors_path = GTEx_setup_dir + r'\GTEx_common_donors.pkl'
    GTEx_gene_intersection_set_path = GTEx_setup_dir + r'\GTEx_gene_intersection_set.pkl'
    GTEx_filtered_gene_set_path = GTEx_setup_dir + r'\GTEx_filtered_gene_set.pkl'
    GTEx_input_df_filtered_path = GTEx_setup_dir + r'\GTEx_input_df_filtered.csv'
    GTEx_target_df_filtered_path = GTEx_setup_dir + r'\GTEx_target_df_filtered.csv'
    GTEx_input_graph_df_path = GTEx_setup_dir + r'\GTEx_input_graph_df.csv'

#STRING Protein set (Load time - 60s)
if 0:
    print("STRING Protein set")
    STRING_protein_set = set()
    with open(protein_links_path, 'r') as f:
        bs = 500000
        bi = 0
        while True:
            try:
                lines = np.array([next(f) for _ in range(bs)])
                arrays = np.array([line.split(" ") for line in lines])
                STRING_protein_set.update({protein for a in arrays for protein in (a[0].split(".")[1], a[1].split(".")[1])})
                EstimateTimePercent(bi*bs, 13715404, bs)
                bi += 1
            except: break

    STRING_protein_set = list(STRING_protein_set)
    savefile(STRING_protein_set, STRING_protein_set_path)


#Load STRING Protein set
if 1:
    STRING_protein_set = loadfile(STRING_protein_set_path)

#ANNOTATIONS -----------------
#Generate STRING Protein to Gene (Needs STRING Protein set) (load time - 320s)
if 0:
    print("Generate STRING Protein to Gene")
    found_proteins = AnnotationSearch(STRING_protein_set, ['protein_id',], ['gene_id',])
    STRING_protein_to_gene = {}
    for protein in STRING_protein_set:
        STRING_protein_to_gene[protein] = set()

    for gene_name, gene_id, protein_id in found_proteins:
        STRING_protein_to_gene[protein_id].add(gene_id)

    for protein in STRING_protein_set:
        STRING_protein_to_gene[protein] = list(STRING_protein_to_gene[protein])
    
    savefile(STRING_protein_to_gene, STRING_protein_to_gene_path)

#Find important genes (load time - 38s)
if 0:
    print("Find important genes")
    find_genes = ["DRD1", "DRD2", "DRD3", "DRD4", "DRD5","SLC6A4","BDNF","COMT","DRD2","GRM3","DISC1","MAOA","NRG1","CACNA1C","FKBP5"]
    find_genes += ["GABRA1","GABRA2","GABRA3","GABRA4","GABRA5","GABRA6","GABRB1","GABRB2","GABRB3","GABRD","GABRE","GABRG1","GABRG2","GABRG3","GABRP","GABRQ","GABRR1","GABRR2","GABRR3","GABBR1","GABBR2"]

    found_list = AnnotationSearch(find_genes, ['gene_name',], ['gene_id',])

    GTEx_important_genes = set()
    for gene_name, gene_id, protein_id in found_list:
        print(gene_name, gene_id)
        GTEx_important_genes.add(gene_id)

    GTEx_important_genes = list(GTEx_important_genes)
    print(len(GTEx_important_genes))
    savefile(GTEx_important_genes, GTEx_important_genes_path)

#-----------------------------
#STRING Gene set (Needs STRING Protein set and Protein to Gene)
if 1:
    print("STRING Gene set")
    
    STRING_protein_set = loadfile(STRING_protein_set_path)
    STRING_protein_to_gene = loadfile(STRING_protein_to_gene_path)

    STRING_gene_set = set()
    for protein in STRING_protein_set:
        STRING_gene_set.update(STRING_protein_to_gene[protein])
    STRING_gene_set = list(STRING_gene_set)
    savefile(STRING_gene_set, STRING_gene_set_path)
    
#GTEx ------------------------
#Filter Data
if 0:
    #Load GTEx Data (load time 8s)
    if 1:
        print("Load GTEx Data")
        input_df = pd.DataFrame(pd.read_csv(input_gene_tpm_path, sep='\t', skiprows=2))
        target_df = pd.DataFrame(pd.read_csv(target_gene_tpm_path, sep='\t', skiprows=2))
    #GTEx Gene set
    if 1:
        print("GTEx Gene set")
        input_gene_set = list(set(input_df['Name'].apply(lambda x: x.split('.')[0])))
        target_gene_set = list(set(target_df['Name'].apply(lambda x: x.split('.')[0])))

        GTEx_gene_set = {'input':input_gene_set,'target':target_gene_set}
        savefile(GTEx_gene_set, GTEx_gene_set_path)

    # GTex common donors
    if 1:
        print("GTex common donors")
        input_donors = [donor[:10] for donor in input_df.columns.tolist()[3:]]
        target_donors = [donor[:10] for donor in target_df.columns.tolist()[3:]]
        GTEx_common_donors = list(set(input_donors) & set(target_donors))
        savefile(GTEx_common_donors, GTEx_common_donors_path)
    #GTex Filter Data (Needs GTEx Data, GTEx and STRING Gene sets, GTEx target important genes, common donors) (Load time 10s)
    if 1:
        print("GTex Filter Data")
        
        GTEx_important_genes = loadfile(GTEx_important_genes_path)

        GTEx_gene_intersection_set = {'input':tuple(set(GTEx_gene_set['input']) & set(STRING_gene_set)),
                                      'target':tuple(set(GTEx_gene_set['target']) & set(GTEx_important_genes))}

        savefile(GTEx_gene_intersection_set, GTEx_gene_intersection_set_path)


        data_name = ['input', 'target']
        GTEx_filtered_gene_set = {'input':(),'target':()}
        dfs = [input_df, target_df]
        df_save_path = [GTEx_input_df_filtered_path, GTEx_target_df_filtered_path]
        for i in range(2):
            df = dfs[i]
            gene_set = GTEx_gene_intersection_set[data_name[i]]

            # Filter genes
            if 1:
                base_gene_ids = df['Name'].apply(lambda x: x.split('.')[0])
                mask = base_gene_ids.isin(gene_set)
                df = df[mask]

            #Filter _PAR_Y
            if 1:
                mask = ~df['Name'].str.contains('_PAR_Y')
                df = df[mask]

            #Filter versions out of name
            if 1: df['Name'] = base_gene_ids
            #common donors
            if 1:
                donor_ids = df.columns[3:].str[:10]
                mask = donor_ids.isin(GTEx_common_donors)
                df = df[df.columns[:3].append(df.columns[3:][mask])]

            #sparsity
            if 1:
                is_missing = df.isna() | (df == 0)
                row_sparsity = is_missing.sum(axis=1) / df.shape[1]
                df = df[row_sparsity<= 0.5]

            # Normalize by average
            if 1:
                flattened_data = df.iloc[:, 3:].values.flatten()
                #quantiles = list(range(10,100,5)) # Example list of quantiles
                #quantile_values = np.percentile(flattened_data, quantiles)
                #print(quantile_values)
                quantile95 = np.percentile(flattened_data, 95)
                #df.iloc[:, 3:] = df.iloc[:, 3:].apply(lambda row: row / row.mean(), axis=1)
                df.iloc[:, 3:] = df.iloc[:, 3:]/quantile95

            #Filtered gene set
            if 1:
                GTEx_filtered_gene_set[data_name[i]] = np.sort(df['Name'].to_numpy())
            # Save df
            df.to_csv(df_save_path[i], index=False)
        
        savefile(GTEx_filtered_gene_set, GTEx_filtered_gene_set_path)
#Filtered Data
if 1:
    #Load GTEx Filtered Data
    if 1:
        print("Load GTEx Filtered Data")
        input_df = pd.DataFrame(pd.read_csv(GTEx_input_df_filtered_path, sep=','))
        target_df = pd.DataFrame(pd.read_csv(GTEx_target_df_filtered_path, sep=','))

        STRING_protein_set = loadfile(STRING_protein_set_path)
        input_filtered_gene_set = loadfile(GTEx_filtered_gene_set_path)['input']
        GTEx_common_donors = loadfile(GTEx_common_donors_path)

    # Graph Dataframe (Needs GTEx Data Filtered, GTEx common donors, input filtered gene set)
    if 1:
        input_graph_df_rows = len(input_filtered_gene_set)
        input_graph_df = pd.DataFrame(columns=input_df.columns, index=range(input_graph_df_rows))

        # Sort genes for index to correspond to graph node
        gene_to_index = {gene:np.where(input_filtered_gene_set==gene)[0][0] for gene in input_filtered_gene_set}
        gene_list = [gene.split(".")[0] for gene in input_df['Name'].tolist()]

        bs = 1000
        n_batches = len(gene_list) // bs + 1
        for bi in range(n_batches):
            I = np.arange(bi * bs, min((bi + 1) * bs, len(gene_list)))
            indexes = [gene_to_index.get(gene_list[i]) for i in I]
            filtered_I_indexes = [(i, index) for i, index in zip(I, indexes) if index is not None]
            if filtered_I_indexes != []:
                I, indexes = zip(*filtered_I_indexes)
                input_graph_df.iloc[list(indexes)] = input_df.iloc[list(I)]
        input_graph_df.to_csv(GTEx_input_graph_df_path, index=False)

# STRING -------------------
if 1:
    # Gene links (Needs STRING protein set STRING protein to gene input filtered gene set) (Load time 128s)
    if 0:
        pg = {}
        i = 0
        for protein in STRING_protein_set:
            gene = STRING_protein_to_gene[protein]
            if gene != [] and gene[0] in input_filtered_gene_set:
                pg[protein] = gene[0]


        print("Generating gene links")
        with open(STRING_gene_links_path, 'w') as of:
            of.write('')
        with open(protein_links_path, 'r') as f, open(STRING_gene_links_path, 'a') as of:
            bs = 150000
            bi = 0
            while True:
                try:
                    lines = np.array([next(f) for _ in range(bs)])
                    arrays = np.array([line.split(" ") for line in lines])
                    arrays = np.array([(a[0].split(".")[1],a[1].split(".")[1],a[2]) for a in arrays])

                    gene_links_lines = np.array([f"{pg[a[0]]} {pg[a[1]]} {a[2]}" for a in arrays if pg.get(a[0]) != None and pg.get(a[1]) != None])

                    of.writelines(gene_links_lines)

                    EstimateTimePercent(bi*bs, 13715404, bs)
                    bi += 1
                except: break

    #Gene link weights
    if 0:
        print("Gene link weights")
        STRING_link_weights = []
        with open(STRING_gene_links_path, 'r') as f:
            bs = 500000
            bi = 0
            while True:
                try:
                    lines = np.array([next(f) for _ in range(bs)])
                    arrays = np.array([line.split(" ") for line in lines])
                    STRING_link_weights += [float(a[2]) for a in arrays]
                    EstimateTimePercent(bi * bs, 13715404, bs)
                    bi += 1
                except:
                    break
        
        savefile(STRING_link_weights, STRING_link_weights_path)

    # Generate dgl graph from Gene Links (Load time 80s)
    if 1:
        print("Generate dgl graph from Gene Links")
        g = dgl.graph(([], []))
        g.edata['w'] = torch.zeros(g.number_of_edges())

        with open(STRING_gene_links_path, 'r') as f:
            bs = 125000
            bi = 0
            while True:
                try:
                    src, dst, weights = zip(*[tuple(next(f).strip().split(" ")) for _ in range(bs)])
                    start_time = time.time()
                    src, dst, weights = np.searchsorted(input_filtered_gene_set, src), np.searchsorted(input_filtered_gene_set, dst), np.array(weights, dtype=np.float32)
                    weights = torch.tensor(weights, dtype=torch.float32)

                    g.add_edges(src, dst)
                    g.add_edges(dst, src)

                    g.edata['w'][g.edge_ids(src, dst)] = weights
                    g.edata['w'][g.edge_ids(dst, src)] = weights

                    EstimateTimePercent(bi*bs, 12966505,bs)
                    bi += 1
                except: break

            print("Finished reading...")
            dgl.save_graphs(dgl_graph_path, [g])
            print("Graph Saved")

