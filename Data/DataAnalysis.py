import os
import pickle
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

def GenerateDir(dir):
    if not os.path.exists(dir):
        os.makedirs(dir)

#PATHS
if 1:
    project_dir = r'C:\Users\xamuc\Desktop\PIC1\Data'

    Annotations_dir = project_dir + r'\Annotations'
    STRING_dir = project_dir + r'\STRING'
    GTEx_dir = project_dir + r'\GTEx'

    Annotations_Analysis_dir = project_dir + r'\Annotations\Analysis'
    STRING_Analysis_dir = project_dir + r'\STRING\Analysis'
    GTEx_Analysis_dir = project_dir + r'\GTEx\Analysis'

    for dir in [Annotations_Analysis_dir, STRING_Analysis_dir, GTEx_Analysis_dir]: GenerateDir(dir)

    #Imported Data
    if 1:
        annotations_path = Annotations_dir + r'\Homo_sapiens.GRCh38.111.gtf'

        protein_links_path = STRING_dir + r'\9606.protein.links.v12.0.txt'

        wholeblood_gene_tpm_path = GTEx_dir + r'\gene_tpm_whole_blood.gct'
        frontalcortex_gene_tpm_path = GTEx_dir + r'\gene_tpm_brain_frontal_cortex_ba9.gct'

        #------------
        STRING_protein_set_path = STRING_dir + r'\STRING_protein_set.pkl'
        STRING_link_weights_path = STRING_dir + r'\STRING_link_weights.pkl'
        STRING_protein_to_gene_path = STRING_dir + r'\STRING_protein_to_gene.pkl'
        STRING_gene_set_path = STRING_dir + r'\STRING_gene_set.pkl'
        STRING_gene_links_path = STRING_dir + r'\STRING_gene_links.txt'
        STRING_graph_links_path = STRING_dir + r'\STRING_graph_links.txt'
        dgl_graph_path = STRING_dir + r'\dgl_graph.bin'

        GTEx_fc_important_genes_path = GTEx_dir + r'\GTEx_fc_important_genes.pkl'
        GTEx_gene_set_path = GTEx_dir + r'\GTEx_gene_set.pkl'
        GTEx_common_donors_path = GTEx_dir + r'\GTEx_common_donors.pkl'
        GTEx_gene_intersection_set_path = GTEx_dir + r'\GTEx_gene_intersection_set.pkl'
        GTEx_filtered_gene_set_path = GTEx_dir + r'\GTEx_filtered_gene_set.pkl'
        GTEx_wbdf_filtered_path = GTEx_dir + r'\GTEx_wbdf_filtered.csv'
        GTEx_fcdf_filtered_path = GTEx_dir + r'\GTEx_fcdf_filtered.csv'
        GTEx_wbg_df_path = GTEx_dir + r'\GTEx_wbg_df.csv'

#STRING Analysis
if 0:
    print("STRING Analysis")

    with open(STRING_protein_set_path, 'rb') as f1, open(STRING_link_weights_path, 'rb') as f2, open(STRING_protein_to_gene_path, 'rb') as f3, open(STRING_gene_set_path, 'rb') as f4:
        STRING_protein_set = pickle.load(f1)
        STRING_link_weights = pickle.load(f2)
        STRING_protein_to_gene = pickle.load(f3)
        STRING_gene_set = pickle.load(f4)

    print(f'STRING protein set lenght: {len(STRING_protein_set)}')

   #STRING protein link weights
    if 1:
        print(f'STRING protein link weights:')
        max_weight = max(STRING_link_weights)
        min_weight = min(STRING_link_weights)
        print(f'Max Weight: {max_weight}\nMin Weight: {min_weight}')
        #Histogram
        if 0:
            print("Distribution")
            bin_edges = np.arange(0, 1100, 100)  # bins from 0 to 1000 with width of 100
            plt.hist(STRING_link_weights, bins=bin_edges, edgecolor='black')
            title = 'STRING link weights histogram'
            plt.title(title)
            plt.xlabel('Weight')
            plt.xticks(np.arange(0, max_weight + 100, 100))
            plt.ylabel('Frequency')
            plt.savefig(STRING_Analysis_dir + rf'\{title}.png')
            plt.show()

    #STRING protein to gene
    if 1:
        print("STRING protein to gene")
        genes_per_protein = {}
        for protein in STRING_protein_set:
            n_genes = str(len(STRING_protein_to_gene[protein]))
            if genes_per_protein.get(n_genes):
                genes_per_protein[n_genes] += 1
            else:
                genes_per_protein[n_genes] = 1

        for n_genes in list(genes_per_protein):
            print(f'{genes_per_protein[n_genes]} proteins have {n_genes} corresponding genes')

print()
#GTEx Analysis
if 1:
    print("GTEx Analysis")
    with open(GTEx_fc_important_genes_path, 'rb') as f1, open( GTEx_gene_set_path, 'rb') as f2, open(GTEx_common_donors_path, 'rb') as f3, open(GTEx_gene_intersection_set_path, 'rb') as f4, open(GTEx_filtered_gene_set_path, 'rb') as f5:
        GTEx_fc_important_genes = pickle.load(f1)
        GTEx_gene_set = pickle.load(f2)
        GTEx_common_donors = pickle.load(f3)
        GTEx_gene_intersection_set = pickle.load(f4)
        GTEx_filtered_gene_set = pickle.load(f5)

        wbdf = pd.DataFrame(pd.read_csv(wholeblood_gene_tpm_path, sep='\t', skiprows=2))
        fcdf = pd.DataFrame(pd.read_csv(frontalcortex_gene_tpm_path, sep='\t', skiprows=2))
        #wbdf_filtered = pd.DataFrame(pd.read_csv(GTEx_wbdf_filtered_path, sep=','))
        #fcdf_filtered = pd.DataFrame(pd.read_csv(GTEx_fcdf_filtered_path, sep=','))
        #GTEx_wbg_df = pd.DataFrame(pd.read_csv(GTEx_wbg_path, sep=','))

    print(f'GTEx gene tpm data frames:')
    #First three columns: id, Name, Description"
    print(f'Wholeblood')
    print(wbdf.shape)
    print('Frontal Cortex')
    print(fcdf.shape)

    #Gene tpm row sparsity histogram
    if 1:
        print("Gene tpm row sparsity")
        dfs = [wbdf, fcdf]
        for i in range(2):
            df = dfs[i]
            is_missing = df.isna() | (df == 0)
            row_sparsity = is_missing.sum(axis=1) / df.shape[1]
            bins = np.arange(0, 1.1, 0.1)
            counts, bins = np.histogram(row_sparsity, bins=bins)
            counts = counts / sum(counts)
            plt.bar(bins[:-1], counts, width=0.1, edgecolor='black', align='edge')
            title = f'{l0[i]} gene tpm row sparsity'
            plt.title(title)
            plt.xlabel('Sparsity')
            plt.xticks(bins)
            plt.ylabel('Frequency')
            plt.grid(True, alpha=0.5)
            plt.savefig(GTEx_Analysis_dir + rf'\{title}.png')
            plt.show()

    print(f'Common donors between Data frames: {len(GTEx_common_donors)}\n')

    l0 = ['Whole blood', 'Frontal Cortex']
    l1 = ['wb', 'fc']
    #Original gene tpm, gene sets
    for i in range(2):
        print(f'GTEx {l0[i]} gene tpm, gene set lenght: {len(GTEx_gene_set[l1[i]])}')
    print()
    # Genes repetidos ? _PAR_Y?

    print(f'GTEx and STRING Wholeblood gene intersection set lenght: {len(GTEx_gene_intersection_set["wb"])}')
    # 19039 genes do STRING interseção 56156 GTEx -> 18607 - Genes não partilhados em ambos os conjuntos
    print(f'GTEx Wholeblood filtered gene set lenght: {len(GTEx_filtered_gene_set["wb"])}')
    print()
    print(f'GTEx frontal cortex important genes lenght: {len(GTEx_fc_important_genes)}')
    print(f'GTEx and Frontal cortex important genes, gene intersection set lenght: {len(GTEx_gene_intersection_set["fc"])}')
    print(f'GTEx Frontal Cortex filtered gene set lenght: {len(GTEx_filtered_gene_set["fc"])}')


