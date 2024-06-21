import os
import pickle
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

def GenerateDir(dir):
    if not os.path.exists(dir):
        os.makedirs(dir)

def savefile(file, file_path):
    with open(file_path, 'wb') as f:
        pickle.dump(file, f)

def loadfile(file_path):
    with open(file_path, 'rb') as f:
        return pickle.load(f)
#PATHS
if 1:
    project_dir = r'C:\Users\xamuc\Desktop\PIC1\Data'
    input_tissue = 'Whole blood'
    target_tissue = 'Amygdala'
    tissues = [input_tissue, target_tissue]

    Annotations_dir = project_dir + rf'\Annotations'
    STRING_dir = project_dir + r'\STRING'
    GTEx_dir = project_dir + r'\GTEx'

    STRING_setup_dir = STRING_dir + rf'\{input_tissue} - {target_tissue}'
    GTEx_setup_dir = GTEx_dir + rf'\{input_tissue} - {target_tissue}'

    Annotations_Analysis_dir = project_dir + rf'\Annotations\Analysis\{input_tissue} - {target_tissue}'
    STRING_Analysis_dir = project_dir + rf'\STRING\Analysis\{input_tissue} - {target_tissue}'
    GTEx_Analysis_dir = project_dir + rf'\GTEx\Analysis\{input_tissue} - {target_tissue}'

    for dir in [Annotations_Analysis_dir, STRING_Analysis_dir, GTEx_Analysis_dir]: GenerateDir(dir)

    #Imported Data
    if 1:
        annotations_path = Annotations_dir + r'\Homo_sapiens.GRCh38.111.gtf'

        protein_links_path = STRING_dir + r'\9606.protein.links.v12.0.txt'

        input_gene_tpm_path = GTEx_dir + r'\gene_tpm_whole_blood.gct'
        target_gene_tpm_path = GTEx_dir + r'\gene_tpm_brain_frontal_cortex_ba9.gct'

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

#STRING Analysis
if 1:
    print("STRING Analysis")

    STRING_protein_set = loadfile(STRING_protein_set_path)
    STRING_link_weights = loadfile(STRING_link_weights_path)
    STRING_protein_to_gene = loadfile(STRING_protein_to_gene_path)
    STRING_gene_set = loadfile(STRING_gene_set_path)

    print(f'STRING protein set lenght: {len(STRING_protein_set)}')

   #STRING protein link weights
    if 1:
        print(f'STRING protein link weights:')
        max_weight = max(STRING_link_weights)
        min_weight = min(STRING_link_weights)
        print(f'Max Weight: {max_weight}\nMin Weight: {min_weight}')
        #Histogram
        if 1:
            print("Distribution")
            bins = np.arange(0, 1100, 100)
            counts, _ = np.histogram(STRING_link_weights, bins=bins)
            counts = counts / sum(counts)
            plt.bar(bins[:-1], counts, edgecolor='black', align='edge', width=100)

            #bin_edges = np.arange(0, 1100, 100)  # bins from 0 to 1000 with width of 100
           # plt.hist(STRING_link_weights, bins=bin_edges, edgecolor='black')
            title = 'Gene link weights histogram'
            plt.title(title)
            plt.xlabel('Weight')
            plt.xticks(np.arange(0, max_weight + 100, 100))
            plt.ylabel('Relative Frequency')
            plt.grid(True, alpha=0.5)
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
    GTEx_target_important_genes = loadfile(GTEx_important_genes_path)
    GTEx_gene_set = loadfile(GTEx_gene_set_path)
    GTEx_common_donors = loadfile(GTEx_common_donors_path)
    GTEx_gene_intersection_set = loadfile(GTEx_gene_intersection_set_path)
    GTEx_filtered_gene_set = loadfile(GTEx_filtered_gene_set_path)

    input_df = pd.DataFrame(pd.read_csv(input_gene_tpm_path, sep='\t', skiprows=2))
    target_df = pd.DataFrame(pd.read_csv(target_gene_tpm_path, sep='\t', skiprows=2))
    input_df_filtered = pd.DataFrame(pd.read_csv(GTEx_input_df_filtered_path, sep=','))
    target_df_filtered = pd.DataFrame(pd.read_csv(GTEx_target_df_filtered_path, sep=','))

    print(f'GTEx gene tpm data frames:')
    #First three columns: id, Name, Description"
    print(f'Wholeblood')
    print(input_df.shape)
    print('Frontal Cortex')
    print(target_df.shape)

    #Gene tpm row sparsity histogram
    l0 = ['Whole blood', 'Frontal Cortex']
    l1 = ['input', 'target']
    if 1:
        print("Gene tpm row sparsity")
        dfs = [input_df, target_df]
        #Histogram
        if 0:
            for i in range(2):
                df = dfs[i]
                is_missing = df.isna() | (df == 0)
                row_sparsity = is_missing.sum(axis=1) / df.shape[1]
                bins = np.arange(0, 1.1, 0.1)
                counts, bins = np.histogram(row_sparsity, bins=bins)
                counts = counts / sum(counts)
                plt.bar(bins[:-1], counts, width=0.1, edgecolor='black', align='edge')
                title = f'{tissues[i]} gene tpm row sparsity'
                plt.title(title)
                plt.xlabel('Sparsity')
                plt.xticks(bins)
                plt.ylabel('Relative Frequency')
                plt.grid(True, alpha=0.5)
                plt.savefig(GTEx_Analysis_dir + rf'\{title}.png')
                plt.show()

    print(f'Common donors between Data frames: {len(GTEx_common_donors)}\n')
    #Original gene tpm, gene sets
    for i in range(2):
        print(f'GTEx {l0[i]} gene tpm, gene set lenght: {len(GTEx_gene_set[l1[i]])}')
    print()
    # Genes repetidos ? _PAR_Y?

    print(f'GTEx and STRING Wholeblood gene intersection set lenght: {len(GTEx_gene_intersection_set["input"])}')
    # 19039 genes do STRING interseção 56156 GTEx -> 18607 - Genes não partilhados em ambos os conjuntos
    print(f'GTEx Wholeblood filtered gene set lenght: {len(GTEx_filtered_gene_set["input"])}')
    print()
    find_genes = ["DRD1", "DRD2", "DRD3", "DRD4", "DRD5","SLC6A4","BDNF","COMT","DRD2","GRM3","DISC1","MAOA","NRG1","CACNA1C","FKBP5"]
    find_genes += ["GABRA1", "GABRA2", "GABRA3", "GABRA4", "GABRA5", "GABRA6", "GABRB1", "GABRB2", "GABRB3", "GABRD",
                   "GABRE", "GABRG1", "GABRG2", "GABRG3", "GABRP", "GABRQ", "GABRR1", "GABRR2", "GABRR3", "GABBR1",
                   "GABBR2"]

    print(f'Genes to find lenght: {len(find_genes)}')
    print(f'GTEx important genes lenght: {len(GTEx_target_important_genes)}')
    print(f'{target_tissue} gene tpm and important genes, gene intersection set lenght: {len(GTEx_gene_intersection_set["target"])}')
    print(f'GTEx {target_tissue} filtered gene set lenght: {len(GTEx_filtered_gene_set["target"])}')

    # Histogram
    dfs = [input_df, target_df]
    if 1:
        for i in range(2):
            df = dfs[i].iloc[:, 3:]
            print(df.shape)
            max_bin = 10
            width = 0.5
            counts_above = sum(df[df > max_bin].count())
            max_outlier = df.max().max()
            bins = np.arange(0, max_bin + width, width)
            counts, _ = np.histogram(df, bins=bins)
            total_counts = sum(counts) + counts_above
            # print(sum(counts))
            # print(counts_above)
            outlier_fraction = counts_above / total_counts
            print(f'{outlier_fraction * 100:.1f}% of data is outlier with max {max_outlier}')
            counts = counts / total_counts
            print(sum(counts))

            plt.bar(bins[:-1], counts, edgecolor='black', align='edge', width=width)
            title = f'{tissues[i]} transcripts per million'
            plt.title(title)
            plt.xlabel('TPM Values')
            plt.xticks(bins)
            plt.ylabel('Relative Frequency')
            #plt.yscale('log')
            plt.grid(True, alpha=0.5)
            plt.savefig(GTEx_Analysis_dir + rf'\{title}.png')
            plt.show()
    # Histogram
    dfs = [input_df_filtered, target_df_filtered]
    if 1:
        for i in range(2):
            df = dfs[i].iloc[:,3:]
            print(df.shape)
            max_bin = 10
            width = 0.5
            counts_above = sum(df[df > max_bin].count())
            max_outlier = df.max().max()
            bins = np.arange(0, max_bin+width, width)
            counts, _ = np.histogram(df, bins=bins)
            total_counts = sum(counts) + counts_above
            outlier_fraction = counts_above / total_counts
            print(f'{outlier_fraction*100:.1f}% of data is outlier with max {max_outlier}')
            counts = counts / total_counts
            
            plt.bar(bins[:-1], counts, edgecolor='black', align='edge', width=width)
            title = f'{tissues[i]} filtered transcripts per million'
            plt.title(title)
            plt.xlabel('TPM Values')
            plt.xticks(bins)
            plt.ylabel('Relative Frequency')
            #plt.yscale('log')
            plt.grid(True, alpha=0.5)
            plt.savefig(GTEx_Analysis_dir + rf'\{title}.png')
            plt.show()

