import os
os.environ["DGLBACKEND"] = "pytorch"

import json
import pandas as pd
import time
import numpy as np
import matplotlib.pyplot as plt


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
    GTex_gene_set_path = r'C:\Users\xamuc\Desktop\PIC1\DataSetup\GTex_gene_set.json'

    wholeblood_gene_tpm_path = r"C:\Users\xamuc\Desktop\PIC1\DataSetup\GTex\gene_tpm_whole_blood.gct"
    wholeblood_gene_tpm_graph_path = r'C:\Users\xamuc\Desktop\PIC1\DataSetup\GTex\gene_tpm_whole_blood_graph.csv'

    GTexString_gene_set_path = r'C:\Users\xamuc\Desktop\PIC1\DataSetup\STRING\GTexString_gene_set.json'
    common_donors_path = r'C:\Users\xamuc\Desktop\PIC1\DataSetup\GTex\comon_donors.json'
    find_genes_path = r'C:\Users\xamuc\Desktop\PIC1\DataSetup\Annotations\find_genes.json'

    frontalcortex_gene_tpm_path = r"C:\Users\xamuc\Desktop\PIC1\DataSetup\GTex\gene_tpm_brain_frontal_cortex_ba9.gct"
    frontalcortex_gene_tpm_graph_path = r"C:\Users\xamuc\Desktop\PIC1\DataSetup\GTex\gene_tpm_brain_frontal_cortex_graph.csv"

# ------------

#GTex common_donor_id set
if 0:
    wbdf = pd.DataFrame(pd.read_csv(wholeblood_gene_tpm_path, sep='\t', skiprows=2, nrows=0))
    fcdf = pd.DataFrame(pd.read_csv(frontalcortex_gene_tpm_path, sep='\t', skiprows=2, nrows=0))
    wb_donors = [donor[:10] for donor in wbdf.columns.tolist()[3:]]
    fc_donors = [donor[:10] for donor in fcdf.columns.tolist()[3:]]
    common_donors = list(set(wb_donors) & set(fc_donors))
    with open(common_donors_path, 'w') as f:
        json.dump({'common_donors':common_donors}, f)

#GTex Whole blood gene set - genes to use for graph
if 0:
    df = pd.DataFrame(pd.read_csv(wholeblood_gene_tpm_path, sep='\t', skiprows=2))

    with open(common_donors_path, 'r') as f:
        common_donors = json.load(f)['common_donors']

    #Common donors onlyy
    collumn_names = df.columns.tolist()
    for i in range(df.shape[1]-3):
        if collumn_names[i + 3][:10] not in common_donors:
            df.iloc[:, i + 3] = np.NaN
    df = df.dropna(axis=1, how='all')

    zero_counts = df.apply(lambda row: (row == 0).sum(), axis=1)
    sparsity = zero_counts.div(df.shape[1]).tolist()

    GTex_gene_set = set()
    for i in range(df.shape[0]):
        gene_code = df.iloc[i,1].split('.')[0]
        if sparsity[i] < 0.5:
            GTex_gene_set.add(gene_code)

    GTex_gene_set = list(GTex_gene_set)
    with open(GTex_gene_set_path, 'w') as json_file:
        json.dump({'GTex_gene_set':GTex_gene_set}, json_file)

    print(len(GTex_gene_set))

# -------------

#GTex Whole blood analysis
if 0:
    with open(common_donors_path, 'r') as f:
        common_donors = json.load(f)['common_donors']
    with open(GTexString_gene_set_path, 'r') as f:
        GTexString_gene_set = json.load(f)['GTexString_gene_set']

    print("Reading dataframe...")
    df = pd.DataFrame(pd.read_csv(wholeblood_gene_tpm_path, sep='\t', skiprows=2))

    collumn_names = df.columns.tolist()
    for i in range(df.shape[1]-3):
        if collumn_names[i + 3][:10] not in common_donors:
            df.iloc[:, i + 3] = np.NaN
    df = df.dropna(axis=1, how='all')

    print("Prepping data...")
    x = list(range(df.shape[0]))
    values = df.columns[3:]
    y = df.iloc[:, 3:].mean(axis=1).tolist()
    yerr = df.loc[x, values].std(axis=1).tolist()
    x, y, yerr = zip(*[(xi, yi, yerri) for xi, yi, yerri in zip(x, y, yerr)])
    avgzero = [yi for yi in y if yi != 0]
    print(f"avg zero = {len(avgzero)}")
    print("Plotting...")
    plt.errorbar(x, y, yerr=yerr, fmt='o', ecolor='red', capsize=5, label='Dados com erro')

    # Personalizar o gráfico
    plt.xlabel('Eixo X')
    plt.ylabel('Eixo Y')
    plt.title('Gráfico com Barras de Erro')
    plt.legend()

    # Exibir o gráfico
    plt.show()

# ---------------

# Generate Whole blood Data Frame for Graph
if 0:
    with open(GTexString_gene_set_path, 'r') as f:
        GTexString_gene_set = json.load(f)['GTexString_gene_set']
    with open(common_donors_path, 'r') as f:
        common_donors = json.load(f)['common_donors']

    df = pd.DataFrame(pd.read_csv(wholeblood_gene_tpm_path, sep='\t', skiprows=2))
    wb_cols = 755
    print("Loaded df")
    gwb_cols = len(common_donors)+1
    gdf = pd.DataFrame([[np.NaN] * gwb_cols for _ in range(len(GTexString_gene_set))], columns = range(gwb_cols))
    print("gdf Created")

    #Only keep common donors
    collumn_names = df.columns.tolist()
    for i in range(wb_cols):
        if collumn_names[i+3][:10] not in common_donors:
            df.iloc[:,i+3] = np.NaN
    df = df.dropna(axis=1, how='all')
    gdf.columns = df.columns[2:]
    print("Set common collumns")

    #Remove Par Y
    version = df.iloc[:,1].str.split('.').str[1]
    df = df[~version.str.contains('PAR_Y', na=False)]

    #Sort genes for index to correspond to graph node
    gene_to_index = {gene: GTexString_gene_set.index(gene) for gene in GTexString_gene_set}
    gene_list = [gene.split(".")[0] for gene in df['Name'].tolist()]
    #Fill graph dataframe
    bs = 1000
    n_batches = len(gene_list) // bs + 1
    for bi in range(n_batches):
        I = np.arange(bi*bs, min((bi+1)*bs,len(gene_list)))
        indexes = [gene_to_index.get(gene_list[i]) for i in I]
        filtered_I_indexes = [(i, index) for i, index in zip(I, indexes) if index is not None]
        if filtered_I_indexes != []:
            I, indexes = zip(*filtered_I_indexes)
            gdf.iloc[list(indexes), 0] = df.iloc[list(I), 1]
            gdf.iloc[list(indexes),1:] = df.iloc[list(I), 3:].values

        EstimateTimePercent(bi*bs, df.shape[0], 1000)

    #Normalize data
    averages = gdf.iloc[:,1:].mean(axis=1).tolist()
    gdf.iloc[:,1:] = gdf.iloc[:,1:].div(averages, axis=0)

    gdf.to_csv(wholeblood_gene_tpm_graph_path, index=False)

# Generate Frontal Cortex Data Frame for Graph
if 0:
    with open(GTexString_gene_set_path, 'r') as f:
        GTexString_gene_set = json.load(f)['GTexString_gene_set']

    with open(common_donors_path, 'r') as f:
        common_donors = json.load(f)['common_donors']

    with open(find_genes_path, 'r') as f:
        find_genes = json.load(f)['find_genes']

    df = pd.DataFrame(pd.read_csv(frontalcortex_gene_tpm_path, sep='\t', skiprows=2))
    print("Loaded df")
    gfc_cols = len(common_donors) + 1
    gfc_rows = len(find_genes)
    gdf = pd.DataFrame([[np.NaN] * gfc_cols for _ in range(gfc_rows)], columns = range(gfc_cols))
    print("gdf Created")

    collumn_names = df.columns.tolist()
    for i in range(df.shape[1]-3):
        if collumn_names[i + 3][:10] not in common_donors:
            df.iloc[:, i + 3] = np.NaN
    df = df.dropna(axis=1, how='all')
    gdf.columns = df.columns[2:]

    gene_to_index = {gene: find_genes.index(gene) for gene in find_genes}
    gene_list = np.array([gene.split(".")[0] for gene in df['Name'].tolist()])

    bs = 1000
    n_batches = df.shape[0] // bs + 1

    for bi in range(n_batches):
        I = np.arange(bi*bs, min((bi+1)*bs,len(gene_list)))
        indexes = [gene_to_index.get(gene_list[i]) for i in I if gene_list[i] in find_genes]
        filtered_I_indexes = [(i, index) for i, index in zip(I, indexes) if index is not None]

        if filtered_I_indexes:
            I, indexes = zip(*filtered_I_indexes)
            gdf.iloc[list(indexes), 0] = df.iloc[list(I), 1]
            gdf.iloc[list(indexes), 1:] = df.iloc[list(I), 3:].values
        EstimateTimePercent(bi*bs, df.shape[0], 1000)

    # Remove sparse rows
    zero_counts = gdf.apply(lambda row: (row == 0).sum(), axis=1)
    sparsity = zero_counts.div(gdf.shape[1]).tolist()
    for i in range(gdf.shape[0]):
        if sparsity[i] > 0.5:
            gdf.iloc[i] = np.NaN
    gdf = gdf.dropna(axis=0, how='all')

    #Normalize
    averages = gdf.iloc[:, 1:].mean(axis=1)
    gdf.iloc[:, 1:] = gdf.iloc[:, 1:].div(averages, axis=0)

    gdf.to_csv(frontalcortex_gene_tpm_graph_path, index = False)

# -------------
# Graph WB analysis
if 0:
    df = pd.DataFrame(pd.read_csv(wholeblood_gene_tpm_graph_path, sep=','))
    x = range(df.shape[0])
    mean = df.iloc[:,1:].mean(axis=1)
    stdev = df.iloc[:,1:].std(axis=1)
    print(f'min std: {min(stdev)} max std: {max(stdev)}')
    plt.errorbar(x,mean, yerr= stdev)
    plt.show()

#Graph FC analysis
if 0:
    df = pd.DataFrame(pd.read_csv(frontalcortex_gene_tpm_graph_path, sep=','))
    x = range(df.shape[0])
    mean = df.iloc[:, 1:].mean(axis=1)
    stdev = df.iloc[:, 1:].std(axis=1)
    print(f'min std: {min(stdev)} max std: {max(stdev)}')
    plt.errorbar(x, mean, yerr=stdev)
    plt.show()