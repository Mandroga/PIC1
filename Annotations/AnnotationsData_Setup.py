import json
import numpy as np
import time

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
    annotations_path = r'C:\Users\xamuc\Desktop\PIC1\DataSetup\Annotations\Homo_sapiens.GRCh38.111.gtf'
    protein_set_path = r'C:\Users\xamuc\Desktop\PIC1\DataSetup\STRING\protein_set.json'
    protein_to_gene_path = r'C:\Users\xamuc\Desktop\PIC1\DataSetup\Annotations\protein_to_gene.json'

    GTexString_gene_set_path = r'C:\Users\xamuc\Desktop\PIC1\DataSetup\STRING\GTexString_gene_set.json'
    find_genes_path = r'C:\Users\xamuc\Desktop\PIC1\DataSetup\Annotations\find_genes.json'

#Generate Protein to Gene Dict
if 0:
        i = 0
        with open(protein_set_path, 'r') as f:
            protein_set = json.load(f)['protein_set']

        protein_to_gene = {}
        for protein in protein_set:
            protein_to_gene[protein] = set()

        with open(annotations_path, 'r') as f:
            bs = 10000
            bi = 0
            for _ in range(5): next(f)  # skip metadata
            while True:
                try:
                    lines = np.array([next(f) for _ in range(bs)])
                    gene_indexes = np.char.find(lines, "gene_id") + 9
                    protein_indexes = np.char.find(lines, "protein_id") + 12
                    protein_gene_ids = [(line[pidx:pidx + 15], line[gidx:gidx + 15]) for line, gidx, pidx in zip(lines, gene_indexes, protein_indexes) if pidx != 11 and line[pidx:pidx + 15] in protein_set]
                    for pi, gi in protein_gene_ids:
                        protein_to_gene[pi].add(gi)
                    EstimateTimePercent(bi*bs, 3420000, bs)
                    bi += 1
                except: break

            for protein in protein_set:
                protein_to_gene[protein] = list(protein_to_gene[protein])

            with open(protein_to_gene_path, 'w') as f:
                json.dump(protein_to_gene, f)

#Fing gene codes
if 1:
    #with open()
    with open(annotations_path, 'r') as f:
        bs = 15000
        bi = 0
        for _ in range(5): next(f)  # skip metadata
        find_genes = [f'DRD{i+1}' for i in range(5)]
        find_genes += ["SLC6A4","BDNF","COMT","DRD2","GRM3","DISC1","MAOA","NRG1","CACNA1C","FKBP5"]
        genes_id_names = set()
        while True:
            try:
                lines = np.array([next(f) for _ in range(bs)])
                #print(lines)
                gene_name_indexes = np.char.find(lines, "gene_name") + 11
                gene_id_indexes = np.char.find(lines, "gene_id") + 9
                genes_id_names.update([(line[giidx:giidx + 15], line[gnidx:gnidx + 4]) for line, giidx, gnidx in zip(lines, gene_id_indexes, gene_name_indexes) if gnidx != 10 and line[gnidx:gnidx + 4] in find_genes])
                EstimateTimePercent(bi * bs, 3420000, bs)
                bi += 1
            except: break
        gene_ids, gene_names = zip(*genes_id_names)
        print(genes_id_names)
        print(gene_ids)
        print(gene_names)

        with open(find_genes_path, 'w') as f:
            json.dump({'find_genes':gene_ids}, f)
        # diferentes genes com o mesmo nome ? DRD5
