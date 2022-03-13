import numpy as np
import matplotlib
matplotlib.use('Agg')
from matplotlib import pyplot as plt
import seaborn as sns

def heatmap(matrix=None, out_path=None, boundary=[], diag=False, matrixPath=None, boundaryPath=None):
    if matrixPath:
        matrix = np.loadtxt(matrixPath)
    if boundaryPath:
        boundary = []
        with open(boundaryPath, "r") as result:
            lines = result.readlines()
            for line in lines:
                info = line.split()
                # boundary.append([int(info[0]), int(info[-1])])
                boundary.append([int(info[1]), int(info[5])])
    matrix = matrix.copy()
    plt.figure()
    sns.set()
    cmap = sns.cubehelix_palette(gamma=0.8, as_cmap=True)
    if diag:
        for i in range(len(matrix)):
            matrix[i, i] = 0
    ax = sns.heatmap(matrix, fmt="d", cmap=cmap)
    ax.axis('off')
    if len(boundary):
        ax = add_boundary(ax, boundary, linewidth=1.0)
    if not out_path:
        plt.show()
    else:
        plt.savefig(out_path + ".pdf", format='pdf')

def add_boundary(fig, boundary, color='cornflowerblue', linewidth=0.5):
    for i in range(0, len(boundary), 1):
        x = np.arange(boundary[i][0] - 1, boundary[i][1] + 1)
        y1 = (boundary[i][0] - 1) * np.ones(len(x))
        y2 = (boundary[i][1]) * np.ones(len(x))
        fig.plot(x, y1, color=color, linewidth=linewidth)
        fig.plot(y2, x, color=color, linewidth=linewidth)
        fig.plot(x, y2, color=color, linewidth=linewidth)
        fig.plot(y1, x, color=color, linewidth=linewidth)
    return fig

def read_boundary(file):
    boundary = []
    with open(file, "r") as result:
        lines = result.readlines()
        length = len(lines)
        for line in range(length):
            info = lines[line].split()
            # boundary.append(int(info[1]))   # start bin
            # boundary.append(int(info[5]))   # end bin
            boundary.append([int(info[1]), int(info[5])])
    return np.array(boundary)

def diff_boundary(boundList1, boundList2, ratio=True):
    """
    Calculate the proportion of common bounds from RAI result.
    :param boundList1: Hi-C inferred bounds
    :param boundList2: RAI inferred bounds
    :return:
    """
    bound1 = np.unique(boundList1)
    bound2 = np.unique(boundList2)
    boundCom = []
    boundUniqRAI = []
    for b2 in bound2:
        match = False
        for b1 in bound1:
            if abs(b1 - b2) <= 0:
                match = True
                break
        if match:
            boundCom.append(b2)
        else:
            boundUniqRAI.append(b2)

    if ratio:
        return np.round(len(boundCom) / len(bound2), 4)
    else:
        return boundCom, boundUniqRAI

def diff_domain(boundList1, boundList2):
    """
    Take Hi-C inferred TADs as ref, calculate the proportion of match, merge, split, and shift TADs.
    Roughly define three types of variations of RAI inferred TADs,
    where merge does not generate novel TAD-like domain boundaries,
    while split and shift do.
    :param boundList1: Hi-C inferred TADs
    :param boundList2: RAI inferred TADs
    :return:
    """
    threshold_align = 2
    threshold_shift = 5

    bound1 = np.unique(boundList1)
    bound2 = np.unique(boundList2)
    boundCom = []
    for b2 in bound2:
        match = False
        for b1 in bound1:
            if abs(b1 - b2) < threshold_align:
                match = True
                break
        if match:
            boundCom.append(b2) # boundCom contains all the aligned boundaries of RAI

    matchCount = 0
    mergeCount = 0
    splitCount = 0
    shiftCount = 0
    for tad2 in boundList2:
        if tad2[0] in boundCom:
            if tad2[1] in boundCom:
                state = 0   # candidate merge/match
            else:
                state = 1   # candidate split
        elif tad2[1] in boundCom:
            state = 1
        else:
            state = 2   # candidate shift

        for tad1 in boundList1:
            len1 = tad1[1] - tad1[0] + 1
            len2 = tad2[1] - tad2[0] + 1
            if abs(tad2[0] - tad1[0]) < threshold_align and abs(tad2[1] - tad1[1]) < threshold_align:
                matchCount += 1
                print("{} is a matched TAD with {}".format(tad2, tad1))
                break
            elif state == 2 and \
                    abs(tad2[0] - tad1[0]) < threshold_shift and \
                    abs(tad2[1] - tad1[1]) < threshold_shift:
                shiftCount += 1
                print("{} is a shifted TAD of {}".format(tad2, tad1))
                break
            elif state < 2:
                if abs(tad2[0] - tad1[0]) < threshold_align or abs(tad2[1] - tad1[1]) < threshold_align:
                    if len2 > len1:
                        mergeCount += 1
                        print("{} is a merged TAD from {}".format(tad2, tad1))
                        break
                    elif len1 > len2:
                        splitCount += 1
                        print("{} is a split TAD from {}".format(tad2, tad1))
                        break
    print("#match = {} #merge = {} #split = {} #shift = {}".format(matchCount, mergeCount, splitCount, shiftCount))
    return np.array([matchCount, mergeCount, splitCount, shiftCount]) * (1/len(boundList2))

def construct_dict_from_refgene(refgene, chrom, resolution, minBin, maxBin):
    # construct the hash from refgene, bin->genes
    geneList = [[] for i in range(maxBin - minBin + 1)]
    with open(refgene, "r") as annot:
        lines = annot.readlines()
        for line in lines:
            info = line.split()
            if info[2] == chrom:
                startbin = int(float(info[4]) / resolution)
                endbin = int(float(info[5]) / resolution)
                for i in range(startbin, endbin + 1):
                    if i + 1 >= minBin and i + 1 <= maxBin:
                        geneList[i - minBin + 1].append(info[12])  # official gene symbol
    return geneList

def construct_dict_from_rawdata(refgene, chrom, resolution, minBin, maxBin, mode):
    if mode == "radiclseq":
        gene = 6    # column
    elif mode == "gridseq":
        gene = 3    # column
    geneList = [[] for i in range(maxBin - minBin + 1)]
    with open(refgene, "r") as annot:
        lines = annot.readlines()
        for line in lines:
            info = line.split()
            if info[0] == chrom:
                startbin = int(float(info[1])/resolution)
                endbin = int(float(info[2])/resolution)
                for i in range(startbin, endbin + 1):
                    if i + 1 >= minBin and i + 1 <= maxBin:
                        info[gene] = info[gene].split(".")[0]
                        if info[gene] not in geneList[i - minBin + 1]:
                            geneList[i - minBin + 1].append(info[gene])
    return geneList

def gene_annotation(RAIresultFile, HICresultFile, refgene, outPath="./", mode="imargi"):
    """
    For iMARGI and RIC-seq: refgene;
    For RADICL-seq and GRID-seq: rawdata;
    :param RAIresultFile:
    :param HICresultFile:
    :param refgene:
    :param outPath:
    :return:
    """
    with open(RAIresultFile, "r") as result:
        lines = result.readlines()
        info = lines[0].split()
        chrom = info[0] # chrom
        resolution = int(info[3]) - int(info[2])    # resolution
    boundRAI = read_boundary(RAIresultFile)
    boundHIC = read_boundary(HICresultFile)
    minBin = int(np.min(boundRAI))
    maxBin = int(np.max(boundRAI))
    boundCom, boundUniqRAI = diff_boundary(boundHIC, boundRAI, ratio=False) # list of boundary bins

    if mode == "imargi" or mode == "ricseq":
        geneList = construct_dict_from_refgene(refgene, chrom, resolution, minBin, maxBin)
        suffix = "official_gene_symbol"
    else:
        geneList = construct_dict_from_rawdata(refgene, chrom, resolution, minBin, maxBin, mode)
        if mode == "radiclseq":
            suffix = "ensembml_geneid"
        elif mode == "gridseq":
            suffix = "flybase_geneid"
    # output the gene list
    geneCom = []
    for bound in boundCom:
        if bound != minBin and bound != maxBin:
            geneCom.extend(geneList[bound-minBin])
    geneCom = list(set(geneCom))    # remove the duplicates
    with open(outPath+"geneSet_combound_{}.txt".format(suffix), "w") as ocom:
        ocom.write("\n".join(geneCom))

    geneUniqRAI = []
    for bound in boundUniqRAI:
        if bound != minBin and bound != maxBin:
            geneUniqRAI.extend(geneList[bound-minBin])
    geneUniqRAI = list(set(geneUniqRAI))
    with open(outPath+"geneSet_uniqRAIbound_{}.txt".format(suffix), "w") as ouniq:
        ouniq.write("\n".join(geneUniqRAI))
    print(geneList[0], geneList[-1])






















