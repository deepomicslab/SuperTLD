# _*_ coding: utf-8 _*_
"""
Time:     2021/12/17 17:35
Author:   ZHANG Yuwei
Version:  V 0.1
File:     evaluation.py
Describe:
"""

import numpy as np
from .distance_decay import get_pcc, calculate_dis_decay, pearsonr
from .SImi_tad import Compare, read_result
from .enrichment_boun_foldchange import FoldChange
from .enrichment_tad import Enrichment


class Evaluate():
    def __init__(self, chrom="chr1", resolution=100000, hicPath=None, bed=None, bedgraph=None, outputName="evaluation"):
        self.chrom = chrom
        self.resolution = resolution
        self.hicPath = hicPath
        self.Nbin = 0
        self.outputName = outputName
        self.sparse = True
        self.suffix = ".multi2D_AllH2_sparse.tsv"
        self.bed = bed
        self.bedgraph = bedgraph

    def run(self, resultList, outPath="./"):
        self.fileList = resultList
        self.outPath = outPath
        # distance decay
        matrix_simi = self.distance_decay_simi()
        # overlapping ratio, nmi
        simi = self.evaluate_simi()
        # enrichment
        resultList = [self.hicPath+self.suffix]
        for i in self.fileList:
            resultList.append(i + self.suffix)
        bound_enrich = self.evaluate_bound_enrich(resultList)  # each row: fold change, p-value
        tad_enrich = self.evaluate_tad_enrich(resultList)
        resultMatrix = np.concatenate((matrix_simi, simi, bound_enrich, np.reshape(tad_enrich, (-1, 1))), axis=1)
        np.savetxt(self.outPath, resultMatrix)
        return resultMatrix

    def distance_decay_simi(self):
        hic = np.loadtxt(self.hicPath)
        hicDecay = calculate_dis_decay(hic)
        self.Nbin = len(hic)
        matrix_simi = [[get_pcc(hic, hic), pearsonr(hicDecay, hicDecay)[0]]]
        for i in range(len(self.fileList)):
            matrixTmp = np.loadtxt(self.fileList[i])
            if len(matrixTmp) != self.Nbin:
                raise ValueError("Matrix's shape doesn't match!")
            decayTmp = calculate_dis_decay(matrixTmp)
            matrix_simi.append([get_pcc(hic, matrixTmp), pearsonr(hicDecay, decayTmp)[0]])
        return np.array(matrix_simi)    # 1+Nfile

    def evaluate_simi(self):
        compare_list = [read_result(self.hicPath + self.suffix)]
        for i in self.fileList:
            compare_list.append(read_result(i + self.suffix))
        result = Compare(compare_list, self.Nbin, self.suffix, output_path=self.outPath).Output()
        # result = np.r_[np.ones((1, len(result[0]))), result]
        return result   # 1+Nfile

    def evaluate_bound_enrich(self, resultList):
        if not self.bed:
            return np.zeros((len(resultList), 2))
        else:
            return FoldChange(self.bed, resultList, self.resolution, 0, int(self.Nbin * self.resolution), self.chrom, output=self.outPath).Output()

    def evaluate_tad_enrich(self, resultList):
        if not self.bedgraph:
            return np.zeros(len(resultList))
        else:
            return Enrichment(self.bedgraph, resultList, self.resolution, 0, int(self.Nbin * self.resolution), self.chrom, output=self.outPath).Output()