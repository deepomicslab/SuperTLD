#!/usr/bin/python
import argparse
import numpy as np
import seaborn as sns
import matplotlib

matplotlib.use('Agg')
import matplotlib.pyplot as plt
import random, multiprocessing, os
from statsmodels.stats.multitest import multipletests

class globalPar:
    stepFactor = 0.1   # step is 0.1*resolution, hic: 0.1
    iterN = 1000    # iteraction time, hic: 1000
    pvalue = 0.1    # FDR corrected pvalue threshold, hic: 0.1

class Enrichment():
    def __init__(self, bedFILE, resultFILE, resolution, start, end, chr, output=None, plot=False):
        self.chr = chr
        self.resolution = resolution
        self.step = globalPar.stepFactor * self.resolution
        self.start = start
        self.end = end
        self.result = []
        if not plot:
            for i in range(len(resultFILE)):
                result = self.read_result(resultFILE[i])
                self.result.append(result)
        else:
            with open(resultFILE[0], "r") as result:
                lines = result.readlines()
                for line in lines:
                    tad = line.split()
                    self.result.append([[int(tad[3]), int(tad[6])]])
            # print(self.result)
        self.profile = []
        for i in range(len(bedFILE)):
            self.profile.append(self.read_profile(bedFILE[i]))
        self.output = os.path.dirname(output)
        self.enrichment_class = np.zeros(shape=(len(self.result), 3))  # each result: %enrich_K27, %enrich_K36, %no enrich
        self.run()

    def run(self):
        pool = multiprocessing.Pool()
        result = pool.map(self.run_per_result, self.result)
        pool.close()
        pool.join()
        self.enrichment_class = np.array(result)
        # print(self.enrichment_class)
        # np.savetxt(self.output + "/enrichment_" + self.chr + "histonemark.txt", self.enrichment_class, fmt='%.4f')

    def read_profile(self, bedFILE):
        binN = int((self.end - self.start) / self.step)
        profile = np.zeros(int(binN))
        with open(bedFILE, "r") as bed:
            lines = bed.readlines()
            for peak in lines:
                peak = peak.split()
                peak_start = int(peak[1])
                peak_end = int(peak[2])
                if peak[0] == self.chr:
                    length = peak_end - peak_start
                    fc = float(peak[3])
                    if peak_start >= self.start and peak_end <= self.end:
                        left = int((peak_start - self.start) / self.step)
                        right = int((peak_end - self.start) / self.step)
                        if left == right:
                            profile[left] += fc * length
                        else:
                            left_part = self.start + self.step * right - peak_start
                            profile[left] += fc * left_part
                            for mid in range(right - left - 1):
                                profile[left + mid + 1] += fc * self.step
                            profile[right] += fc * (length - left_part - (right - left - 1) * self.step)
                    elif peak_start < self.start and peak_end > self.start:
                        right = int((peak_end - self.start) / self.step)
                        if right == 0:
                            profile[0] += fc * (peak_end - self.start)
                        else:
                            for mid in range(right):
                                profile[mid] += fc * self.step
                            profile[right] += fc * (peak_end - right * self.step - self.start)
                    elif peak_start < self.end and peak_end > self.end:
                        left = int((peak_start - self.start) / self.step)
                        if left == binN - 1:
                            profile[left] += fc * (self.end - peak_start)
                        else:
                            for mid in range(binN - 1, left, -1):
                                profile[mid] += fc * self.step
                            profile[left] += fc * (self.end - (binN - 1 - left) * self.step - peak_start)

        return profile / self.step

    def read_result(self, resultFILE):
        """
        Derive the list of TADs.
        :param resultFILE:
        :return:
        """
        tad_list = []
        with open(resultFILE, "r") as result:
            lines = result.readlines()
            for line in lines:
                tad = line.split()
                if int(tad[3]) < int(tad[6]):
                    tad_list.append([int(tad[3]), int(tad[6])])
                elif int(tad[2]) < int(tad[7]):
                    tad_list.append([int(tad[2]), int(tad[7])])
                else:
                    print(tad)
            # tad_list.append([int(tad[1])*self.resolution, int(tad[5])*self.resolution-self.resolution])
        return tad_list

    def run_per_result(self, result):
        tadN = len(result)

        avg_lr = []
        # calculate the observed average LR value
        for tad in result:
            tad_size = tad[1] - tad[0]
            start_bin = int((tad[0] - self.start) / self.step)
            avg_lr.append(self.calculate_aver_lr(self.profile, start_bin, tad_size))

        # Shuffle intervals
        shuff_lr = []
        ori_list = list(range(len(self.profile[0])))
        for time in range(globalPar.iterN):  # Shuffle times
            shuf_pos = random.sample(ori_list, len(ori_list))
            new_profile = []
            new_profile.append(self.profile[0][shuf_pos])
            new_profile.append(self.profile[1][shuf_pos])
            for tad in result:
                tad_size = tad[1] - tad[0]
                start_bin = int((tad[0] - self.start) / self.step)
                shuff_lr.append(self.calculate_aver_lr(new_profile, start_bin, tad_size))
        median_shuf_lr = np.median(shuff_lr)

        # calculate pvalue per tad
        pvalue = np.zeros(tadN)
        tad_state = np.zeros(tadN)
        for tad in range(tadN):
            query = avg_lr[tad]
            if query >= median_shuf_lr:
                pvalue[tad] = np.sum(shuff_lr >= query) / len(shuff_lr)
                tad_state[tad] = 1  # enriched for K27_value
            else:
                pvalue[tad] = np.sum(shuff_lr <= query) / len(shuff_lr)
                tad_state[tad] = -1 # enriched for K36_value
        # print(query, median_shuf_lr, pvalue[tad])

        # correction using BH
        reject, qvalue, _, _ = multipletests(pvalue, method='fdr_bh', alpha=globalPar.pvalue)
        for tad in range(len(reject)):
            if reject[tad] == False:
                tad_state[tad] = 0
        return [len(tad_state[tad_state == 1]) / tadN, len(tad_state[tad_state == -1]) / tadN, len(tad_state[tad_state == 0]) / tadN]

    def Output(self):
        return self.enrichment_class[:, 0] + self.enrichment_class[:, 1]

    def calculate_aver_lr(self, profile, start_bin, tad_size):
        interval_list = []
        for i in range(int(tad_size / self.step)):
            K27_value = profile[0][start_bin + i]
            K36_value = profile[1][start_bin + i]
            if K27_value > 0 and K36_value > 0:
                interval_list.append(np.log10(K27_value / K36_value))
        try:
            avg_lr = np.sum(interval_list) * self.step / tad_size
        except:
            print("error::", start_bin, tad_size, interval_list)
            avg_lr = 0
        return avg_lr


def main():
    print("blocksize={}, iterN={}, pvalue={}".format(globalPar.stepFactor, globalPar.iterN, globalPar.pvalue))
    parser = argparse.ArgumentParser()
    parser.add_argument("--bed", dest="bed", nargs="+", required=True, help="Epigenome bedgraph file, eg. H3K*me3")
    parser.add_argument("-i", dest="result", nargs="+", required=True, help="TADcaller result file in 8col format")
    parser.add_argument("-r", dest="resolution", type=int, required=True)
    parser.add_argument("-c", dest="chrom", required=True)
    parser.add_argument("-o", dest="output", default=None)
    parser.add_argument("--plot", dest="plot", action="store_true", help="if given, draw the bound profile plot")
    parser.add_argument("-s", dest="start", type=int, default=0, required=True, help="the start bp position")
    parser.add_argument("-e", dest="end", type=int, default=np.inf, required=True, help="the end bp position")
    parser.add_argument("--list", dest="list", action="store_true", help="if given, the input is a list of results")
    parser.add_argument("--suffix", dest="suffix", default=".multi2D_AllH2.tsv", help="default: .binary.original.tsv")
    args = parser.parse_args()

    if args.output == None:
        output = os.path.splitext(args.result[0])[0]
    else:
        output = args.output
    if args.list == True:
        resultList = []
        for line in open(args.result[0], "r"):
            line = line.strip("\n")
            resultList.append(line + args.suffix)
    else:
        resultList = []
        for i in args.result:
            resultList.append(i+args.suffix)
    Enrichment(args.bed, resultList, args.resolution, args.start, args.end, args.chrom, output, args.plot)


if __name__ == "__main__":
    main()
