#!/usr/bin/python
# -*- coding: utf-8 -*-
import argparse, os, time, multiprocessing
import math
import pulp
import numpy as np
from .MultichildTree import MultiChildTree

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("-i", dest="result", nargs="+", required=True)
    parser.add_argument("-n", dest="count", type=int, required=True)
    parser.add_argument("-b", dest="binInput", action="store_true", help="if given, the input is binList")
    parser.add_argument("--list", dest="listInput", action="store_true", help="if given, the input is a list of results")
    parser.add_argument("--suffix", dest="suffix", default=".binary.original.tsv", help="default: .binary.original.tsv")
    args = parser.parse_args()
    compare_list = []
    if args.listInput == True:
        for line in open(args.result[0], "r"):
            line = line.strip("\n")
            compare_list.append(read_result(line+args.suffix, args.binInput))
    else:
        for i in args.result:
            compare_list.append(read_result(i, args.binInput))
    Compare(compare_list, args.count, suffix=args.suffix)


class Compare():
    def __init__(self, compare_list, binN, suffix="", output_path="."):
        self.n = binN
        self.clusters1 = compare_list[0]    # truth
        self.leaflist1 = self.get_leafList(self.clusters1)
        pool = multiprocessing.Pool()
        result = pool.map(self.run, compare_list)
        pool.close()
        pool.join()
        self.output = np.array(result)
        # np.savetxt(output_path+"/similarity_log_{}.txt".format(suffix), self.output)

    def run(self, cluster):
        self.clusters2 = cluster
        self.leaflist2 = self.get_leafList(self.clusters2)
        overlap = round(self.similarity1(), 4)
        # weighted = round(self.similarity2(), 4)
        # mod = round(self.calculate_moc(), 4)
        nmi = round(self.normMI(), 4)
        # ari = round(self.ARI(), 4)
        # print(overlap, weighted, mod, nmi, ari)
        # return [overlap, weighted, mod, nmi, ari]
        # print(overlap, nmi)
        return [overlap, nmi]

    def Output(self):
        return self.output

    def similarity1(self):
        total_length_1 = len(sum(self.clusters1, []))
        total_length_2 = len(sum(self.clusters2, []))
        # print(total_length_1, total_length_2)
        self.wts = {}
        from_nodes = [k for k in range(1, len(self.clusters1) + 1)]
        to_nodes = [k for k in range(1, len(self.clusters2) + 1)]
        self.ucap = {}
        self.vcap = {}
        for i in range(0, len(from_nodes)):
            self.ucap[from_nodes[i]] = 1
        for i in range(0, len(to_nodes)):
            self.vcap[to_nodes[i]] = 1
        for i in range(0, len(self.clusters1)):
            for j in range(0, len(self.clusters2)):
                interc = list(set(self.clusters1[i]).intersection(set(self.clusters2[j])))
                self.wts[(i + 1, j + 1)] = 2 * len(interc)  # weight assigned when calculating overlapping
        wt = self.create_wt_doubledict(from_nodes, to_nodes)
        p = self.solve_wbm(from_nodes, to_nodes, wt)
        count, selected_edges = self.get_selected_edges(p)
        # print("Sum of wts of selected edges = ", self.print_solution(p) / (total_length_1 + total_length_2))
        # return self.print_solution(p) / count
        return self.print_solution(p) / (total_length_1 + total_length_2)

    # just a convenience function to generate a dict of dicts
    def create_wt_doubledict(self, from_nodes, to_nodes):
        wt = {}
        for u in from_nodes:
            wt[u] = {}
            for v in to_nodes:
                wt[u][v] = 0
        for k, val in self.wts.items():
            u, v = k[0], k[1]
            wt[u][v] = val
        return wt

    def solve_wbm(self, from_nodes, to_nodes, wt):
        ''' A wrapper function that uses pulp to formulate and solve a WBM'''

        prob = pulp.LpProblem("WBM Problem", pulp.LpMaximize)
        # Create The Decision variables
        choices = pulp.LpVariable.dicts("e", (from_nodes, to_nodes), 0, 1, pulp.LpInteger)
        # Add the objective function
        prob += pulp.lpSum([wt[u][v] * choices[u][v]
                            for u in from_nodes
                            for v in to_nodes])  # Calculate the sum of a list of linear expressions
        # Constraint set ensuring that the total from/to each node is less than its capacity
        for u in from_nodes:
            for v in to_nodes:
                prob += pulp.lpSum([choices[u][v] for v in to_nodes]) <= self.ucap[u], ""
                prob += pulp.lpSum([choices[u][v] for u in from_nodes]) <= self.vcap[v], ""
        # The problem data is written to an .lp file
        # prob.writeLP("WBM.lp")
        # The problem is solved using PuLP's choice of Solver
        prob.solve()
        # The status of the solution is printed to the screen
        # print("Status:", pulp.LpStatus[prob.status])
        return prob

    def print_solution(self, prob):
        return round(pulp.value(prob.objective), 4)

    def get_selected_edges(self, prob):
        selected_from = [v.name.split("_")[1] for v in prob.variables() if v.value() > 1e-3]
        selected_to = [v.name.split("_")[2] for v in prob.variables() if v.value() > 1e-3]
        selected_edges = []
        for su, sv in list(zip(selected_from, selected_to)):
            selected_edges.append((su, sv))
            # print(su, sv, self.wts[(int(su), int(sv))])
        return len(selected_edges), selected_edges

    def similarity2(self):
        """
        The weighted similarity between Q and P.
        :param self.clusters1: Q
        :param self.clusters2: P, as the denominator
        :return:
        """
        numerator = 0
        denominator = 0
        for i in range(0, len(self.clusters1)):
            weight_of_everytad1 = []
            for j in range(0, len(self.clusters2)):
                interc = list(set(self.clusters1[i]).intersection(set(self.clusters2[j])))
                simi = len(interc) / (math.sqrt(len(self.clusters1[i]) * len(self.clusters2[j])))
                weight_of_everytad1.append(simi)
            numerator += len(self.clusters1[i]) * max(weight_of_everytad1)
            denominator += len(self.clusters1[i])
        print("weighted similarity = ", numerator / denominator)
        return numerator / denominator

    def get_leafList(self, tad_list):
        data_list_new = sorted(tad_list, key=lambda x: len(x), reverse=True)
        multichildtree = MultiChildTree(1, self.n)
        for i in range(0, len(data_list_new)):
            # print(data_list_new[i][0], data_list_new[i][-1])
            multichildtree.insert(data_list_new[i][0], data_list_new[i][-1], "F")
        node_list = multichildtree.acquire_list()
        leaf_list = []
        total_conv = []
        for i in range(0, len(node_list)):
            if len(node_list[i].child) == 0:
                bin_list = list(range(node_list[i].val[0], node_list[i].val[1] + 1))
                leaf_list.append(bin_list)
                total_conv.extend(bin_list)
        if len(total_conv) != self.n:  # fill in gaps
            # print(len(total_conv), "self.n", self.n)
            covered = True
            gap = []
            for i in range(1, self.n + 1, 1):
                if i not in total_conv and covered == True:
                    gap = []
                    gap.append(i)
                    covered = False
                elif i not in total_conv and covered == False:
                    gap.append(i)
                elif i in total_conv and covered == False:
                    leaf_list.append(gap)
                    total_conv.extend(gap)
                    covered = True
                else:
                    pass
            # print("proof: #bin = {}, #covered bin = {}".format(self.n, len(total_conv)))
            if len(total_conv) > self.n:
                print("&&&&&&&&&&&&&&&&&&&&&&&&&&&&error&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&")
        return leaf_list

    def calculate_moc(self):
        tad_n1 = len(self.leaflist1)
        tad_n2 = len(self.leaflist2)
        if tad_n1 == 1 and tad_n2 == 1:
            return 1
        else:
            sum = 0
            for i in range(tad_n1):
                for j in range(tad_n2):
                    interc = len(set(self.leaflist1[i]).intersection(set(self.leaflist2[j])))
                    sum += pow(interc, 2) / (len(self.leaflist1[i]) * len(self.leaflist2[j]))
            # print("sum = {}".format(sum))
            return (sum - 1) / (np.sqrt(tad_n1 * tad_n2) - 1)

    def normMI(self):
        """
        The normalized mutual information (NMI) is to compare clustering with different numbers of clusters.
        NMI is always a number btw 0 and 1.
        :return: The value of NMI.
        """
        MI = 0
        H1 = 0
        H2 = 0
        time = 0
        for c1 in self.leaflist1:
            len1 = len(list(set(c1)))
            H1 -= (len1/self.n)*np.log2(len1/self.n)
            time += 1
            for c2 in self.leaflist2:
                len2 = len(list(set(c2)))
                if time == 1:
                    H2 -= (len2/self.n)*np.log2(len2/self.n)
                intersection = len(list(set(c1)&set(c2)))
                # print("len1={}, len2={}, inter={}, MI={}".format(len1, len2, intersection, MI))
                if intersection != 0:
                    MI += (intersection/self.n)*np.log2((self.n*intersection)/(len1*len2))
        return MI/np.sqrt(H1 * H2)

    def ARI(self):
        sum_inter = 0
        sum_a = 0
        sum_b = 0
        time = 0
        for c1 in self.leaflist1:
            len1 = len(list(set(c1)))
            sum_a += len1 * (len1 - 1) / 2.0
            time += 1
            for c2 in self.leaflist2:
                len2 = len(list(set(c2)))
                if time == 1:
                    sum_b += len2 * (len2 - 1) / 2.0
                intersection = len(list(set(c1) & set(c2)))
                sum_inter += intersection * (intersection - 1)/ 2.0
        expectedIndex = sum_a*sum_b*2.0/(self.n*(self.n-1))
        return (sum_inter-expectedIndex)/(0.5*(sum_a+sum_b)-expectedIndex)


def read_result(dirnames, binInput=False):
    if os.path.exists(dirnames):
        result_list = []
        with open(dirnames, "r") as result:
            lines = result.readlines()
            for line in lines:
                line = line.split()
                if line != []:
                    if binInput:
                        a = list(map(int, line))
                    else:
                        a = list(range(int(line[1]), int(line[5])+1))
                    result_list.append(a)
        # print("Successfully loading", dirnames, "with", len(result_list), "lines.")
        if len(result_list) == 0:
            return [[0]]
        return result_list
    else:
        return [[0]]


if __name__ == "__main__":
    main()
