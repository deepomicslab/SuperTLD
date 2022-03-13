#!/usr/bin/python
import argparse, multiprocessing, os
import numpy as np
import seaborn as sns
import matplotlib

matplotlib.use('Agg')
import matplotlib.pyplot as plt
from scipy import interpolate
from scipy.stats import norm


class globalPar:
	step = 5000  # splitting to small blocks with step size, hic: 5000
	peakbinN = 1  # define the size of peak region through #bin, hic: 3
	backgroundbpSize = 50000  # define the size of background region, hic: 100kb
	maxbpSize = 500000  # consider the max size around the boundary, hic: 500kb


def main():
	print("step={}, peakbinN={}, backgSize={}, maxSize={}".format(globalPar.step, globalPar.peakbinN,
																  globalPar.backgroundbpSize, globalPar.maxbpSize))
	parser = argparse.ArgumentParser()
	parser.add_argument("--bed", dest="bed", nargs="+", required=True, help="Epigenome bed file, eg. CTCF")
	parser.add_argument("-i", dest="result", nargs="+", required=True, help="TADcaller result file in 8col format")
	parser.add_argument("-r", dest="resolution", type=int, required=True)
	parser.add_argument("-c", dest="chrom", required=True)
	parser.add_argument("-o", dest="output", default=None)
	parser.add_argument("--plot", dest="plot", action="store_true", help="if given, draw the bound profile plot")
	parser.add_argument("-s", dest="start", type=int, default=0, help="the start bp position")
	parser.add_argument("-e", dest="end", type=int, default=np.inf, required=True, help="the end bp position")
	parser.add_argument("--common", dest="common", action="store_true", help="if given, evaluate the common part of two results")
	parser.add_argument("--list", dest="list", action="store_true", help="if given, the input is a list of results")
	parser.add_argument("--suffix", dest="suffix", default=".multi2D_AllH2.tsv", help="default: .binary.original.tsv")
	args = parser.parse_args()

	if args.output == None:
		output = os.path.split(os.path.realpath(args.result[0]))[0]
	else:
		output = args.output
	if args.list == True:
		resultList = []
		for line in open(args.result[0], "r"):
			line = line.strip("\n")
			resultList.append(line + args.suffix)
		FoldChange(args.bed, resultList, args.resolution, args.start, args.end, args.chrom, output, args.plot,
				   args.common)
	else:
		resultList = []
		for result in args.result:
			resultList.append(result + args.suffix)
		FoldChange(args.bed, resultList, args.resolution, args.start, args.end, args.chrom, output, args.plot,
				   args.common)


class FoldChange():
	def __init__(self, bedFILE, resultFILE, resolution, start, end, chr, output=None, plot=False, common=False):
		self.chr = chr
		self.name = resultFILE[0]
		self.start = start - globalPar.maxbpSize
		self.end = end + globalPar.maxbpSize
		# self.step = int(0.1 * resolution)
		self.step = globalPar.step
		self.profile = []
		for i in range(len(bedFILE)):
			self.profile.append(self.read_profile(bedFILE[i]))
		self.resolution = resolution
		self.peak = np.zeros(int((globalPar.peakbinN * self.resolution) / self.step))
		# self.peak = np.zeros(int(2 * gap / self.step))
		self.background = np.zeros(int(globalPar.backgroundbpSize / self.step))
		self.output = os.path.dirname(output)
		self.plot = plot
		self.result = []
		for i in range(len(resultFILE)):
			self.result.append(self.read_result(resultFILE[i]))
			print("{}th result with {} boundaries".format(i, len(self.result[i])))
		self.common = common
		if self.common:	# only two result files
			self.boundB = []
			self.result = self.deriveCommon(self.result)
		self.plotprofile = None
		self.foldchangeResult = np.zeros((len(self.result), len(self.profile) * 2))
		self.profileTmp = None
		if self.plot:
			self.run()
		else:
			self.run_fast()

	def run_fast(self):
		for i in range(len(self.profile)):
			self.profileTmp = self.profile[i]
			pool = multiprocessing.Pool()
			result = pool.map(self.calculate_fold, self.result)
			pool.close()
			pool.join()
			for j in range(len(self.result)):
				self.foldchangeResult[j, i*2], self.foldchangeResult[j, 2*i+1] = result[j]
		# np.savetxt(self.output + "/enrichment_" + self.chr + "foldchange.txt", self.foldchangeResult, fmt='%.4f')

	def run(self):
		# color = sns.color_palette("Paired_r", 14)
		color = sns.color_palette("muted", 8)
		xaxis = list(range(np.negative(globalPar.maxbpSize), globalPar.maxbpSize + self.resolution, self.step))
		xnew = np.linspace(np.min(xaxis), np.max(xaxis), 1000)
		for i in range(len(self.profile)):
			self.profileTmp = self.profile[i]
			if self.plot:
				sns.set(style="ticks")
				fig, ax = plt.subplots(figsize=[5, 3.5])
			# label = ["Hi-C only: 0.065", "iMARGI only: 0.875", "Common: 2.545"]
			label = ["TAD uniq: 0.014", "TLD uniq: 0.1433", "Common: 0.3115"]
			for j in range(len(self.result)):
				self.foldchangeResult[j, 2 * i], self.foldchangeResult[j, 2 * i + 1] = self.calculate_fold(self.result[j])
				if self.plot:
					func = interpolate.interp1d(xaxis, self.plotprofile / len(self.result[j]), kind='cubic')
					if self.common:
						sns.lineplot(xnew, func(xnew), color=color[j], label=label[j])
					else:
						sns.lineplot(xnew, func(xnew), color=color[j], label="label_{}".format(j))
			# fig.subplots_adjust(right=0.75, top=0.8, bottom=0.2)
			if self.plot:
				plt.ylabel("Number of peaks per {}kb".format(int(self.step / 1000)), fontsize=16)
				plt.xlabel("")
				plt.title("GRID-seq-chr2R", size=20)
				bonus = int(0.5*self.resolution)
				plt.xticks([-400000, -200000, bonus, 200000+self.resolution, 400000+self.resolution],
						   [r"-400kb", r"-200kb", r"boundary", r"200kb", r"400kb"])
				plt.tight_layout()
				plt.savefig(self.output + "/enrichment_profile" + str(i) + "_" + self.chr + ".pdf")
				if self.common:
					fig, ax = plt.subplots(figsize=[5, 3.5])
					color1 = sns.color_palette("Paired_r", len(self.boundB))
					result = []
					for j in range(len(self.boundB)):
						result.append(self.calculate_fold([self.boundB[j]]))
						if self.plot:
							func = interpolate.interp1d(xaxis, self.plotprofile, kind='cubic')
							sns.lineplot(xnew, func(xnew), color=color1[j], label='result' + str(j))

					plt.ylabel("Number of peaks per {}kb".format(int(self.step / 1000)), fontsize=18)
					plt.xlabel("")
					plt.tight_layout()

					plt.savefig(self.output + "/enrichment_profile_boundB" + "_" + self.chr + ".pdf")
					np.savetxt(self.output + "/enrichment_" + self.chr + "boundB_foldchange.txt",
							   np.array(np.concatenate([self.boundB, result], axis=1), dtype=np.str), fmt="%s")
		# print(self.foldchangeResult)
		np.savetxt(self.output + "/enrichment_" + self.chr + "foldchange.txt", self.foldchangeResult, fmt='%.4f')

	def Output(self):
		return self.foldchangeResult

	def read_profile(self, bedFILE):
		binN = (self.end - self.start) / self.step
		profile = np.zeros(int(binN))
		with open(bedFILE, "r") as bed:
			lines = bed.readlines()
			for peak in lines:
				peak = peak.split()
				if peak[0] == self.chr and int(peak[1]) >= self.start and int(peak[2]) <= self.end:
					left = int((int(peak[1]) - self.start) / self.step)
					right = int((int(peak[2]) - self.start) / self.step)
					if left == right:
						profile[left] += 1
					elif right - left == 1:
						length = int(peak[2]) - int(peak[1]) + 1
						left_part = self.start + self.step * right - int(peak[1])
						profile[left] += left_part / length
						profile[right] += 1 - (left_part / length)
					# print("right=left+1, ", peak, left, right, length, left_part)
					else:
						print(peak, "stoped.")
						exit()
				elif peak[0] == self.chr:
					if int(peak[2]) > self.start and int(peak[2]) <= self.end:
						length = int(peak[2]) - int(peak[1]) + 1
						pos = int((int(peak[2]) - self.start) / self.step)
						if pos == 0:
							profile[pos] += (int(peak[2]) - self.start) / length
						elif pos == 1:
							profile[0] += self.step / length
							profile[1] += (int(peak[2]) - self.start - self.step) / length
						else:
							print(peak, "stoped.")
							exit()
					elif int(peak[1]) >= self.start and int(peak[1]) < self.end:
						length = int(peak[2]) - int(peak[1]) + 1
						pos = int((int(peak[2]) - self.start) / self.step)
						if binN == pos + 1:
							profile[pos] += (self.end - int(peak[1])) / length
						elif binN == pos + 2:
							profile[pos] += (self.end - self.step - int(peak[1])) / length
							profile[pos + 1] += self.step / length
						else:
							print(peak, "stoped.")
							exit()
		return profile

	def read_result(self, resultFILE):
		"""
		Derive all the boundaries except the start and end position (allow duplication).
		:param resultFILE:
		:return: list of bounds, bound: [start pos, end pos]
		"""
		start = self.start + globalPar.maxbpSize
		end = self.end - globalPar.maxbpSize
		bound_list = []
		result = np.loadtxt(resultFILE, np.str)
		for i in range(len(result)):
			bound1 = int(result[i][2])
			bound2 = int(result[i][7])
			if bound1 != start and bound1 != end:
				bound_list.append([bound1, bound1 + self.resolution])
			if bound2 != start and bound2 != end:
				bound_list.append([bound2 - self.resolution, bound2])

		return bound_list

	def calculate_fold(self, result):
		self.peak = np.zeros(int(globalPar.peakbinN * self.resolution / self.step))
		self.background = np.zeros(int(globalPar.backgroundbpSize / self.step))
		self.plotprofile = np.zeros(int((2 * globalPar.maxbpSize+self.resolution) / self.step))
		for i in result:
			self.process_tad(i, self.profileTmp)
		peak_value = np.mean(self.peak / len(result))
		background_profile = self.background / (len(result) * 2)
		background_value = np.mean(background_profile)
		foldChange = (peak_value / background_value) - 1  # important!
		# calculate pvalue
		pvalue_of_peak = 1 - norm.cdf(peak_value, loc=background_value, scale=np.std(background_profile))
		if len(result) == 1:
			print(result, foldChange, pvalue_of_peak)
		return foldChange, pvalue_of_peak

	def process_tad(self, bound, profile):
		start_cor = int(((bound[0] - (globalPar.peakbinN - 1)*0.5*self.resolution) - self.start) / self.step)
		back_cor = int(((bound[0] - globalPar.maxbpSize) - self.start) / self.step)
		back_cor_2 = int(((bound[1] + globalPar.maxbpSize - globalPar.backgroundbpSize) - self.start) / self.step)
		for i in range(len(self.plotprofile)):
			self.plotprofile[i] += profile[back_cor + i]
			if i < len(self.background):
				# print(bound, start_cor, back_cor, back_cor_2, i)
				self.background[i] += profile[back_cor + i]
				self.background[i] += profile[back_cor_2 + i]
			if i < len(self.peak):
				self.peak[i] += profile[start_cor + i]

	def deriveCommon(self, resultList):
		"""
		Derive three lists of bounds from two derived lists of TAD bounds (not allow duplication).
		:param resultList: two derived lists of TAD bounds
		:return:
		"""
		boundA = []
		boundB = []
		boundCommon = []
		for b1 in resultList[0]:
			match = False
			for b2 in resultList[1]:
				if abs(b1[0] - b2[0]) / self.resolution <= 0:
					match = True
					# print("b1={}, b2={}".format(b1, b2))
					break
			if match:
				boundCommon.append(b1)
			else:
				boundA.append(b1)
		for b2 in resultList[1]:
			match = False
			for b1 in resultList[0]:
				if abs(b1[0] - b2[0]) / self.resolution <= 0:
					match = True
					# print("b1={}, b2={}".format(b1, b2))
					break
			if not match:
				boundB.append(b2)
		# boundA = [i for i in resultList[0] if i not in resultList[1]]
		# boundB = [i for i in resultList[1] if i not in resultList[0]]
		# boundCommon = [i for i in resultList[0] if i in resultList[1]]
		boundA = self.removedup(boundA)
		boundB = self.removedup(boundB)
		boundCommon = self.removedup(boundCommon)
		print("Specific Bound for Hi-C: ", len(boundA))
		print("Specific Bound for B: ", len(boundB))
		print("Common Bound: ", len(boundCommon))
		for i in boundB:
			if not i in self.boundB:
				self.boundB.append(i)
		return [boundA, boundB, boundCommon]

	def removedup(self, list):
		new_list = []
		for i in list:
			if not i in new_list:
				new_list.append(i)
		return new_list

if __name__ == "__main__":
	main()
