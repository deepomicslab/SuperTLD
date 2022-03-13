# -*- coding: utf-8 -*-
import argparse, os, re
import numpy as np
import matplotlib

matplotlib.use('Agg')
from matplotlib import pyplot as plt
from scipy.stats import pearsonr, spearmanr
import seaborn as sns


def get_pcc(D_0, D_1, effective_lists=None, verbose=False):
	if D_0.shape != D_1.shape:
		raise ValueError("shape doesn't match")
	n = D_0.shape[0]
	D_0_tmp = D_0.tolist()
	D_1_tmp = D_1.tolist()
	# flatten
	d_0 = []
	d_1 = []
	size = 0
	if effective_lists:
		for i in range(n):
			for j in effective_lists[i]:
				d_0.append(D_0_tmp[i][j])
				d_1.append(D_1_tmp[i][j])
				size += 1
	else:
		d_0 = D_0.flatten().tolist()
		d_1 = D_1.flatten().tolist()
		size = D_0.size
	pcc_ref, pvalue = pearsonr(d_0, d_1)
	# print("pcc_ref=%f, pvalue_ref=%f" % (pcc_ref, pvalue))
	return pcc_ref

def calculate_dis_decay(matrix):
	length = len(matrix)
	decay = [0] * (length - 1)
	for i in range(length - 1):
		count = 0
		sum = 0
		for start in range(length - i - 1):
			count += 1
			if matrix[start][start + i + 1] > 0:
				sum += matrix[start][start + i + 1]
			decay[i] = sum * 1.0 / count
	return decay


def calculateBarcodeDecay(matrix, barcode):
	length = len(matrix)
	decay = [0] * (length - 1)
	count = [0] * (length - 1)
	for i in range(len(barcode)):
		for j in range(i + 1, len(barcode)):
			dis = barcode[j] - barcode[i] - 1
			count[dis] += 1
			decay[dis] += matrix[barcode[i] - 1][barcode[j] - 1]
	for i in range(length - 1):
		if count[i] != 0:
			decay[i] = decay[i] / count[i]
	return decay


def acquire_extn(string):
	filename = os.path.basename(string)
	return os.path.splitext(filename)[0]


def acquireBarcode(matrix, file):
	barcodeDecay = []
	with open(file, "r") as barcode:
		lines = barcode.readlines()
		for num in range(len(lines)):
			if (re.match('>', lines[num]) != None):  # ignore remarks
				info = lines[num].split()[1:]
				info = list(map(int, info))
				print("Barcode: {} with {} coverage".format(info, info[-1] - info[0] + 1))
				barcodeDecay.append(calculateBarcodeDecay(matrix, info))
	return barcodeDecay


def main():
	parser = argparse.ArgumentParser()
	parser.add_argument("-i", dest="matrix", nargs="+", required=True)
	parser.add_argument("-o", dest="output", default="./distance_decay")
	parser.add_argument("-b", dest="barcode", help="specific for LingZhao's result")
	parser.add_argument("-l", dest="length", default=80, type=int)
	parser.add_argument("--list", dest="inputList", action="store_true",
						help="if given, the input is a list of matrix files.")
	parser.add_argument("--name", dest="name", default="iMARGI")
	args = parser.parse_args()

	inputList = []
	matrixList = []
	nameList = []
	decay_list = []
	_N = np.inf  # min count
	if args.inputList:
		for line in open(args.matrix[0], "r"):
			line = line.strip("\n")
			inputList.append(line)
	else:
		inputList = args.matrix
	_M = len(inputList)
	print("Below is the pearsonr btw matrix:")
	for i in range(_M):
		matrixTmp = np.loadtxt(inputList[i])
		matrixList.append(matrixTmp)
		nameList.append(acquire_extn(inputList[i]))
		if _N > len(matrixTmp):
			_N = len(matrixTmp)
		if i != 0:
			print(round(get_pcc(matrixList[0][:_N, :_N], matrixList[i][:_N, :_N]), 4))
	print("Below is the pearsonr btw matrix distance decay:")
	for i in range(_M):
		decay_list.append(calculate_dis_decay(matrixList[i]))
		if i != 0:
			pear, pearPvalue = pearsonr(decay_list[0][:_N - 1], decay_list[i][:_N - 1])
			print(round(pear, 4), round(pearPvalue, 4))

	for i in range(_M):
		print("matrix{}: {}".format(i + 1, nameList[i]))
	plt.figure()
	# color = sns.color_palette("Paired_r", 14)
	color = sns.color_palette("muted", 8)
	print(color)
	try:
		barcodeDecay = acquireBarcode(matrixList[0], args.barcode)
		for i in range(len(barcodeDecay)):
			legend = False
			for j in range(args.length):
				if barcodeDecay[i][j] > 0 and legend == False:
					plt.plot(j + 1, barcodeDecay[i][j], c=color[i + 1], marker='x', label="barcode{}".format(i + 1),
							 markersize=3)
					legend = True
				elif barcodeDecay[i][j] > 0:
					plt.plot(j + 1, barcodeDecay[i][j], c=color[i + 1], marker='x', markersize=3)
		print(list(range(1, args.length + 1)))
		print(decay_list[0][:args.length])
		plt.plot(range(1, args.length + 1), decay_list[0][:args.length], color=color[0],
				 label=nameList[0], linewidth=1.5)
	except:
		print("No input of barcode!")
		for i in range(len(nameList)):
			nameList[0] = "Hi-C contact map"
			nameList[1] = "RAI derived contact map"
			plt.plot(range(1, args.length + 1), decay_list[i][:args.length] / decay_list[i][0], color=color[i],
					 label=nameList[i], linewidth=1.5)
	plt.text(40, 0.2, '**Pearson Correlation={}'.format(round(pear, 4)), fontsize=13)
	plt.xlabel("Distance between two bins (100kb)", size=18)
	plt.ylabel("Average interaction frequency", size=18)
	plt.title("Distance decay (iMARGI-chr10)", size=20)
	plt.legend()
	# plt.xticks([0, 10, 20, 30, 40, 50])
	plt.savefig(args.output + ".pdf")


if __name__ == "__main__":
	main()
