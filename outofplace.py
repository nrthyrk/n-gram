from os import listdir
from os.path import isfile, join
from collections import Counter
import math
import logging
import collections

class OutOfPlaceModel:
	'''Character N-Gram Out-of-place Model

	This class implements Character N-Gram Out-of-place Model, and 
	use leave-one-out method to test the accuracy of the model.

	The log file is named oop-gram_n-start-end.log under the program
	directory.
	'''
	def __init__(self, gram_n, start, end):
		self.gram_n = gram_n

		self.logger = logging.getLogger()
		self.logger.setLevel(logging.DEBUG)

		handler = logging.StreamHandler()
		handler.setLevel(logging.INFO)
		formatter = logging.Formatter("%(asctime)s %(message)s")
		handler.setFormatter(formatter)
		self.logger.addHandler(handler)

		handler = logging.FileHandler("oop-" + str(gram_n) + "-" + str(start) + "-" + str(end) + ".log", "w")
		handler.setLevel(logging.DEBUG)
		formatter = logging.Formatter("%(asctime)s %(message)s")
		handler.setFormatter(formatter)
		self.logger.addHandler(handler)

		self.start = start
		self.end = end

	def read_pos_data(self):
		'''Read Positive Data'''
		data = []
		data_path_pos = "data/pos/"
		for file_name in listdir(data_path_pos):
				file_path = join(data_path_pos,file_name)
				if isfile(file_path):
					f = open(file_path)
					text = f.read()
					f.close()
					data.append(text)
		return data

	def read_neg_data(self):
		'''Read Negative Data'''
		data = []
		data_path_neg = "data/neg/"
		for file_name in listdir(data_path_neg):
				file_path = join(data_path_neg,file_name)
				if isfile(file_path):
					f = open(file_path)
					text = f.read()
					f.close()
					data.append(text)
		return data

	def split_test(self, data, idx):
		'''Split the data, get test data'''
		return data[idx]

	def split_training(self, data, idx):
		'''Split the data, get training data'''
		return_data = list(data)
		del return_data[idx]
		return return_data

	def read_freqlist(self, data):
		'''Read frequency list'''
		freqlist = Counter()
		for text in data:
			for i in xrange(len(text) - (self.gram_n - 1)):
				freqlist[text[i: i+self.gram_n]] += 1
		freqlist = freqlist.most_common(len(freqlist))
		ranklist = {}
		for i in xrange(len(freqlist)):
			ranklist[freqlist[i][0]] = i
		return ranklist

	def compute_distance(self, train_freqlist, test_freqlist):
		'''Compute out-of-place distance'''
		distance = 0
		for text in test_freqlist.keys():
			if text not in train_freqlist:
				distance += len(train_freqlist)
			else:
				distance += abs(train_freqlist[text] - test_freqlist[text])
		return distance

	def classify(self, pos_training_data, neg_training_data, test_data, target):
		'''Classify a document'''
		pos_freqlist = self.read_freqlist(pos_training_data)
		neg_freqlist = self.read_freqlist(neg_training_data)
		test_freqlist = self.read_freqlist(test_data)
		
		pos_dist = self.compute_distance(pos_freqlist, test_freqlist)
		neg_dist = self.compute_distance(neg_freqlist, test_freqlist)
		self.logger.info("Distance to postive profile: %i, to negative profile: %i", pos_dist, neg_dist)
		if (pos_dist > neg_dist and target == 0) or (pos_dist < neg_dist and target == 1):
			return 1
		else:
			return 0

	def run(self):
		'''Run this classifier'''
		pos_data = self.read_pos_data()
		neg_data = self.read_neg_data()
		count = 0
		for i in xrange(self.start, self.end):
			if i < 1000:
				self.logger.info("Test #%i, running...", i)
				idx = slice(i, i+1)
				pos_training_data = self.split_training(pos_data, idx)
				neg_training_data = neg_data
				test_data = self.split_test(pos_data, idx)
				correct = self.classify(pos_training_data, neg_training_data, test_data, 1)
				count += correct
				self.logger.info("Correct rate: %i / %i", count, i+1-self.start)
			else:
				self.logger.info("Test #%i, running...", i)
				idx = slice(i-1000, i+1-1000)
				pos_training_data = pos_data
				neg_training_data = self.split_training(neg_data, idx)
				test_data = self.split_test(neg_data, idx)
				correct = self.classify(pos_training_data, neg_training_data, test_data, 0)
				count += correct
				self.logger.info("Correct rate: %i / %i", count, i+1-self.start)