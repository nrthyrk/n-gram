from os import listdir
from os.path import isfile, join
from collections import Counter
import math
import logging
import collections

class NaiveBayesModel:
	'''Character N-Gram Naive Bayes Model

	This class implements Character N-Gram Naive Bayes Model, and 
	use leave-one-out method to test the accuracy of the model.

	The log file is named nb-gram_n-start-end.log under the program
	directory.
	'''
	def __init__(self, gram_n, start, end):
		self.gram_n = gram_n

		# init logger
		self.logger = logging.getLogger()
		self.logger.setLevel(logging.DEBUG)

		# stream logger - output to console
		handler = logging.StreamHandler()
		handler.setLevel(logging.INFO)
		formatter = logging.Formatter("%(asctime)s %(message)s")
		handler.setFormatter(formatter)
		self.logger.addHandler(handler)

		# file logger - output to log file
		handler = logging.FileHandler("nb-" + str(gram_n) + "-" + str(start) + "-" + str(end) + ".log", "w")
		handler.setLevel(logging.DEBUG)
		formatter = logging.Formatter("%(asctime)s %(message)s")
		handler.setFormatter(formatter)
		self.logger.addHandler(handler)

		self.start = start
		self.end = end

	def read_pos_data(self):
		'''Read Positive Data'''
		self.pos_training_data = []
		data_path_pos = "data/pos/"
		for file_name in listdir(data_path_pos):
				file_path = join(data_path_pos,file_name)
				if isfile(file_path):
					f = open(file_path)
					text = f.read()
					f.close()
					self.pos_training_data.append(text)

	def read_neg_data(self):
		'''Read Negative Data'''
		self.neg_training_data = []
		data_path_neg = "data/neg/"
		for file_name in listdir(data_path_neg):
				file_path = join(data_path_neg,file_name)
				if isfile(file_path):
					f = open(file_path)
					text = f.read()
					f.close()
					self.neg_training_data.append(text)

	def read_rank(self):
		'''Count the N-grams, and sort them in reverse order according to 
		the number of occurrences.
		'''
		self.pos_rank = []
		self.neg_rank = []

		for text in self.pos_training_data:
			freqlist = Counter()
			for i in xrange(len(text) - (self.gram_n - 1)):
				freqlist[text[i: i+self.gram_n]] += 1
			freqlist = freqlist.most_common(len(freqlist))
			self.pos_rank.append(freqlist)

		for text in self.neg_training_data:
			freqlist = Counter()
			for i in xrange(len(text) - (self.gram_n - 1)):
				freqlist[text[i: i+self.gram_n]] += 1
			freqlist = freqlist.most_common(len(freqlist))
			self.neg_rank.append(freqlist)



	def split_test(self, idx, ispos):
		'''Split the data into training data and test data'''
		self.pos_training_rank = self.pos_rank[:]
		self.neg_training_rank = self.neg_rank[:]

		if ispos == 1:
			self.test_rank = self.pos_training_rank[idx]
			del self.pos_training_rank[idx]
		elif ispos == 0:
			self.test_rank = self.neg_training_rank[idx]
			del self.neg_training_rank[idx]

	def read_freqlist(self):
		'''Compute the feature length list used in Naive Bayes model'''
		self.feature_counts = collections.defaultdict(lambda: 1)
		self.feature_length = collections.defaultdict(lambda: 0)

		for j in xrange(len(self.pos_training_rank)):
			for i in xrange(len(self.pos_training_rank[j])):
				if self.feature_counts[(1, self.pos_training_rank[j][i][0], i)] == 1:
					self.feature_length[self.pos_training_rank[j][i][0]] += 1
				self.feature_counts[(1, self.pos_training_rank[j][i][0], i)] += 1

		for j in xrange(len(self.neg_training_rank)):
			for i in xrange(len(self.neg_training_rank[j])):
				if self.feature_counts[(1, self.neg_training_rank[j][i][0], i)] == 1 and self.feature_counts[(0, self.neg_training_rank[j][i][0], i)] == 1:
					self.feature_length[self.neg_training_rank[j][i][0]] += 1
				self.feature_counts[(0, self.neg_training_rank[j][i][0], i)] += 1

		self.test_ranklist = {}
		for i in xrange(len(self.test_rank[0])):
			self.test_ranklist[self.test_rank[0][i][0]] = i


	def classify(self, target):
		'''Compute the probability, and classify test example'''
		self.read_freqlist()
		pos_prob = 0
		neg_prob = 0
		label_counts = 0
		
		for gram in self.test_ranklist.keys():
			label_counts += self.feature_length[gram]
		for gram in self.test_ranklist.keys():
			pos_prob += math.log(self.feature_counts[(1, gram, self.test_ranklist[gram])] / float(label_counts + len(self.pos_training_rank)))
			neg_prob += math.log(self.feature_counts[(0, gram, self.test_ranklist[gram])] / float(label_counts + len(self.neg_training_rank)))
		
		pos_prob += math.log(len(self.pos_training_rank) / float(len(self.pos_training_rank) + len(self.neg_training_rank)))
		neg_prob += math.log(len(self.neg_training_rank) / float(len(self.pos_training_data) + len(self.neg_training_rank))) 
		self.logger.info("Possibility of postive: %f, of negative: %f", pos_prob, neg_prob)
		if (pos_prob > neg_prob and target == 1) or (pos_prob < neg_prob and target == 0):
			return 1
		else:
			return 0

	def run(self):
		'''run the classifier'''
		self.read_pos_data()
		self.read_neg_data()
		self.read_rank()
		count = 0
		for i in xrange(self.start, self.end):
			if i < 1000: # positive test example
				self.logger.info("Test #%i, running...", i)
				idx = slice(i, i+1)
				self.split_test(idx, 1)
				correct = self.classify(1)
				count += correct
				self.logger.info("Correct rate: %i / %i", count, i+1-self.start)
			else: # negative test example
				self.logger.info("Test #%i, running...", i)
				idx = slice(i-1000, i+1-1000)
				self.split_test(idx, 0)
				correct = self.classify(0)
				count += correct
				self.logger.info("Correct rate: %i / %i", count, i+1-self.start)