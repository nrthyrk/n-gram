import sys
from outofplace import OutOfPlaceModel
from naivebayes import NaiveBayesModel

def main():
	if len(sys.argv) != 5:
		raise Exception("4 arguments are required!")

	# read the arguments
	alg = sys.argv[1]
	gram_n = int(sys.argv[2])
	start = int(sys.argv[3])
	end = int(sys.argv[4])
	if start >= end or start < 0 or end > 2000:
		raise Exception("start must be smaller than end in the range of [0, 2000]!")

	# run the program
	if alg == "oop":
		model = OutOfPlaceModel(gram_n, start, end)
		model.run()
	elif alg == "nb":
		model = NaiveBayesModel(gram_n, start, end)
		model.run()
	else:
		raise Exception("algorithm not supported!")

if __name__ == "__main__":
	main()