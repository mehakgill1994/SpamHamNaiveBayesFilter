import os
import re
import math

class Classifier:

	DATASET_PATH = "./data"
	TEST_DATASET_PATH = DATASET_PATH + "/test"
	TRAIN_DATASET_PATH = DATASET_PATH + "/train"

	SMOOTHING_DELTA = 1.5

	def __init__(self):
		self.vocabulary = []
		self.spam_vocabulary_frequencies = {}
		self.ham_vocabulary_frequencies = {}

		self.spam_vocabulary_probs = {}
		self.ham_vocabulary_probs = {}

	def build_model(self):
		print("building model...")
		all_training_file_names = os.listdir(Classifier.TRAIN_DATASET_PATH)
		all_training_file_names.sort()

		for file_name in all_training_file_names:
			file = open(Classifier.TRAIN_DATASET_PATH+"/"+file_name, encoding="latin-1")
			lines = file.readlines()
			vocabulary_frequencies = self.spam_vocabulary_frequencies if 'spam' in file_name else self.ham_vocabulary_frequencies
			
			for line in lines:
				words_list = re.split('[^a-zA-Z]',line.lower())
				# remove empty strings
				words_list = [word for word in words_list if word]
				# populate vocabulary
				for word in words_list:
					# push data in vocabulary
					self.vocabulary.append(word)
					
					if word in vocabulary_frequencies:
						vocabulary_frequencies[word] += 1
					else:
						vocabulary_frequencies[word] = 1
			file.close()

		self.vocabulary = list(set(self.vocabulary))

	def add_smoothing(self, smoothing_value=SMOOTHING_DELTA):
		print("adding smoothing")
		spam_words = self.spam_vocabulary_frequencies.keys()
		ham_words = self.ham_vocabulary_frequencies.keys()

		for word in self.vocabulary:
			if word not in spam_words:
				self.spam_vocabulary_frequencies[word] = smoothing_value
			else:
				self.spam_vocabulary_frequencies[word] += smoothing_value

			if word not in ham_words:
				self.ham_vocabulary_frequencies[word] = smoothing_value
			else:
				self.ham_vocabulary_frequencies[word] += smoothing_value

	def write_model_data(self, output_file_name, vocabulary, spam_total_word_count=None, ham_total_word_count=None):
		file = open(output_file_name, "w")
		print("Writing to %s" % output_file_name)
		spam_total_words = 0
		ham_total_words = 0
		spam_vocabulary_probs = {}
		ham_vocabulary_probs = {}

		if spam_total_word_count is None:
			spam_total_words = sum(self.spam_vocabulary_frequencies.values())
			ham_total_words = sum(self.ham_vocabulary_frequencies.values())	
		else:
			spam_total_words = spam_total_word_count
			ham_total_words = ham_total_word_count
		
		for index, word in enumerate(sorted(vocabulary)):
			ham_vocabulary_probs[word] = self.ham_vocabulary_frequencies[word]/ham_total_words
			spam_vocabulary_probs[word] =self.spam_vocabulary_frequencies[word]/spam_total_words

			index = int(index) + 1
			if index != 1:
				file.write("\n")
			file.write("%s  " % index)
			file.write(word + '  ')
			file.write("%s  " % (int(self.ham_vocabulary_frequencies[word] - Classifier.SMOOTHING_DELTA)))
			file.write("%s  " % ham_vocabulary_probs[word])
			file.write("%s  " % (int(self.spam_vocabulary_frequencies[word] - Classifier.SMOOTHING_DELTA)))
			file.write("%s" % spam_vocabulary_probs[word])

		file.close()
		return spam_vocabulary_probs, ham_vocabulary_probs

	def test_model(self, output_file_name, spam_prob, ham_prob):
		file_to_write = open(output_file_name, "w")
		print("Writing to %s" % output_file_name)
		all_training_file_names = os.listdir(Classifier.TEST_DATASET_PATH)
		all_training_file_names.sort()
		classified_wrong = 0
		ham_classified_wrong = 0
		spam_classified_wrong = 0

		for index, file_name in enumerate(all_training_file_names):
			file = open(Classifier.TEST_DATASET_PATH+"/"+file_name, encoding="latin-1")
			lines = file.readlines()

			spam_score = 0
			ham_score = 0
			
			total_words = []
			
			for line in lines:
				words_list = re.split('[^a-zA-Z]',line.lower())
				# remove empty strings
				words_list = [word for word in words_list if word]
				total_words.extend(words_list)

			for word in total_words:
				#TODO: what to do when the word is not in train data?
				if(word in spam_prob.keys()):
					if spam_prob[word] != 0:
						spam_score += math.log(spam_prob[word])
					
					if ham_prob[word] != 0:	
						ham_score += math.log(ham_prob[word])
					
			index = int(index) + 1
			if index != 1:
				file_to_write.write("\n")

			file_to_write.write("%s  " % index)
			file_to_write.write("%s  " % file_name)
			if spam_score >= ham_score:
				file_to_write.write("spam  ")
			else:
				file_to_write.write("ham  ")
			file_to_write.write("%s  " % ham_score)
			file_to_write.write("%s  " % spam_score)
			if 'ham' in file_name:
				correct_class = "ham"
				file_to_write.write("ham  ")
			else:
				correct_class = "spam"
				file_to_write.write("spam  ")

			if correct_class == "spam" and spam_score >= ham_score:
				file_to_write.write("right")
			elif correct_class == "ham" and ham_score > spam_score:
				file_to_write.write("right")
			else:
				if correct_class == "spam":
					spam_classified_wrong+=1
				if correct_class == "ham":
					ham_classified_wrong+=1

				file_to_write.write("wrong")
				classified_wrong+=1

			file.close()

		print('Classified Wrong in %s = %d' % (output_file_name, classified_wrong))
		print('Classified Ham Wrong in %s = %d' % (output_file_name, ham_classified_wrong))
		print('Classified spam Wrong in %s = %d' % (output_file_name, spam_classified_wrong))
		file_to_write.close()

	def experiment2_stop_words(self):
		stop_word_vocabulary = self.vocabulary[:]
		spam_total_words = sum(self.spam_vocabulary_frequencies.values())
		ham_total_words = sum(self.ham_vocabulary_frequencies.values())
		spam_vocabulary_probs = {}
		ham_vocabulary_probs = {}

		#build model
		file_input = open(Classifier.DATASET_PATH+"/stopWords.txt", encoding="latin-1")
		lines = file_input.readlines()
		stop_words = []
		for line in lines:
				words_list = re.split('[^a-zA-Z]',line.lower())
				# remove empty strings
				words_list = [word for word in words_list if word]
				# populate vocabulary
				for word in words_list:
					# push data in vocabulary
					stop_words.append(word)

		for word in stop_words:
			if word in stop_word_vocabulary:
				stop_word_vocabulary.remove(word)
				spam_total_words-=self.spam_vocabulary_frequencies[word]
				ham_total_words-=self.ham_vocabulary_frequencies[word]
		
		file_input.close()
			
		#write model
		spam_vocabulary_probs, ham_vocabulary_probs = self.write_model_data('stopword-model.txt', stop_word_vocabulary, spam_total_words, ham_total_words)
		
		#test
		self.test_model('stopword-result.txt', spam_vocabulary_probs, ham_vocabulary_probs)
	
	def experiment3_length_filtering(self):
		length_filtered_vocabulary = self.vocabulary[:]
		spam_total_words = sum(self.spam_vocabulary_frequencies.values())
		ham_total_words = sum(self.ham_vocabulary_frequencies.values())
		spam_vocabulary_probs = {}
		ham_vocabulary_probs = {}

		#build model
		for word in self.vocabulary:
			if len(word) <= 2 or len(word) >= 9:
				length_filtered_vocabulary.remove(word)
				spam_total_words-=self.spam_vocabulary_frequencies[word]
				ham_total_words-=self.ham_vocabulary_frequencies[word]
				
		#write model
		spam_vocabulary_probs, ham_vocabulary_probs = self.write_model_data('wordlength-model.txt', length_filtered_vocabulary, spam_total_words, ham_total_words)
		
		#test
		self.test_model('wordlength-result.txt', spam_vocabulary_probs, ham_vocabulary_probs)

	def experiment4_frequency_filtering(self, file_name, lower_cutoff_frequency, higher_cutoff_frequency):
		frequency_filtered_vocabulary = self.vocabulary[:]
		spam_total_words = sum(self.spam_vocabulary_frequencies.values())
		ham_total_words = sum(self.ham_vocabulary_frequencies.values())
		spam_vocabulary_probs = {}
		ham_vocabulary_probs = {}

		#build model
		for word in self.vocabulary:
			word_frequency = self.spam_vocabulary_frequencies[word] + self.ham_vocabulary_frequencies[word]
			#subtract smoothing frequency before checking the threshold
			word_frequency-=(2*Classifier.SMOOTHING_DELTA)
			if lower_cutoff_frequency <= word_frequency and word_frequency <= higher_cutoff_frequency:
				frequency_filtered_vocabulary.remove(word)
				spam_total_words-=self.spam_vocabulary_frequencies[word]
				ham_total_words-=self.ham_vocabulary_frequencies[word]

		print("Vocab size: %d" % len(frequency_filtered_vocabulary))
				
		#write model
		spam_vocabulary_probs, ham_vocabulary_probs = self.write_model_data(file_name + '-model.txt', frequency_filtered_vocabulary, spam_total_words, ham_total_words)
		
		#test
		self.test_model(file_name + '-result.txt', spam_vocabulary_probs, ham_vocabulary_probs)

	def experiment4_most_frequent_filtering(self, file_name, frequency_percent):
		frequency_filtered_vocabulary = self.vocabulary[:]
		spam_total_words = sum(self.spam_vocabulary_frequencies.values())
		ham_total_words = sum(self.ham_vocabulary_frequencies.values())
		spam_vocabulary_probs = {}
		ham_vocabulary_probs = {}

		vocabulary_size = len(frequency_filtered_vocabulary)
		number_of_words_to_remove = (frequency_percent/100) * vocabulary_size

		vocabulary_dictonary = {}

		#build model
		for word in self.vocabulary:
			word_frequency = self.spam_vocabulary_frequencies[word] + self.ham_vocabulary_frequencies[word]
			vocabulary_dictonary[word] = word_frequency

		vocabulary_dictonary = sorted(vocabulary_dictonary.items(), key=lambda kv: kv[1])
		vocabulary_dictonary.reverse()
		vocabulary_dictonary = dict(vocabulary_dictonary)
			
		for index, word in enumerate(vocabulary_dictonary):
			if index < number_of_words_to_remove:
				frequency_filtered_vocabulary.remove(word)
				spam_total_words-=self.spam_vocabulary_frequencies[word]
				ham_total_words-=self.ham_vocabulary_frequencies[word]
			else:
				break

		print("Vocab size: %d" % len(frequency_filtered_vocabulary))
				
		#write model
		spam_vocabulary_probs, ham_vocabulary_probs = self.write_model_data(file_name + '-model.txt', frequency_filtered_vocabulary, spam_total_words, ham_total_words)
		
		#test
		self.test_model(file_name + '-result.txt', spam_vocabulary_probs, ham_vocabulary_probs)


