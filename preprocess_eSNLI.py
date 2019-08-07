import numpy as np
import os
import nltk.data
import csv


def remove_file(file):
	try:
		os.remove(file)
	except Exception as e:
		print("\nCouldn't remove " + file + " because ", e, "\n")
		pass


def get_dir(file):
	directory = "."
	f = file.split("/")
	if len(f) == 1:
		return directory
	for i in range(len(f) - 1):
		if i == 0:
			directory = f[i]
		else:
			directory += "/" + f[i]
	return directory, f[-1]


# transform to ascii, ignore unicode
# if more sentences, replace the "." at the end of the first k-1 sentences with ";"
# lowercase
def preproc1_expl(expl_file):
	tokenizer = nltk.data.load('tokenizers/punkt/english.pickle')

	f = open(expl_file)
	content = f.readlines()

	directory, f_name = get_dir(expl_file)
	preproc_expl_file = os.path.join(directory, "preproc1_" + f_name)
	remove_file(preproc_expl_file)
	preproc_expl_f = open(preproc_expl_file, "a")

	i = 0
	for expl in content:
		i += 1
		expl = expl.decode('ascii', 'ignore')
		sentences = tokenizer.tokenize(expl)
		new_expl = ""
		if len(sentences) != 0:
			if len(sentences) > 1:
				for sent in sentences[:-1]:
					if sent[-1] == '.':
						sent = sent[:-1] + " ; "
					elif sent[-3:-1] == '..':
						sent = sent[-3:-1] + " ; "
					if new_expl == "":
						new_expl = sent
					else:
						new_expl += " " + sent
			if new_expl == "":
				new_expl = sentences[-1]
			else:
				new_expl += " " + sentences[-1]
			new_expl = new_expl.lower()
		if i > 1:
			preproc_expl_f.write("\n")
		preproc_expl_f.write(new_expl)

	f.close()
	preproc_expl_f.close()


# truncate to the first max_tokens. I DIDN'T USE IT in the end!!
def truncate_sent(sentence_file, max_tokens):
	f = open(sentence_file)
	content = f.readlines()

	directory, f_name = get_dir(sentence_file)
	preproc_sent_file = os.path.join(directory, "maxtokens_" + str(max_tokens) + "_" + f_name)
	remove_file(preproc_sent_file)
	preproc_sent_f = open(preproc_sent_file, "a")

	i = 0
	count_truncated = 0
	for sent in content:
		i += 1
		tokens = sent.split()
		if len(tokens) > max_tokens:
			count_truncated += 1 
			truncated_sent = ""
			for t in tokens[0:max_tokens]:
				truncated_sent += t + " "
		if i > 1:
			preproc_sent_f.write("\n")
		preproc_sent_f.write(truncated_sent)

	f.close()
	preproc_sent_f.close()
	print "count truncated ", count_truncated


def compute_frequences(expl_files):
	word_frequences = {}
	for expl_file in expl_files:
		f = open(expl_file)
		content = f.readlines()
		for sent in content:
			for word in sent.split():
				if word not in word_frequences:
					word_frequences[word] = 1
				else:
					word_frequences[word] += 1
	return word_frequences


def count_freqs_less_k(word_frequences, k):
	return sum(i < k for i in word_frequences.values())


def words_less_k(word_frequences, k):
	infreq_words = []
	for w, freq in word_frequences.iteritems():
		if freq < k:
			infreq_words.append(w)
	print "number of infreq_words ", len(infreq_words)
	return infreq_words


def replace_infreq(expl_file, word_frequences, k):
	f = open(expl_file)
	content = f.readlines()
	
	directory, f_name = get_dir(expl_file)
	preproc_expl_file = os.path.join(directory, "UNK_freq_" + str(k) + "_" + f_name)
	remove_file(preproc_expl_file)
	preproc_expl_f = open(preproc_expl_file, "a")

	total_words = 0
	replaced_UNK = 0
	words_UNK = []
	distinct_words = []
	i_expl = 0
	for expl in content:
		i_expl += 1
		words = expl.split()
		total_words += len(words)
		new_expl = ""
		i_word = 0
		for word in words:
			i_word += 1
			if word not in distinct_words:
				distinct_words.append(word)
			if ((word not in word_frequences) or (word_frequences[word] < k)):
				if word not in words_UNK:
					words_UNK.append(word)
				word = "<UNK>"
				replaced_UNK += 1
			if i_word == 1:
				new_expl = word
			else:
				new_expl += " " + word
		if i_expl > 1:
			preproc_expl_f.write("\n")
		preproc_expl_f.write(new_expl)

	f.close()
	preproc_expl_f.close()
	print "words_UNK ", words_UNK
	print "total_words ", total_words
	print "total distinct words ", len(distinct_words)
	print "replaced_UNK ", replaced_UNK
	print "number of distinct words infrequent ", len(words_UNK)
	return total_words, replaced_UNK, words_UNK, distinct_words


# create a file with N empty lines as explanations for MultiNLI, N = number of examples in the MultiNLI set
def expl_multinli(fname):
	f = open(fname)
	content = f.readlines()

	gname = "MultiNLI/expl_1.train"
	if 'dev' in fname:
		gname = "MultiNLI/expl_1.dev"
	
	remove_file(gname)
	g = open(gname, 'a')
	for line in content:
		g.write("\n")
	g.close()
	f.close()
	

def concat_files(list_files, out_file):
	count = 0

	remove_file(out_file)
	g = open(out_file, "a")

	for file in list_files:
		f = open(file)
		content = f.readlines()
		for line in content:
			g.write(line)
			count += 1
		f.close()
	g.close()
	print "total lines for ", out_file, count


def prepend_label_expl(label_file, expl_file):
	label_f = open(label_file)
	content_label = label_f.readlines()
	expl_f = open(expl_file)
	content_expl = expl_f.readlines()

	out_file = expl_file.split(".")[0] + "_label." + expl_file.split(".")[1]
	remove_file(out_file)
	g = open(out_file, "a")

	count = 0
	for label_line in content_label:
		expl_line = content_expl[count]
		final_line = label_line.strip() + " " + expl_line.strip()
		g.write(final_line + "\n")
		count += 1

	print count

	g.close()
	label_f.close()
	expl_f.close()


def append_label_expl(label_file, expl_file):
	label_f = open(label_file)
	content_label = label_f.readlines()
	expl_f = open(expl_file)
	content_expl = expl_f.readlines()

	out_file = expl_file.split(".")[0] + "_label_end." + expl_file.split(".")[1]
	remove_file(out_file)
	g = open(out_file, "a")

	count = 0
	for label_line in content_label:
		expl_line = content_expl[count]
		final_line = expl_line.strip() + " " + label_line.strip()
		g.write(final_line + "\n")
		count += 1

	print count

	g.close()
	label_f.close()
	expl_f.close()


def csv_to_txt(csv_file):
	f = open(csv_file)
	reader = csv.DictReader(f)

	out_file = "expl_1.train"
	remove_file(out_file)
	g = open(out_file, 'a')

	for row in reader:
		g.write(row['Explanation_1'] + "\n")

	f.close()
	g.close()


def sentence_lenghts(file):
	f=open(file, 'r')
	lengths = []
	for line in f:
		lengths.append(len(line.split()))

	m = np.mean(lengths)
	stdev = np.std(lengths)
	maxim = max(lengths)

	count_within_3stdevs = 0
	m_3stds = m + 3*stdev
	for l in lengths:
		if l <= m_3stds:
			count_within_3stdevs += 1
			
	print(file, "mean: ", m, "stdev: ", stdev, "max: ", maxim, "count_within_3stdevs: ", count_within_3stdevs, "count_within_3stdevs_%", count_within_3stdevs*100.0 / len(lengths) )
	return m, stdev, maxim

	f.close()

'''
sentence_lenghts("eSNLI/expl_to_inp/attention_nips/s1.train")
sentence_lenghts("eSNLI/expl_to_inp/attention_nips/s2.train")
sentence_lenghts("eSNLI/expl_to_inp/attention_nips/expl.train")
'''

word_frequences = compute_frequences(["eSNLI/expl_to_inp/eSNLI/s2.train"])
replace_infreq("eSNLI/expl_to_inp/eSNLI/s2.train", word_frequences, k=10)
replace_infreq("eSNLI/expl_to_inp/eSNLI/s2.dev", word_frequences, k=10)
replace_infreq("eSNLI/expl_to_inp/eSNLI/s2.test", word_frequences, k=10)


#word_frequences = compute_frequences(["eSNLI/expl_to_inp/attention_nips/s2.train"])
#replace_infreq("eSNLI/expl_to_inp/attention_nips/s2.dev", word_frequences, k=5)
#replace_infreq("eSNLI/expl_to_inp/attention_nips/s2.train", word_frequences, k=5)
#replace_infreq("eSNLI/expl_to_inp/attention_nips/s2.test", word_frequences, k=5)


'''
word_frequences = compute_frequences(["eSNLI/expl_to_inp/attention_nips/s1.train", "eSNLI/expl_to_inp/attention_nips/s2.train"])
print word_frequences
print len(word_frequences.keys())
print count_freqs_less_k(word_frequences, 15)
print count_freqs_less_k(word_frequences, 10)
print count_freqs_less_k(word_frequences, 5)
print count_freqs_less_k(word_frequences, 3)
'''

#csv_to_txt("eSNLI/esnli_train.csv")
#preproc1_expl("eSNLI/expl_1.train")


#prepend_label_expl("eSNLI_ordered/labels.train", "eSNLI_ordered/UNK_freq_15_preproc1__expl_1.train")

'''
prepend_label_expl("eSNLI_ordered/labels.dev", "eSNLI_ordered/UNK_freq_15_preproc1__expl_1.dev")
prepend_label_expl("eSNLI_ordered/labels.dev", "eSNLI_ordered/UNK_freq_15_preproc1__expl_2.dev")
prepend_label_expl("eSNLI_ordered/labels.dev", "eSNLI_ordered/UNK_freq_15_preproc1__expl_3.dev")

prepend_label_expl("eSNLI_ordered/labels.test", "eSNLI_ordered/UNK_freq_15_preproc1__expl_1.test")
prepend_label_expl("eSNLI_ordered/labels.test", "eSNLI_ordered/UNK_freq_15_preproc1__expl_2.test")
prepend_label_expl("eSNLI_ordered/labels.test", "eSNLI_ordered/UNK_freq_15_preproc1__expl_3.test")
'''

'''
append_label_expl("eSNLI_ordered/labels.dev", "eSNLI_ordered/UNK_freq_15_preproc1__expl_1.dev")
append_label_expl("eSNLI_ordered/labels.dev", "eSNLI_ordered/UNK_freq_15_preproc1__expl_2.dev")
append_label_expl("eSNLI_ordered/labels.dev", "eSNLI_ordered/UNK_freq_15_preproc1__expl_3.dev")

append_label_expl("eSNLI_ordered/labels.test", "eSNLI_ordered/UNK_freq_15_preproc1__expl_1.test")
append_label_expl("eSNLI_ordered/labels.test", "eSNLI_ordered/UNK_freq_15_preproc1__expl_2.test")
append_label_expl("eSNLI_ordered/labels.test", "eSNLI_ordered/UNK_freq_15_preproc1__expl_3.test")
'''

'''
append_label_expl("eSNLI/labels.train_esnli", "eSNLI/UNK_freq_15_preproc1_expl_1.train_esnli")

append_label_expl("eSNLI/labels.dev_esnli", "eSNLI/UNK_freq_15_preproc1_expl_1.dev_esnli")
append_label_expl("eSNLI/labels.dev_esnli", "eSNLI/UNK_freq_15_preproc1_expl_2.dev_esnli")
append_label_expl("eSNLI/labels.dev_esnli", "eSNLI/UNK_freq_15_preproc1_expl_3.dev_esnli")

append_label_expl("eSNLI/labels.test_esnli", "eSNLI/UNK_freq_15_preproc1_expl_1.test_esnli")
append_label_expl("eSNLI/labels.test_esnli", "eSNLI/UNK_freq_15_preproc1_expl_2.test_esnli")
append_label_expl("eSNLI/labels.test_esnli", "eSNLI/UNK_freq_15_preproc1_expl_3.test_esnli")

#append_label_expl("ALLeNLI/labels.train", "ALLeNLI/UNK_freq_15_preproc1_maxtokens_40_expl_1.train")
'''

'''
concat_files(["eSNLI/s1.dev", "MultiNLI/s1.dev"], "ALLeNLI/s1.dev")
concat_files(["eSNLI/s2.dev", "MultiNLI/s2.dev"], "ALLeNLI/s2.dev")
concat_files(["eSNLI/labels.dev", "MultiNLI/labels.dev"], "ALLeNLI/labels.dev")
'''

'''
w_freq = compute_frequences(['ALLeNLI/s1.train', 'ALLeNLI/s2.train'])
np.save('word_freq_ALLeNLITrainS1S2.npy', w_freq)

#replace_infreq('ALLeNLI/s1.train', w_freq, k=15)
#replace_infreq('ALLeNLI/s2.train', w_freq, k=15)

replace_infreq('ALLeNLI/s1.dev', w_freq, k=15)
replace_infreq('ALLeNLI/s2.dev', w_freq, k=15)
'''

'''
w_freq = compute_frequences(['eSNLI_ordered/s1.train', 'eSNLI_ordered/s2.train', 'eSNLI_ordered/preproc1__expl_1.train'])
np.save('word_freq_TrainS1S2Expl1.npy', w_freq)

replace_infreq('eSNLI_ordered/s1.dev', w_freq, k=15)
replace_infreq('eSNLI_ordered/s2.dev', w_freq, k=15)
replace_infreq('eSNLI_ordered/preproc1__expl_1.dev', w_freq, k=15)
replace_infreq('eSNLI_ordered/preproc1__expl_2.dev', w_freq, k=15)
replace_infreq('eSNLI_ordered/preproc1__expl_3.dev', w_freq, k=15)

replace_infreq('eSNLI_ordered/s1.test', w_freq, k=15)
replace_infreq('eSNLI_ordered/s2.test', w_freq, k=15)
replace_infreq('eSNLI_ordered/preproc1__expl_1.test', w_freq, k=15)
replace_infreq('eSNLI_ordered/preproc1__expl_2.test', w_freq, k=15)
replace_infreq('eSNLI_ordered/preproc1__expl_3.test', w_freq, k=15)
'''


'''
concat_files(['Comp/test/s1.comp_ml_long', 'Comp/test/s1.comp_ml_short'], 'Comp/test/s1.comp_ml')
concat_files(['Comp/test/s2.comp_ml_long', 'Comp/test/s2.comp_ml_short'], 'Comp/test/s2.comp_ml')
concat_files(['Comp/test/labels.comp_ml_long', 'Comp/test/labels.comp_ml_short'], 'Comp/test/labels.comp_ml')

concat_files(['Comp/test/s1.comp_same_long', 'Comp/test/s1.comp_same_short'], 'Comp/test/s1.comp_same')
concat_files(['Comp/test/s2.comp_same_long', 'Comp/test/s2.comp_same_short'], 'Comp/test/s2.comp_same')
concat_files(['Comp/test/labels.comp_same_long', 'Comp/test/labels.comp_same_short'], 'Comp/test/labels.comp_same')

concat_files(['Comp/test/s1.comp_not_long', 'Comp/test/s1.comp_not_short'], 'Comp/test/s1.comp_not')
concat_files(['Comp/test/s2.comp_not_long', 'Comp/test/s2.comp_not_short'], 'Comp/test/s2.comp_not')
concat_files(['Comp/test/labels.comp_not_long', 'Comp/test/labels.comp_not_short'], 'Comp/test/labels.comp_not')
'''

#concat_files(["eSNLI/s1.train_esnli", "MultiNLI/s1.train"], "ALLeNLI/s1.train")
#concat_files(["eSNLI/s2.train_esnli", "MultiNLI/s2.train"], "ALLeNLI/s2.train")
#concat_files(["eSNLI/UNK_freq_15_preproc1_maxtokens_40_expl_1.train_esnli", "MultiNLI/expl_1.train"], "ALLeNLI/expl_1.train")
#concat_files(["eSNLI/labels.train_esnli", "MultiNLI/labels.train"], "ALLeNLI/labels.train")

#expl_multinli("MultiNLI/s1.train")
#expl_multinli("MultiNLI/s1.dev.matched")
#expl_multinli("MultiNLI/s1.dev.mismatched")


'''
# WRITE FILES WITH <UNK> FOR INFREQ WORDS for premise/hypothesis for the decoder baseline
w_freq = compute_frequences(['eSNLI_ordered/s1.train', 'eSNLI_ordered/s2.train'])
np.save('word_freq_TrainS1S2.npy', w_freq)

files = ["eSNLI_ordered/s1.train", "eSNLI_ordered/s2.train"]
for file in files:
	replace_infreq(file, w_freq, 15)
'''

'''
# COMPUTE FREQUENCES
word_frequences = compute_frequences(["eSNLI/s1.train_esnli", "eSNLI/s2.train_esnli", "eSNLI/preproc1_expl_1.train_esnli"])
file_name = "word_freq_TrainS1S2Expl1.npy"
remove_file(file_name)
np.save(file_name, word_frequences)
'''

'''
# WRITE FILES WITH <UNK> FOR INFREQ WORDS for explanations
file_name = "word_freq_TrainS1S2Expl1.npy"
word_frequences = np.load(file_name).item()
#print word_frequences['bounded']
replace_infreq('eSNLI_ordered/preproc1__expl_1.train', word_frequences, 15)
'''

'''
files = ["eSNLI/preproc1_expl_1.train", "eSNLI/preproc1_expl_1.dev", "eSNLI/preproc1_expl_2.dev", "eSNLI/preproc1_expl_3.dev", "eSNLI/preproc1_expl_1.test", "eSNLI/preproc1_expl_2.test", "eSNLI/preproc1_expl_3.test"]
for file in files:
	replace_infreq(file, word_frequences, 15)
'''

'''
# COMPUTE STATISTICS OF FREQUENCES
#file_name = "word_freq_preproc1_maxtokens40_expl_TrainS1S2Expl.npy"
word_frequences = np.load(file_name).item()
print "total vocab", len(word_frequences.keys())
k = 3
print "number of words less than "  + str(k) + " times", count_freqs_less_k(word_frequences, k)
infreq_words = words_less_k(word_frequences, k)
print infreq_words
print "number of infreq_words", len(infreq_words)
'''


'''
# PREPROC1 : LOWERCASE, REPLACE "." AT THE END OF INTERMEDIATE SENTENCE WITH ";"
expl_files = ["eSNLI_ordered/expl_1.dev", "eSNLI_ordered/expl_2.dev", "eSNLI_ordered/expl_3.dev", "eSNLI_ordered/expl_1.test", "eSNLI_ordered/expl_2.test", "eSNLI_ordered/expl_3.test"]
for expl_f in expl_files:
	preproc1_expl(expl_f, 40)
'''






