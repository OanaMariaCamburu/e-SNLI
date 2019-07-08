import os
import math
import csv
import time
import numpy as np

import matplotlib
matplotlib.use('agg')
import matplotlib.pyplot as plt

import torch
from torch.autograd import Variable

from data_attention_bottom import get_train, get_dev_test_original_expl, get_batch, build_vocab, get_target_expl_batch, get_dev_test_with_expl, get_dev_or_test_without_expl, NLI_DIC_LABELS
import sys

from utils.mutils import get_sentence_from_indices, get_key_from_val, remove_file, assert_sizes, bleu_prediction


GLOVE_PATH = '../dataset/GloVe/glove.840B.300d.txt'


# dataset is the name: dev or test
# visualize is True or False, if True it plots the att weights as an image for each *first* element of the minibatch
def evaluate_snli_final(esnli_net, expl_to_labels_net, criterion_expl, dataset, data, snli_dev_no_unk, snli_test_no_unk, word_vec, word_index, batch_size, print_every, current_run_dir, visualize):
	assert dataset in ['snli_dev', 'snli_test']
	print dataset.upper()
	esnli_net.eval()

	correct = 0.
	cum_test_ppl = 0
	cum_test_n_words = 0

	headers = ["gold_label", "Premise", "Hypothesis", "pred_label", "pred_expl", "Expl_1", "Expl_2", "Expl_3"]
	expl_csv = os.path.join(current_run_dir, time.strftime("%d:%m") + "_" + time.strftime("%H:%M:%S") + "_" + dataset + ".csv")
	remove_file(expl_csv)
	expl_f = open(expl_csv, "a")
	writer = csv.writer(expl_f)
	writer.writerow(headers)

	s1 = data['s1']
	s2 = data['s2']
	expl_1 = data['expl_1']
	expl_2 = data['expl_2']
	expl_3 = data['expl_3']
	label = data['label']


	for i in range(0, len(s1), batch_size):
		#print "\n\n\n i ", i
		# prepare batch
		s1_batch, s1_len = get_batch(s1[i:i + batch_size], word_vec)
		s2_batch, s2_len = get_batch(s2[i:i + batch_size], word_vec)
		s1_batch, s2_batch = Variable(s1_batch.cuda()), Variable(s2_batch.cuda())
		tgt_label_batch = Variable(torch.LongTensor(label[i:i + batch_size])).cuda()
		
		# print example
		if i % print_every == 0:
			print "Final SNLI example from " + dataset
			print "Sentence1:  ", ' '.join(s1[i]), " LENGHT: ", s1_len[0]
			print "Sentence2:  ", ' '.join(s2[i]), " LENGHT: ", s2_len[0]
			print "Gold label:  ", get_key_from_val(label[i], NLI_DIC_LABELS)

		out_lbl = [0, 1, 2, 3]
		for index in range(1, 4):
			expl = eval("expl_" + str(index))
			input_expl_batch, _ = get_batch(expl[i:i + batch_size], word_vec)
			input_expl_batch = Variable(input_expl_batch[:-1].cuda())
			if i % print_every == 0:
				print "Explanation " + str(index) + " :  ", ' '.join(expl[i])
			tgt_expl_batch, lens_tgt_expl = get_target_expl_batch(expl[i:i + batch_size], word_index)
			assert tgt_expl_batch.dim() == 2, "tgt_expl_batch.dim()=" + str(tgt_expl_batch.dim())
			tgt_expl_batch = Variable(tgt_expl_batch).cuda()
			if i % print_every == 0:
				print "Target expl " + str(index) + " :  ", get_sentence_from_indices(word_index, tgt_expl_batch[:, 0]), " LENGHT: ", lens_tgt_expl[0]
			
			
			# model forward, tgt_labels is still None bcs in test mode we get the predicted labels
			out_expl = esnli_net((s1_batch, s1_len), (s2_batch, s2_len), input_expl_batch, mode="teacher", visualize=False)
			# ppl
			loss_expl = criterion_expl(out_expl.view(out_expl.size(0) * out_expl.size(1), -1), tgt_expl_batch.view(tgt_expl_batch.size(0) * tgt_expl_batch.size(1)))
			cum_test_n_words += lens_tgt_expl.sum()
			cum_test_ppl += loss_expl.data[0]
			answer_idx = torch.max(out_expl, 2)[1]
			if i % print_every == 0:
				print "Decoded explanation " + str(index) + " :  ", get_sentence_from_indices(word_index, answer_idx[:, 0])
				print "\n"

		pred_expls = esnli_net((s1_batch, s1_len), (s2_batch, s2_len), input_expl_batch, mode="forloop", visualize=visualize)
		if visualize:
			weights_1 = pred_expls[1]
			weights_2 = pred_expls[2]
			pred_expls = pred_expls[0]

			# plot attention weights
			sentence1_split = s1[i]
			#print "sentence1_split ", sentence1_split
			sentence2_split = s2[i]
			#print "sentence2_split ", sentence2_split
			pred_explanation_split = pred_expls[0].split()
			#print "pred_explanation_split ", pred_explanation_split
			#print " weights_1 ", weights_1.size()
			#print " weights_2 ", weights_2.size()
			#print "weights_1[0:len(sentence1_split) ", weights_1[:, :len(sentence1_split)].size()
			#print "weights_2[0:len(sentence2_split) ", weights_2[:, :len(sentence2_split)].size()
			all_weights = torch.cat([weights_1[:, :len(sentence1_split)], weights_2[:, :len(sentence2_split)]], 1).transpose(1,0) 
			# size: (len_p + len_h) x current_T_dec
			all_weights = all_weights.data.cpu().numpy()
			# yaxis is the concatenation of premise and hypothesis
			y = np.array(range(len(sentence1_split) + len(sentence2_split)))
			#print "len(sentence1_split) + len(sentence2_split) ", len(sentence1_split) + len(sentence2_split)
			my_yticks = np.append(sentence1_split, sentence2_split)
			#print "my_yticks ", my_yticks
			# x axis is the pred expl
			x = np.array(range(len(pred_explanation_split)))
			#print "len(pred_explanation_split) ", len(pred_explanation_split)
			my_xticks = pred_explanation_split
			plt.xticks(x, my_xticks)
			plt.xticks(rotation=90)
			plt.yticks(y, my_yticks)
			plt.imshow(all_weights, cmap="gray", vmin=0, vmax=1)
			plt.savefig(os.path.join(current_run_dir, time.strftime("%d:%m") + "_" + time.strftime("%H:%M:%S") + "_att_" + str(i) + ".png"), dpi=1000)
			#plt.show()

		if i % print_every == 0:
			print "Fully decoded explanation: ", pred_expls[0]

		pred_expls_with_sos = np.array([['<s>'] + [word for word in sent.split()] + ['</s>'] for sent in pred_expls])
		pred_expl_batch, pred_expl_len = get_batch(pred_expls_with_sos, word_vec)
		pred_expl_batch = Variable(pred_expl_batch.cuda())

		out_lbl = expl_to_labels_net((pred_expl_batch, pred_expl_len))

		# accuracy
		pred = out_lbl.data.max(1)[1]
		if i % print_every == 0:
			print "Predicted label:  ", get_key_from_val(pred[0], NLI_DIC_LABELS), "\n\n\n"
		correct += pred.long().eq(tgt_label_batch.data.long()).cpu().sum()

		# write csv row of predictions
		# headers = ["gold_label", "Premise", "Hypothesis", "pred_label", "pred_expl", "Expl_1", "Expl_2", "Expl_3"]
		for j in range(len(pred_expls)):
			row = []
			row.append(get_key_from_val(label[i+j], NLI_DIC_LABELS))
			row.append(' '.join(s1[i+j][1:-1]))
			row.append(' '.join(s2[i+j][1:-1]))
			row.append(get_key_from_val(pred[j], NLI_DIC_LABELS))
			row.append(pred_expls[j])
			row.append(' '.join(expl_1[i+j][1:-1]))
			row.append(' '.join(expl_2[i+j][1:-1]))
			row.append(' '.join(expl_3[i+j][1:-1]))
			writer.writerow(row)
		
	expl_f.close()
	eval_acc = round(100 * correct / len(s1), 2)
	eval_ppl = math.exp(cum_test_ppl/cum_test_n_words)

	if dataset == 'snli_dev':
		bleu_score = bleu_prediction(expl_csv, snli_dev_no_unk)
	else:
		bleu_score = bleu_prediction(expl_csv, snli_test_no_unk)

	bleu_score = 100 * bleu_score
	print dataset.upper() + ' SNLI accuracy: ', eval_acc, 'bleu score: ', bleu_score, 'ppl: ', eval_ppl
	return eval_acc, round(bleu_score, 2), round(eval_ppl, 2)


def eval_datasets_without_expl(esnli_net, expl_to_labels_net, which_set, data, word_vec, word_vec_expl, word_emb_dim, batch_size, print_every, current_run_dir):

	dict_labels = NLI_DIC_LABELS
	
	esnli_net.eval()
	correct = 0.

	s1 = data['s1']
	s2 = data['s2']
	label = data['label']

	headers = ["gold_label", "Premise", "Hypothesis", "pred_label", "pred_expl"]
	expl_csv = os.path.join(current_run_dir, time.strftime("%d:%m") + "_" + time.strftime("%H:%M:%S") + "_" + which_set +".csv")
	remove_file(expl_csv)
	expl_f = open(expl_csv, "a")
	writer = csv.writer(expl_f)
	writer.writerow(headers)

	for i in range(0, len(s1), batch_size):
		# prepare batch
		s1_batch, s1_len = get_batch(s1[i:i + batch_size], word_vec)
		s2_batch, s2_len = get_batch(s2[i:i + batch_size], word_vec)

		current_bs = s1_batch.size(1)
		assert_sizes(s1_batch, 3, [s1_batch.size(0), current_bs, word_emb_dim])
		assert_sizes(s2_batch, 3, [s2_batch.size(0), current_bs, word_emb_dim])
		
		s1_batch, s2_batch = Variable(s1_batch.cuda()), Variable(s2_batch.cuda())
		tgt_label_batch = Variable(torch.LongTensor(label[i:i + batch_size])).cuda()
			
		expl_t0 = Variable(torch.from_numpy(word_vec['<s>']).float().unsqueeze(0).expand(current_bs, word_emb_dim).unsqueeze(0)).cuda()
		assert_sizes(expl_t0, 3, [1, current_bs, word_emb_dim])

		# model forward
		pred_expls = esnli_net((s1_batch, s1_len), (s2_batch, s2_len), expl_t0, mode="forloop", visualize=False)

		pred_expls_with_sos = np.array([['<s>'] + [word for word in sent.split()] + ['</s>'] for sent in pred_expls])
		pred_expl_batch, pred_expl_len = get_batch(pred_expls_with_sos, word_vec_expl)
		pred_expl_batch = Variable(pred_expl_batch.cuda())

		out_lbl = expl_to_labels_net((pred_expl_batch, pred_expl_len))

		# accuracy
		pred = out_lbl.data.max(1)[1]
		correct += pred.long().eq(tgt_label_batch.data.long()).cpu().sum()

		# write csv row of predictions
		# Look up for the headers order
		for j in range(len(pred_expls)):
			row = []
			row.append(get_key_from_val(label[i+j], dict_labels))
			row.append(' '.join(s1[i+j][1:-1]))
			row.append(' '.join(s2[i+j][1:-1]))
			row.append(get_key_from_val(pred[j], dict_labels))
			row.append(pred_expls[j])
			writer.writerow(row)

		# print example
		if i % print_every == 0:
			print which_set.upper() + " example: "
			print "Premise:  ", ' '.join(s1[i]), " LENGHT: ", s1_len[0]
			print "Hypothesis:  ", ' '.join(s2[i]), " LENGHT: ", s2_len[0]
			print "Gold label:  ", get_key_from_val(label[i], dict_labels)
			print "Predicted label:  ", get_key_from_val(pred[0], dict_labels)
			print "Predicted explanation:  ", pred_expls[0], "\n\n\n"

	eval_acc = round(100 * correct / len(s1), 2)
	print which_set.upper() + " no train ", eval_acc, '\n\n\n'
	expl_f.close()
	return eval_acc


def eval_all(esnli_net, expl_to_labels_net, criterion_expl, params):
	word_index = params.word_index 
	word_emb_dim = params.word_emb_dim 
	batch_size = params.eval_batch_size
	print_every = params.print_every 
	current_run_dir = params.current_run_dir

	snli_test_no_unk = get_dev_test_original_expl(params.esnli_path, 'test')
	snli_dev_no_unk = get_dev_test_original_expl(params.esnli_path, 'dev')
	
	esnli_net.eval()
	
	# save auxiliary tasks results at each epoch in a csv file
	dev_csv = os.path.join(current_run_dir, time.strftime("%d:%m") + "_" + time.strftime("%H:%M:%S") + "_" + "artifacts.csv")
	remove_file(dev_csv)
	dev_f = open(dev_csv, "a")
	writer = csv.writer(dev_f)

	headers = []
	headers.append('set')
	row_dev = ['dev']
	row_test = ['test']
	
	preproc = params.preproc_expl + "_"
	snli_train = get_train(params.esnli_path, preproc, params.min_freq, params.n_train)
	snli_dev = get_dev_test_with_expl(params.esnli_path, 'dev', preproc, params.min_freq)
	snli_test = get_dev_test_with_expl(params.esnli_path, 'test', preproc, params.min_freq)
	snli_sentences = snli_train['s1'] + snli_train['s2'] + snli_train['expl_1'] + snli_dev['s1'] + snli_dev['s2'] + snli_dev['expl_1'] + snli_dev['expl_2'] + snli_dev['expl_3'] + snli_test['s1'] + snli_test['s2'] + snli_test['expl_1'] + snli_test['expl_2'] + snli_test['expl_3']
	word_vec = build_vocab(snli_sentences, GLOVE_PATH)
	for split in ['s1', 's2', 'expl_1', 'expl_2', 'expl_3']:
		for data_type in ['snli_dev', 'snli_test']:
			eval(data_type)[split] = np.array([['<s>'] +
				[word for word in sent.split() if word in word_vec] +
				['</s>'] for sent in eval(data_type)[split]])


	
	# SNLI
	test_acc, test_bleu_score, test_ppl = evaluate_snli_final(esnli_net, expl_to_labels_net, criterion_expl, 'snli_test', snli_test, snli_dev_no_unk, snli_test_no_unk, word_vec, word_index, batch_size, print_every, current_run_dir, visualize=True)
	#final_dev_acc, dev_bleu_score, final_dev_ppl = evaluate_snli_final(esnli_net, expl_to_labels_net, criterion_expl, 'snli_dev', snli_dev, snli_dev_no_unk, snli_test_no_unk, word_vec, word_index, batch_size, print_every, current_run_dir, visualize=False)
	
	final_dev_acc, dev_bleu_score, final_dev_ppl = 0, 0, 0
	headers.append('SNLI-acc')
	row_dev.append(final_dev_acc)
	row_test.append(test_acc)

	headers.append('SNLI-ppl')
	row_dev.append(final_dev_ppl)
	row_test.append(test_ppl)

	headers.append('SNLI-BLEU')
	row_dev.append(dev_bleu_score)
	row_test.append(test_bleu_score)


	writer.writerow(headers)
	writer.writerow(row_dev)
	writer.writerow(row_test)
	dev_f.close()


