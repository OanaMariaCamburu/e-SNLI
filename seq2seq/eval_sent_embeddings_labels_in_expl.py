import os
import math
import csv
import time
import numpy as np

import torch
from torch.autograd import Variable

from data_label_in_expl import get_dev_test_original_expl, get_batch, build_vocab, get_target_expl_batch, get_dev_test_with_expl, get_dev_or_test_without_expl, NLI_DIC_LABELS, NLI_LABELS_TO_NLI

import sys
#sys.path.append("..")
sys.path.append("/raid/data/oanuru/infer_sent_esnli/utils")
from mutils import get_sentence_from_indices, get_key_from_val, remove_file, assert_sizes, bleu_prediction
sys.path.append("/raid/data/oanuru/infer_sent_esnli")
import senteval

PATH_TO_DATA = '../data/senteval_data/'
GLOVE_PATH = '../dataset/GloVe/glove.840B.300d.txt'


# dataset is the name: dev or test
def evaluate_snli_final(esnli_net, criterion_expl, dataset, data, expl_no_unk, word_vec, word_index, batch_size, print_every, current_run_dir):
	assert dataset in ['snli_dev', 'snli_test']
	print dataset.upper()
	esnli_net.eval()

	correct = 0.
	correct_labels_expl = 0.
	cum_test_ppl = 0
	cum_test_n_words = 0

	headers = ["gold_label", "Premise", "Hypothesis", "pred_label", "pred_expl", "pred_lbl_decoder", "Expl_1", "Expl_2", "Expl_3"]
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
	label_expl = data['label_expl']

	for i in range(0, len(s1), batch_size):
		# prepare batch
		s1_batch, s1_len = get_batch(s1[i:i + batch_size], word_vec)
		s2_batch, s2_len = get_batch(s2[i:i + batch_size], word_vec)
		s1_batch, s2_batch = Variable(s1_batch.cuda()), Variable(s2_batch.cuda())
		tgt_label_batch = Variable(torch.LongTensor(label[i:i + batch_size])).cuda()
		tgt_label_expl_batch = label_expl[i:i + batch_size]
		
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
				print "Predicted label by decoder " + str(index) + " :  ", ' '.join(expl[i][0])
			tgt_expl_batch, lens_tgt_expl = get_target_expl_batch(expl[i:i + batch_size], word_index)
			assert tgt_expl_batch.dim() == 2, "tgt_expl_batch.dim()=" + str(tgt_expl_batch.dim())
			tgt_expl_batch = Variable(tgt_expl_batch).cuda()
			if i % print_every == 0:
				print "Target expl " + str(index) + " :  ", get_sentence_from_indices(word_index, tgt_expl_batch[:, 0]), " LENGHT: ", lens_tgt_expl[0]
			
			# model forward, tgt_labels is still None bcs in test mode we get the predicted labels
			out_expl, out_lbl[index-1] = esnli_net((s1_batch, s1_len), (s2_batch, s2_len), input_expl_batch, mode="teacher")
			# ppl
			loss_expl = criterion_expl(out_expl.view(out_expl.size(0) * out_expl.size(1), -1), tgt_expl_batch.view(tgt_expl_batch.size(0) * tgt_expl_batch.size(1)))
			cum_test_n_words += lens_tgt_expl.sum()
			cum_test_ppl += loss_expl.data[0]
			answer_idx = torch.max(out_expl, 2)[1]
			if i % print_every == 0:
				print "Decoded explanation " + str(index) + " :  ", get_sentence_from_indices(word_index, answer_idx[:, 0])
				print "\n"

		pred_expls, out_lbl[3] = esnli_net((s1_batch, s1_len), (s2_batch, s2_len), input_expl_batch, mode="forloop")
		if i % print_every == 0:
			print "Fully decoded explanation: ", pred_expls[0].strip().split()[1:-1]
			print "Predicted label from decoder: ", pred_expls[0].strip().split()[0]

		for b in range(len(pred_expls)):
			assert tgt_label_expl_batch[b] in ['entailment', 'neutral', 'contradiction']
			if len(pred_expls[b]) > 0:
				words = pred_expls[b].strip().split()
				assert words[0] in ['entailment', 'neutral', 'contradiction'], words[0]
				if words[0] == tgt_label_expl_batch[b]:
					correct_labels_expl += 1

		assert(torch.equal(out_lbl[0], out_lbl[1]))
		assert(torch.equal(out_lbl[1], out_lbl[2]))
		assert(torch.equal(out_lbl[2], out_lbl[3]))
		
		# accuracy
		pred = out_lbl[0].data.max(1)[1]
		if i % print_every == 0:
			print "Predicted label from classifier:  ", get_key_from_val(pred[0], NLI_DIC_LABELS), "\n\n\n"
		correct += pred.long().eq(tgt_label_batch.data.long()).cpu().sum()

		# write csv row of predictions
		for j in range(len(pred_expls)):
			row = []
			row.append(get_key_from_val(label[i+j], NLI_DIC_LABELS))
			row.append(' '.join(s1[i+j][1:-1]))
			row.append(' '.join(s2[i+j][1:-1]))
			row.append(get_key_from_val(pred[j], NLI_DIC_LABELS))
			row.append(' '.join(pred_expls[j].strip().split()[1:-1]))
			assert pred_expls[j].strip().split()[0] in ['entailment', 'contradiction', 'neutral'], pred_expls[j].strip().split()[0]
			row.append(pred_expls[j].strip().split()[0])
			#row.append(' '.join(expl_1[i+j][2:-1]))
			#row.append(' '.join(expl_2[i+j][2:-1]))
			#row.append(' '.join(expl_3[i+j][2:-1]))
			row.append(expl_no_unk['expl_1'][i+j])
			row.append(expl_no_unk['expl_2'][i+j])
			row.append(expl_no_unk['expl_3'][i+j])
			writer.writerow(row)
		
	eval_acc = round(100 * correct / len(s1), 2)
	eval_acc_label_expl = round(100 * correct_labels_expl/len(s1), 2)
	eval_ppl = math.exp(cum_test_ppl/cum_test_n_words)

	expl_f.close()
	bleu_score = 100 * bleu_prediction(expl_csv, expl_no_unk)

	print dataset.upper() + ' SNLI accuracy: ', eval_acc, 'bleu score: ', bleu_score, 'ppl: ', eval_ppl, 'eval_acc_label_expl: ', eval_acc_label_expl
	return eval_acc, round(bleu_score, 2), round(eval_ppl, 2), eval_acc_label_expl


def eval_datasets_without_expl(esnli_net, which_set, data, word_vec, word_emb_dim, batch_size, print_every, current_run_dir):

	dict_labels = NLI_DIC_LABELS

	esnli_net.eval()
	correct = 0.
	correct_labels_expl = 0.

	s1 = data['s1']
	s2 = data['s2']
	label = data['label']
	label_expl = data['label_expl']

	headers = ["gold_label", "Premise", "Hypothesis", "pred_label", "pred_expl", "pred_lbl_decoder"]
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
		tgt_label_expl_batch = label_expl[i:i + batch_size]
			
		expl_t0 = Variable(torch.from_numpy(word_vec['<s>']).float().unsqueeze(0).expand(current_bs, word_emb_dim).unsqueeze(0)).cuda()
		assert_sizes(expl_t0, 3, [1, current_bs, word_emb_dim])

		# model forward
		pred_expls, out_lbl = esnli_net((s1_batch, s1_len), (s2_batch, s2_len), expl_t0, mode="forloop")
		assert len(pred_expls) == current_bs, "pred_expls: " + str(len(pred_expls)) + " current_bs: " + str(current_bs)
		
		for b in range(len(pred_expls)):
			assert tgt_label_expl_batch[b] in ['entailment', 'neutral', 'contradiction']
			if len(pred_expls[b]) > 0:
				words = pred_expls[b].strip().split(" ")
				if words[0] == tgt_label_expl_batch[b]:
					correct_labels_expl += 1

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
			row.append(pred_expls[j][1:-1])
			row.append(pred_expls[j][0])
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
	eval_acc_label_expl = round(100 * correct_labels_expl / len(s1), 2)
	print which_set.upper() + " no train ", eval_acc, '\n\n\n'
	expl_f.close()
	return eval_acc, eval_acc_label_expl



def eval_all(esnli_net, criterion_expl, params):
	word_index = params.word_index 
	word_emb_dim = params.word_emb_dim 
	batch_size = params.eval_batch_size
	print_every = params.print_every 
	current_run_dir = params.current_run_dir
	train_snli_classif = params.train_snli_classif 
	use_prototype_senteval = params.use_prototype_senteval

	esnli_net.eval()

	transfer_tasks = ['MR', 'CR', 'SUBJ', 'MPQA', 'SST2', 'TREC', 'MRPC', 'SICKEntailment', 'SICKRelatedness', 'STS14', 'STSBenchmark']
	if params.do_image_caption:
		transfer_tasks.append('ImageCaptionRetrieval')

	accuracy_tasks = ['MR', 'CR', 'SUBJ', 'MPQA', 'SST2', 'TREC', 'MRPC', 'SICKEntailment']

	infersent_allnli = { 'MR':81.1, 'CR':86.3, 'SUBJ':92.4, 'MPQA':90.2, 'SST2':84.6, 'TREC':88.2, 'MRPC_acc':76.2, 'MRPC_f1':83.1, 'SICKRelatedness':0.884, 'SICKEntailment':86.3, 'STSB_pearson':75.8, 'STSB_spearman':75.5}

	# save auxiliary tasks results at each epoch in a csv file
	dev_csv = os.path.join(current_run_dir, time.strftime("%d:%m") + "_" + time.strftime("%H:%M:%S") + "_" + "aux_tasks.csv")
	remove_file(dev_csv)
	dev_f = open(dev_csv, "a")
	writer = csv.writer(dev_f)

	headers = []
	headers.append('set')
	row_dev = ['dev']
	row_test = ['test']

	
	# SNLI
	expl_no_unk_dev = get_dev_test_original_expl(params.esnli_path, 'dev')
	expl_no_unk_test = get_dev_test_original_expl(params.esnli_path, 'test')

	preproc = params.preproc_expl + "_"
	snli_dev = get_dev_test_with_expl(params.esnli_path, 'dev', preproc, params.min_freq)
	snli_test = get_dev_test_with_expl(params.esnli_path, 'test', preproc, params.min_freq)
	snli_sentences = snli_dev['s1'] + snli_dev['s2'] + snli_dev['expl_1'] + snli_dev['expl_2'] + snli_dev['expl_3'] + snli_test['s1'] + snli_test['s2'] + snli_test['expl_1'] + snli_test['expl_2'] + snli_test['expl_3']
	word_vec = build_vocab(snli_sentences, GLOVE_PATH)

	for split in ['s1', 's2', 'expl_1', 'expl_2', 'expl_3']:
		for data_type in ['snli_dev', 'snli_test']:
			eval(data_type)[split] = np.array([['<s>'] +
				[word for word in sent.split() if word in word_vec] +
				['</s>'] for sent in eval(data_type)[split]])

	final_dev_acc, dev_bleu_score, final_dev_ppl, acc_from_expl_dev = evaluate_snli_final(esnli_net, criterion_expl, 'snli_dev', snli_dev, expl_no_unk_dev, word_vec, word_index, batch_size, print_every, current_run_dir)
	test_acc, test_bleu_score, test_ppl, acc_from_expl_test = evaluate_snli_final(esnli_net, criterion_expl, 'snli_test', snli_test, expl_no_unk_test, word_vec, word_index, batch_size, print_every, current_run_dir)
	
	headers.append('SNLI-acc')
	row_dev.append(final_dev_acc)
	row_test.append(test_acc)

	headers.append('SNLI-acc_from_expl')
	row_dev.append(acc_from_expl_dev)
	row_test.append(acc_from_expl_test)

	headers.append('SNLI-ppl')
	row_dev.append(final_dev_ppl)
	row_test.append(test_ppl)

	headers.append('SNLI-BLEU')
	row_dev.append(dev_bleu_score)
	row_test.append(test_bleu_score)

	 
	# Run best model on downstream tasks.
	def prepare(params, samples):
		params.infersent.build_vocab([' '.join(s) for s in samples], tokenize=False)

	def batcher(params, batch):
		#batch = [['<s>'] + s + ['</s>'] for s in batch]
		sentences = [' '.join(s) for s in batch]
		embeddings = params.infersent.encode(sentences, bsize=params.batch_size, tokenize=False)
		return embeddings


	# final params
	params_senteval = {'task_path': PATH_TO_DATA, 'usepytorch': True, 'kfold': 10}
	params_senteval['classifier'] = {'nhid': 0, 'optim': 'adam', 'batch_size': 64, 'tenacity': 5, 'epoch_size': 4}

	# prototype params to speed up, for development only
	if use_prototype_senteval:
		params_senteval = {'task_path': PATH_TO_DATA, 'usepytorch': True, 'kfold': 5}
		params_senteval['classifier'] = {'nhid': 0, 'optim': 'rmsprop', 'batch_size': 128, 'tenacity': 3, 'epoch_size': 2}

	params_senteval['infersent'] = esnli_net.encoder
	params_senteval['infersent'].set_glove_path(GLOVE_PATH)

	se = senteval.engine.SE(params_senteval, batcher, prepare)
	results = se.eval(transfer_tasks)
	print "results ", results

	macro_dev = 0
	micro_dev = 0
	n_total_dev = 0

	macro_test = 0
	micro_test = 0
	n_total_test = 0

	delta = 0

	for task in transfer_tasks:
		if task in accuracy_tasks:
			if task == 'MRPC':
				headers.append('MRPC-acc')
				row_dev.append(round(results[task]['devacc'], 1))
				row_test.append(round(results[task]['acc'], 1))

				headers.append('MRPC-F1')
				row_test.append(round(results[task]['f1'], 1))
				row_dev.append(" ")

				delta += results[task]['f1'] - infersent_allnli['MRPC_f1']
			else:
				headers.append(task)
				row_test.append(round(results[task]['acc'], 1))
				row_dev.append(round(results[task]['devacc'], 1))
				delta += results[task]['acc'] - infersent_allnli[task]
			
			macro_test += round(results[task]['acc'], 1)
			micro_test += round(results[task]['ntest'] * results[task]['acc'], 1)
			n_total_test += results[task]['ntest']

			macro_dev += round(results[task]['devacc'], 1)
			micro_dev += round(results[task]['ndev'] * results[task]['devacc'], 1)
			n_total_dev += results[task]['ndev']

		elif task == "SICKRelatedness":
			headers.append('SICK-R_pearson')
			row_test.append(round(results[task]['pearson'], 3))
			row_dev.append(round(results[task]['devpearson'], 3))
			delta += 100 * (results[task]['pearson'] - infersent_allnli[task])

		elif task == "STS14":
			headers.append('STS14_pearson')
			row_dev.append(" ")
			row_test.append(round(results[task]['all']['pearson']['mean'], 2))

			headers.append('STS14_spearman')
			row_test.append(round(results[task]['all']['spearman']['mean'], 2))
			row_dev.append(" ")

		elif task == "STSBenchmark":
			headers.append('STSB_pearson')
			row_dev.append(round(results[task]['devpearson'], 3))
			row_test.append(round(results[task]['pearson'], 3))

			headers.append('STSB_spearman')
			row_test.append(round(results[task]['spearman'], 3))
			row_dev.append(" ")

			delta += round(100 * results[task]['spearman'], 1) - infersent_allnli['STSB_spearman']

		elif task == "ImageCaptionRetrieval":
			headers += ['Caption_retrival_R1', 'Caption_retrival_R5', 'Caption_retrival_R10', 'Caption_retrival_Medr', 'Image_retrival_R1', 'Image_retrival_R5', 'Image_retrival_R10', 'Image_retrival_Medr']
			for i in range(8):
				row_dev.append(" ")
			for j in range(2):
				for i in range(4):
					row_test.append(results[task]['acc'][j][i])

	headers.append('Delta')
	delta = round(delta/10, 2)
	row_dev.append("")
	row_test.append(delta)

	headers.append('MACRO')
	row_dev.append(round(macro_dev/len(accuracy_tasks), 1))
	row_test.append(round(macro_test/len(accuracy_tasks), 1))
	
	headers.append('MICRO')
	row_dev.append(round(micro_dev/n_total_dev, 1))
	row_test.append(round(micro_test/n_total_test, 1))
	
	if train_snli_classif:
		# Ignore the trained classifier(or it might not even be trained if alpha=0) and train the same architecure of MLP classifier on top of the learned embeddings. For the case when we had trained a classifier, let's see how the new one compares to it.
		params_senteval = {'task_path': PATH_TO_DATA, 'usepytorch': True, 'kfold': 10}
		params_senteval['classifier'] = {'nhid': params.fc_dim, 'optim': 'adam', 'batch_size': 128, 'tenacity': 5, 'epoch_size': 4}
		params_senteval['infersent'] = esnli_net.encoder
		params_senteval['infersent'].set_glove_path(GLOVE_PATH)

		se = senteval.engine.SE(params_senteval, batcher, prepare)
		results = se.eval(['SNLI'])
		print "results SNLI classif trained with SentEval ", results

		headers.append('SNLI_train_classif')
		row_dev.append(round(results['SNLI']['devacc'], 1))
		row_test.append(round(results['SNLI']['acc'], 1))
	

	writer.writerow(headers)
	writer.writerow(row_dev)
	writer.writerow(row_test)
	dev_f.close()


