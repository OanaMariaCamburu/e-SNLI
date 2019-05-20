import os
import argparse
import numpy as np
import time
from shutil import copy2

import torch
from torch.autograd import Variable
import torch.nn as nn

from models_attention_bottom_separate import eSNLIAttention
from eval_attention import eval_all

from data_attention_bottom import get_train, get_batch, build_vocab, get_word_dict, get_target_expl_batch, get_dev_test_with_expl, get_dev_or_test_without_expl, NLI_DIC_LABELS

import streamtologger

parser = argparse.ArgumentParser(description='eval')

# saved trained models
parser.add_argument("--directory", type=str, default='')
parser.add_argument("--state_path", type=str, default='')


def visualize_attention(esnli_net, dataset, data, word_vec, word_index, current_run_dir, batch_size=1):
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
			out_expl = esnli_net((s1_batch, s1_len), (s2_batch, s2_len), input_expl_batch, mode="teacher")
			# ppl
			loss_expl = criterion_expl(out_expl.view(out_expl.size(0) * out_expl.size(1), -1), tgt_expl_batch.view(tgt_expl_batch.size(0) * tgt_expl_batch.size(1)))
			cum_test_n_words += lens_tgt_expl.sum()
			cum_test_ppl += loss_expl.data[0]
			answer_idx = torch.max(out_expl, 2)[1]
			if i % print_every == 0:
				print "Decoded explanation " + str(index) + " :  ", get_sentence_from_indices(word_index, answer_idx[:, 0])
				print "\n"

		pred_expls = esnli_net((s1_batch, s1_len), (s2_batch, s2_len), input_expl_batch, mode="forloop")
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
		bleu_score = bleu_prediction(expl_csv, snli_dev_no_unk, False, False)
	else:
		bleu_score = bleu_prediction(expl_csv, snli_test_no_unk, False, False)

	print dataset.upper() + ' SNLI accuracy: ', eval_acc, 'bleu score: ', bleu_score, 'ppl: ', eval_ppl
	return eval_acc, round(bleu_score, 2), round(eval_ppl, 2)



# attention model
state_att = torch.load(os.path.join(eval_params.directory, eval_params.state_path))
model_config_att = state_att['config_model']
model_state_dict = state_att['model_state']
att_net = eSNLIAttention(model_config_att).cuda()
att_net.load_state_dict(model_state_dict)
params = state_att['params']
params.word_vec_expl = model_config_att['word_vec']
params.current_run_dir = eval_params.directory

# set gpu device
torch.cuda.set_device(0)

visualize_attention()