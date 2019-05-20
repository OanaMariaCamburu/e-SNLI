import os
import argparse
import numpy as np
import time
import csv
from shutil import copy2

import torch
from torch.autograd import Variable
import torch.nn as nn

from models_esnli import eSNLINet
from eval_exp2 import eval_all
from models_expl_to_labels import ExplToLabelsNet
from data_esnli import get_train, get_batch, build_vocab, get_word_dict, get_target_expl_batch, get_dev_test_with_expl, get_dev_or_test_without_expl, NLI_DIC_LABELS


import streamtologger


def evaluate_test(esnli_net, snli_data, params, dataset='test'):
	esnli_net.eval()
	
	correct = 0.

	print('TEST')

	headers = ["gold_label", "Premise", "Hypothesis", "Expl_1", "pred_lbl_1", "Expl_2", "pred_lbl_2", "Expl_3", "pred_lbl_3"]
	expl_csv = os.path.join(params.directory, time.strftime("%d:%m") + "_" + time.strftime("%H:%M:%S") + "_" + dataset + ".csv")
	remove_file(expl_csv)
	expl_f = open(expl_csv, "a")
	writer = csv.writer(expl_f)
	writer.writerow(headers)


	# eSNLI
	premise = snli_data
	expl_1 = snli_data['expl_1']
	expl_2 = snli_data['expl_2']
	expl_3 = snli_data['expl_3']
	label = snli_data['label']

	for i in range(0, len(expl_1), params.eval_batch_size):
		# prepare batch
		tgt_label_batch = Variable(torch.LongTensor(label[i:i + params.eval_batch_size])).cuda()
		
		# print example
		if i % params.print_every == 0:
			print current_run_dir, '\n'
			print "SNLI TEST example" 
			print "Gold label:  ", get_key_from_val(label[i], NLI_DIC_LABELS)

		row = []
		row.append()

		for index in range(1, 4):
			expl = eval("expl_" + str(index))
			expl_batch, len_expl = get_batch(expl[i:i + params.eval_batch_size], word_vec)
			expl_batch = Variable(expl_batch.cuda())

			if i % params.print_every == 0:
				print "Explanation " + str(index) + " :  ", ' '.join(expl[i])
			
			# model fwd
			out_lbl = esnli_net((expl_batch, len_expl))
			pred = out_lbl.data.max(1)[1]
			if i % params.print_every == 0:
				print "Predicted label:  ", get_key_from_val(pred[0], NLI_DIC_LABELS), "\n"
			
			correct += pred.long().eq(tgt_label_batch.data.long()).cpu().sum()

	total_test_points = 3 * len(expl_1)

	# accuracy
	eval_acc = round(100 * correct / total_test_points, 2)
	return eval_acc
	print 'test accuracy ', eval_acc


GLOVE_PATH = '../dataset/GloVe/glove.840B.300d.txt'

parser = argparse.ArgumentParser(description='eval')


# paths
parser.add_argument("--train_set", type=str, default='eSNLI', help="eSNLI or ALLeNLI")
parser.add_argument("--esnli_path", type=str, default='../dataset/eSNLI/', help="eSNLI data path")
parser.add_argument("--n_train", type=int, default=-1)
parser.add_argument("--save_title", type=str, default='')
parser.add_argument("--results_dir", type=str, default='results')
parser.add_argument("--print_every", type=int, default=500)
parser.add_argument("--avg_every", type=int, default=100)


# training
parser.add_argument("--max_norm", type=float, default=5., help="max norm (grad clipping)")
parser.add_argument("--n_epochs", type=int, default=20)
parser.add_argument("--early_stopping_epochs", type=int, default=50)
parser.add_argument("--dpout_enc", type=float, default=0., help="encoder dropout")
parser.add_argument("--dpout_dec", type=float, default=0.5, help="decoder dropout")
parser.add_argument("--dpout_fc", type=float, default=0., help="classifier dropout")
parser.add_argument("--nonlinear_fc", action='store_true', dest='nonlinear_fc', help="use nonlinearity in fc")
parser.add_argument("--optimizer", type=str, default="sgd,lr=0.1", help="adam or sgd,lr=0.1")
parser.add_argument("--lrshrink", type=float, default=5, help="shrink factor for sgd, 1 = no shrink, default infersent = 5")
parser.add_argument("--decay", type=float, default=0.99, help="lr decay")
parser.add_argument("--minlr", type=float, default=1e-5, help="minimum lr")


# senteval
parser.add_argument("--use_prototype_senteval", action='store_true', dest='use_prototype_senteval')
parser.add_argument("--do_image_caption", action='store_true', dest='do_image_caption')
parser.add_argument("--train_snli_classif", action='store_true', dest='train_snli_classif')


# vocab preproc on explanations
parser.add_argument("--preproc_expl", type=str, default="preproc1")
parser.add_argument("--max_tokens", type=int, default=40) # to truncate the expl if longer
parser.add_argument("--min_freq", type=int, default=15) # 0 for using all words


# model
parser.add_argument("--model_v", type=int, default='1', help="1 or 2")
parser.add_argument("--n_layers_dec", type=int, default=1)
parser.add_argument("--encoder_type", type=str, default='BLSTMEncoder', help="see list of encoders")
parser.add_argument("--decoder_type", type=str, default='lstm', help="lstm or gru")

parser.add_argument("--use_vocab_proj", action='store_true', dest='use_vocab_proj')
parser.add_argument("--vocab_proj_dim", type=int, default=512)

parser.add_argument("--use_init", action='store_true', dest='use_init')

parser.add_argument("--use_diff_prod_sent_embed", action='store_true', dest='use_diff_prod_sent_embed')

parser.add_argument("--use_smaller_inp_dec_dim", action='store_true', dest='use_smaller_inp_dec_dim')
parser.add_argument("--smaller_inp_dec_dim", type=int, default=2048)

parser.add_argument("--batch_size", type=int, default=64)
parser.add_argument("--eval_batch_size", type=int, default=64)
parser.add_argument("--max_T_decoder", type=int, default=40)
parser.add_argument("--enc_rnn_dim", type=int, default=2048, help="encoder nhid dimension")
parser.add_argument("--dec_rnn_dim", type=int, default=4096, help="explanation decoder nhid dimension")
parser.add_argument("--fc_dim", type=int, default=512, help="nhid of fc layers")
parser.add_argument("--n_classes", type=int, default=3, help="entailment/neutral/contradiction")
parser.add_argument("--pool_type", type=str, default='max', help="max or mean")



# alpha
parser.add_argument("--alpha", type=float, default=0.1, help="alpha * lbl + (1-alpha) * expl ; use 1 for training just the classif and 0 for training just explanations")
parser.add_argument("--annealing_alpha", action='store_true', dest='annealing_alpha', help="it starts with the alpha above and increases it by 0.1 every epoch until it reaches 1") 
parser.add_argument("--annealing_rate", type=float, default=0.1) 
parser.add_argument("--annealing_max", type=float, default=0.9) 
parser.add_argument("--lmbda", type=int, default=1, help='multiply with both losses to see the efect of alpha=0.5') 

parser.add_argument("--annealing_down", action='store_true', dest='annealing_down')
parser.add_argument("--annealing_min", type=float, default=0.5) 


# gpu
parser.add_argument("--gpu_id", type=int, default=0, help="GPU ID")
parser.add_argument("--gpu", type=int, default=None, help="for printing purposes only")
parser.add_argument("--seed", type=int, default=1234, help="seed")


# saved models paths
parser.add_argument("--directory", type=str, default='')
parser.add_argument("--state_path", type=str, default='')


params = parser.parse_args()

params.current_run_dir = params.directory_att_model

streamtologger.redirect(target=os.path.join(params.current_run_dir, time.strftime("%d:%m") + "_" + time.strftime("%H:%M:%S") + 'log_eval.txt'))


"""
ALL DATA, some will only be needed for eval for we want to build glove vocab once
"""
preproc = params.preproc_expl + "_maxtokens_" + str(params.max_tokens) + "_"
train_path = os.path.join('dataset', params.train_set)
train = get_train(train_path, preproc, params.min_freq, params.n_train)
snli_dev = get_dev_test_with_expl(params.esnli_path, 'dev', preproc, params.min_freq)

expl_sentences_train = train['expl_1']
word_index_train = get_word_dict(expl_sentences_train)
expl_sentences = train['expl_1'] + snli_dev['expl_1'] + snli_dev['expl_2'] + snli_dev['expl_3']
word_index = get_word_dict(expl_sentences)
params.word_index = word_index

params.word_emb_dim = 300
params.n_vocab = len(word_index)

"""
END of copy from train
"""

# attention model
state_att = torch.load(os.path.join(params.directory_att_model, params.state_path_att_model))
model_config_att = state_att['config_model']
model_state_dict = state_att['model_state']
params.word_vec_expl = model_config_att['word_vec']


# load models
att_net = eSNLINet(model_config_att).cuda()
att_net.load_state_dict(model_state_dict)

# set gpu device
torch.cuda.set_device(0)

# criterion
pad_idx = params.word_index["<p>"]
criterion_expl = nn.CrossEntropyLoss(ignore_index=pad_idx).cuda()
criterion_expl.size_average = False

eval_all(att_net, expl_net, criterion_expl, params)


txt_file = 'DONE_eval.txt'
file = os.path.join(params.current_run_dir, txt_file)
f = open(file,'w')
f.write("DONE")
f.close()



