import os
import sys
import time
import argparse
import glob
import math
import csv
import numpy as np
import random
from shutil import copy2

import torch
from torch.autograd import Variable
import torch.nn as nn

import streamtologger

import matplotlib
matplotlib.use("agg")
import matplotlib.pyplot as plt

from data_label_in_expl import get_train, get_batch, build_vocab, get_word_dict, get_target_expl_batch, get_dev_test_with_expl, get_dev_or_test_without_expl, NLI_DIC_LABELS, NLI_LABELS_TO_NLI
from models_esnli_init import eSNLINet
from eval_sent_embeddings_labels_in_expl import eval_all
sys.path.append("..")
from utils.mutils import get_optimizer, makedirs, pretty_duration, get_sentence_from_indices, get_key_from_val, n_parameters, remove_file, assert_sizes, permute


GLOVE_PATH = '../dataset/GloVe/glove.840B.300d.txt'


parser = argparse.ArgumentParser(description='NLI training')
# paths
parser.add_argument("--train_set", type=str, default='eSNLI', help="eSNLI or ALLeNLI")
parser.add_argument("--esnli_path", type=str, default='../dataset/eSNLI/old_dataset_format/', help="eSNLI data path")
parser.add_argument("--n_train", type=int, default=-1)
parser.add_argument("--save_title", type=str, default='')
parser.add_argument("--results_dir", type=str, default='results_label_in_expl')
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
parser.add_argument("--relu_before_pool", action='store_true', dest='relu_before_pool')


# vocab preproc on explanations
parser.add_argument("--preproc_expl", type=str, default="preproc1")
parser.add_argument("--min_freq", type=int, default=15) # 0 for using all words


# model
parser.add_argument("--n_layers_dec", type=int, default=1)
parser.add_argument("--encoder_type", type=str, default='BLSTMEncoder', help="see list of encoders")
parser.add_argument("--decoder_type", type=str, default='lstm', help="lstm or gru")

parser.add_argument("--use_vocab_proj", action='store_true', dest='use_vocab_proj')
parser.add_argument("--vocab_proj_dim", type=int, default=512)

parser.add_argument("--not_use_init", action='store_false', dest='use_init')
parser.add_argument("--only_diff_prod", action='store_true', dest='only_diff_prod')
parser.add_argument("--not_use_diff_prod_sent_embed", action='store_false', dest='use_diff_prod_sent_embed')

parser.add_argument("--use_smaller_inp_dec_dim", action='store_true', dest='use_smaller_inp_dec_dim')
parser.add_argument("--smaller_inp_dec_dim", type=int, default=2048)

parser.add_argument("--batch_size", type=int, default=64)
parser.add_argument("--eval_batch_size", type=int, default=64)
parser.add_argument("--max_T_decoder", type=int, default=40)
parser.add_argument("--enc_rnn_dim", type=int, default=2048, help="encoder nhid dimension")
parser.add_argument("--dec_rnn_dim", type=int, default=512, help="explanation decoder nhid dimension")
parser.add_argument("--fc_dim", type=int, default=512, help="nhid of fc layers")
parser.add_argument("--n_classes", type=int, default=3, help="entailment/neutral/contradiction")
parser.add_argument("--pool_type", type=str, default='max', help="max or mean")


# alpha
parser.add_argument("--alpha", type=float, default=0.6, help="alpha * lbl + (1-alpha) * expl ; use 1 for training just the classif and 0 for training just explanations")
parser.add_argument("--annealing_alpha", action='store_true', dest='annealing_alpha', help="it starts with the alpha above and increases it by 0.1 every epoch until it reaches 1") 
parser.add_argument("--annealing_rate", type=float, default=0.1) 
parser.add_argument("--annealing_max", type=float, default=1) 
parser.add_argument("--lmbda", type=int, default=1, help='multiply with both losses to see the efect of alpha=0.5') 


# gpu
parser.add_argument("--gpu_id", type=int, default=0, help="GPU ID")
parser.add_argument("--gpu", type=int, default=None, help="for printing purposes only")
parser.add_argument("--seed", type=int, default=1234, help="seed")
parser.add_argument("--cudnn_nondeterministic", action='store_false', dest='cudnn_deterministic')


params = parser.parse_args()
assert(params.gpu is not None)
encoder_types = ['BLSTMEncoder', 'BLSTMprojEncoder', 'BGRUlastEncoder', 'InnerAttentionYANGEncoder', 'InnerAttentionNAACLEncoder', 'ConvNetEncoder', 'LSTMEncoder'] # 'InnerAttentionMILAEncoder', 
assert params.encoder_type in encoder_types, "encoder_type must be in " + str(encoder_types)
assert params.decoder_type in ['gru', 'lstm']
assert params.train_set in ['eSNLI', 'ALLeNLI']

if params.alpha < 0.01:
	params.train_snli_classif = True

if params.alpha == 0.5:
	params.lmbda = 2

if params.n_train != -1:
	params.use_prototype_senteval = True
	params.save_title += "__DEV__"

# CUDNN deterministic
torch.backends.cudnn.deterministic = params.cudnn_deterministic


params.results_dir = params.results_dir + "_" + params.train_set 
makedirs(params.results_dir)

params.save_title += "_dec" + str(params.decoder_type.upper()) + "_" + params.optimizer + "_Enc" + str(params.enc_rnn_dim) + "_Dec" + str(params.dec_rnn_dim) + "_bs" + str(params.batch_size) + "_a" + str(params.alpha) + "_gpu" + str(params.gpu)

if not params.nonlinear_fc:
	print '\n\n\n WARNING: Classifier is linear only \n\n\n'
	params.save_title += "___LINEAR_CLASSIF___"

if params.fc_dim != 512:
	params.save_title += "_MLP_dim" + str(params.fc_dim)

if params.encoder_type != 'BLSTMEncoder':
	params.save_title += '_' + params.encoder_type

if params.min_freq != 15:
	params.save_title += "_min_freq" + str(params.min_freq)

if "sgd" in params.optimizer:
	params.save_title += "_LRdecay" + str(params.decay)

if params.dpout_enc > 0:
	params.save_title += "_dpout_enc" + str(params.dpout_enc)

if params.dpout_dec > 0:
	params.save_title += "_dpout_dec" + str(params.dpout_dec)

if params.dpout_fc > 0:
	params.save_title += "_dpout_classif" + str(params.dpout_fc)

if params.encoder_type in ["BLSTMEncoder", "BGRUlastEncoder", "BLSTMprojEncoder"]:
	params.save_title += "_" + params.pool_type.upper() + "pool"

if params.annealing_alpha:
	params.save_title += "_anneal_rate" + str(params.annealing_rate) + "_anneal_max" + str(params.annealing_max)

if params.lrshrink != 1:
	params.save_title += "_lrshrink" + str(params.lrshrink)

if params.lmbda != 1:
	params.save_title += "_lmbda" + str(params.lmbda)

if params.seed != 1234:
	params.save_title += "_seed" + str(params.seed)

if params.use_vocab_proj:
	params.save_title += "_use_vocab_proj" + str(params.vocab_proj_dim)

if not params.use_init:
	params.save_title += "_NOinit"

if params.use_smaller_inp_dec_dim:
	params.save_title += "_inp_dec" + str(params.smaller_inp_dec_dim)

if params.max_norm > 5:
	params.save_title += "__max_norm" + str(params.max_norm)

if params.use_diff_prod_sent_embed:
	params.save_title += "__diff_prod_embed"

if params.n_layers_dec > 1:
	params.save_title += "__nlayersDec" + str(params.n_layers_dec)

if params.early_stopping_epochs < params.n_epochs:
	params.save_title += "__earlystop" + str(params.early_stopping_epochs)

if not params.cudnn_deterministic:
	params.save_title += "__NONdeterministicCUDNN"

if params.relu_before_pool:
	params.save_title += "__relu"

current_run_dir = params.results_dir + "/" + time.strftime("%d:%m") + "_" + time.strftime("%H:%M:%S") + params.save_title # + str(sys.argv[1:])
params.current_run_dir = current_run_dir
makedirs(current_run_dir)
copy2('models_esnli_init.py', current_run_dir)
copy2('train_label_in_expl.py', current_run_dir)
copy2('data_label_in_expl.py', current_run_dir)
copy2('eval_sent_embeddings_labels_in_expl.py', current_run_dir)

streamtologger.redirect(target=current_run_dir + '/log.txt')

# set gpu device
torch.cuda.set_device(params.gpu_id)

# print parameters passed, and all parameters
print('\ntogrep : {0}\n'.format(sys.argv[1:]))
print(params)


"""
SEED
"""
np.random.seed(params.seed)
random.seed(params.seed)
torch.manual_seed(params.seed)
torch.cuda.manual_seed(params.seed)


"""
ALL DATA, some will only be needed for eval for we want to build glove vocab once
"""
preproc = params.preproc_expl + "_"
train = get_train(params.esnli_path, preproc, params.min_freq, params.n_train)

snli_dev = get_dev_test_with_expl(params.esnli_path, 'dev', preproc, params.min_freq)

all_sentences = train['s1'] + train['s2'] + train['expl_1'] + snli_dev['s1'] + snli_dev['s2'] + snli_dev['expl_1'] + snli_dev['expl_2'] + snli_dev['expl_3']
word_vec = build_vocab(all_sentences, GLOVE_PATH)

expl_sentences_train = train['expl_1']
word_index_train = get_word_dict(expl_sentences_train)
expl_sentences = train['expl_1'] + snli_dev['expl_1'] + snli_dev['expl_2'] + snli_dev['expl_3'] + snli_test['expl_1'] + snli_test['expl_2'] + snli_test['expl_3']
word_index = get_word_dict(expl_sentences)
params.word_index = word_index

print "difference ", set(word_index.keys()) - set(word_index_train.keys())
if params.n_train == -1:
	# there may be some words that appear in premise and hypothesis of train as well as in expl of dev or test but not in explanation of train. There was only one as far as i looked but if there are too many we should maybe take care of this.
	assert len(word_index) - len(word_index_train) < 5, "n words in train " + str(len(word_index_train)) + " while n words in total " + str(len(word_index))

params.word_emb_dim = 300
params.n_vocab = len(word_index)
print("Total number of EXPL words", params.n_vocab)

for split in ['s1', 's2', 'expl_1']:
	train[split] = np.array([['<s>'] + [word for word in sent.split() if word in word_vec] + ['</s>'] for sent in train[split]])

for split in ['s1', 's2', 'expl_1', 'expl_2', 'expl_3']:
	snli_dev[split] = np.array([['<s>'] + [word for word in sent.split() if word in word_vec] + ['</s>'] for sent in snli_dev[split]])


"""
CREATE MODEL
"""
# model config
config_nli_model = {
	'word_emb_dim'   :  params.word_emb_dim   ,
	'enc_rnn_dim'    :  params.enc_rnn_dim    ,
	'dec_rnn_dim'    :  params.dec_rnn_dim    ,
	'dpout_enc'      :  params.dpout_enc      ,
	'dpout_dec'      :  params.dpout_dec      ,
	'dpout_fc'       :  params.dpout_fc       ,
	'fc_dim'         :  params.fc_dim         ,
	'bsize'          :  params.batch_size     ,
	'n_classes'      :  params.n_classes      ,
	'pool_type'      :  params.pool_type      ,
	'nonlinear_fc'   :  params.nonlinear_fc   ,
	'encoder_type'   :  params.encoder_type   ,
	'decoder_type'   :  params.decoder_type   ,
	'use_cuda'       :  True                  ,
	'n_vocab'        :  params.n_vocab        ,
	'word_vec'       :  word_vec              ,
	'word_index'     :  word_index            ,
	'max_T_decoder'  :  params.max_T_decoder  ,
	'use_vocab_proj' :  params.use_vocab_proj ,
	'vocab_proj_dim' :  params.vocab_proj_dim ,
	'use_init'       :  params.use_init       ,
	'n_layers_dec'   :  params.n_layers_dec   ,
	'only_diff_prod' :  params.only_diff_prod ,
	'use_smaller_inp_dec_dim' : params. use_smaller_inp_dec_dim,
	'smaller_inp_dec_dim'     :  params.smaller_inp_dec_dim    ,
	'use_diff_prod_sent_embed' :  params.use_diff_prod_sent_embed,
	'relu_before_pool' : params.relu_before_pool,

}


# model
esnli_net = eSNLINet(config_nli_model)
print(esnli_net)
print("Number of trainable paramters: ", n_parameters(esnli_net))

# loss labels
criterion_labels = nn.CrossEntropyLoss()
criterion_labels.size_average = False

# loss expl
criterion_expl = nn.CrossEntropyLoss(ignore_index=word_index["<p>"])
criterion_expl.size_average = False

# optimizer
optim_fn, optim_params = get_optimizer(params.optimizer)
optimizer = optim_fn(esnli_net.parameters(), **optim_params)

# cuda by default
esnli_net.cuda()
criterion_labels.cuda()
criterion_expl.cuda()

"""
TRAIN
"""
def trainepoch(epoch):
	print('\nTRAINING : Epoch ' + str(epoch))
	esnli_net.train()

	if (epoch > 1) and (params.annealing_alpha) and (params.alpha + params.annealing_rate <= params.annealing_max):
		params.alpha += params.annealing_rate
		print "alpha: ", str(params.alpha)
	
	label_costs = []
	expl_costs = []
	all_losses = []
	cum_n_words = 0
	cum_ppl = 0
	correct = 0.

	# shuffle the data
	permutation = np.random.permutation(len(train['s1']))

	s1 = train['s1'][permutation]
	s2 = train['s2'][permutation]
	expl_1 = train['expl_1'][permutation]
	label = train['label'][permutation]
	label_expl = permute(train['label_expl'], permutation)

	optimizer.param_groups[0]['lr'] = optimizer.param_groups[0]['lr'] * params.decay if epoch>1\
		and 'sgd' in params.optimizer else optimizer.param_groups[0]['lr']
	print('Learning rate : {0}'.format(optimizer.param_groups[0]['lr']))

	for stidx in range(0, len(s1), params.batch_size):
		# prepare batch
		s1_batch, s1_len = get_batch(s1[stidx:stidx + params.batch_size], word_vec)
		s2_batch, s2_len = get_batch(s2[stidx:stidx + params.batch_size], word_vec)
		input_expl_batch, _ = get_batch(expl_1[stidx:stidx + params.batch_size], word_vec)
		   
		# eliminate last input to explanation because we wouldn't need to input </s> and we need same number of input and output
		input_expl_batch = input_expl_batch[:-1] 
		
		s1_batch, s2_batch, input_expl_batch = Variable(s1_batch.cuda()), Variable(s2_batch.cuda()), Variable(input_expl_batch.cuda())
		tgt_label_batch = Variable(torch.LongTensor(label[stidx:stidx + params.batch_size])).cuda()
		tgt_label_expl_batch = label_expl[stidx:stidx + params.batch_size]

		tgt_expl_batch, lens_tgt_expl = get_target_expl_batch(expl_1[stidx:stidx + params.batch_size], word_index)
		assert tgt_expl_batch.dim() == 2, "tgt_expl_batch.dim()=" + str(tgt_expl_batch.dim())
		tgt_expl_batch = Variable(tgt_expl_batch).cuda()
		
		# model forward train
		out_expl, out_lbl = esnli_net((s1_batch, s1_len), (s2_batch, s2_len), input_expl_batch, 'teacher')
		
		pred = out_lbl.data.max(1)[1]
		current_bs = len(pred)
		correct += pred.long().eq(tgt_label_batch.data.long()).cpu().sum()
		assert len(pred) == len(s1[stidx:stidx + params.batch_size]), "len(pred)=" + str(len(pred)) + " while len(s1[stidx:stidx + params.batch_size])=" + str(len(s1[stidx:stidx + params.batch_size]))
		answer_idx = torch.max(out_expl, 2)[1]

		# print example
		if stidx % params.print_every == 0:
			print current_run_dir, '\n'
			print 'epoch: ', epoch
			print "Sentence1:  ", ' '.join(s1[stidx]), " LENGTH: ", s1_len[0]
			print "Sentence2:  ", ' '.join(s2[stidx]), " LENGTH: ", s2_len[0]
			print "Gold label:  ", get_key_from_val(label[stidx], NLI_DIC_LABELS)
			print "Predicted label:  ", get_key_from_val(pred[0], NLI_DIC_LABELS)
			print "Explanation:  ", ' '.join(expl_1[stidx])
			print "Target expl:  ", get_sentence_from_indices(word_index, tgt_expl_batch[:, 0]), " LENGTH: ", lens_tgt_expl[0]
			print "Decoded explanation:  ", get_sentence_from_indices(word_index, answer_idx[:, 0]), "\n\n\n"
			

		# loss labels
		loss_labels = criterion_labels(out_lbl, tgt_label_batch)
		label_costs.append(loss_labels.data[0])

		# loss expl; out_expl is T x bs x vocab_sizes, tgt_expl_batch is T x bs
		loss_expl = criterion_expl(out_expl.view(out_expl.size(0) * out_expl.size(1), -1), tgt_expl_batch.view(tgt_expl_batch.size(0) * tgt_expl_batch.size(1)))
		expl_costs.append(loss_expl.data[0])
		cum_n_words += lens_tgt_expl.sum()
		cum_ppl += loss_expl.data[0]

		# backward
		loss = params.lmbda * (params.alpha * loss_labels + (1 - params.alpha) * loss_expl)
		all_losses.append(loss.data[0])
		optimizer.zero_grad()
		loss.backward()

		# infersent version of gradient clipping
		shrink_factor = 1

		# total grads norm
		total_norm = 0
		for p in esnli_net.parameters():
			if p.requires_grad:
				p.grad.data.div_(current_bs)
				total_norm += p.grad.data.norm() ** 2
		total_norm = np.sqrt(total_norm)
		total_norms.append(total_norm)

		# encoder grads norm
		enc_norm = 0
		for p in esnli_net.encoder.parameters():
			if p.requires_grad:
				enc_norm += p.grad.data.norm() ** 2
		enc_norm = np.sqrt(enc_norm)
		enc_norms.append(enc_norm)

		if total_norm > params.max_norm:
			shrink_factor = params.max_norm / total_norm
		current_lr = optimizer.param_groups[0]['lr'] # current lr (no external "lr", for adam)
		optimizer.param_groups[0]['lr'] = current_lr * shrink_factor # just for update
	
		# optimizer step
		optimizer.step()
		optimizer.param_groups[0]['lr'] = current_lr

		# print and reset losses
		if len(all_losses) == params.avg_every:
			train_all_losses.append(np.mean(all_losses))
			train_expl_costs.append(params.lmbda * (1 - params.alpha) * np.mean(expl_costs))
			train_label_costs.append(params.lmbda * params.alpha * np.mean(label_costs))
			train_ppl.append(math.exp(cum_ppl/cum_n_words))
			print '{0} ; epoch: {1}, total loss : {2} ; lmbda * alpha * (lbl loss) : {3}; lmbda * (1-alpha) * (expl loss) : {4} ; train ppl : {5}; accuracy train esnli : {6}'.format(stidx, epoch, round(train_all_losses[-1], 2), round(train_label_costs[-1], 2), round(train_expl_costs[-1], 2), round(train_ppl[-1], 2), round(100.*correct/(stidx+s1_batch.size(1)), 2))
			label_costs = []
			expl_costs = []
			all_losses = []
			cum_n_words = 0
			cum_ppl = 0
	train_acc = round(100 * correct/len(s1), 2)
	print('results : epoch {0} ; mean accuracy train esnli : {1}'.format(epoch, train_acc))
	return train_acc


def evaluate_dev(epoch):
	esnli_net.eval()
	global val_acc_best, val_ppl_best, stop_training, last_improvement_epoch

	correct = 0.
	cum_dev_ppl = 0
	cum_dev_n_words = 0

	print('\DEV : Epoch {0}'.format(epoch))

	# eSNLI
	s1 = snli_dev['s1']
	s2 = snli_dev['s2']
	expl_1 = snli_dev['expl_1']
	expl_2 = snli_dev['expl_2']
	expl_3 = snli_dev['expl_3']
	label = snli_dev['label']

	for i in range(0, len(s1), params.eval_batch_size):
		# prepare batch
		s1_batch, s1_len = get_batch(s1[i:i + params.eval_batch_size], word_vec)
		s2_batch, s2_len = get_batch(s2[i:i + params.eval_batch_size], word_vec)
		s1_batch, s2_batch = Variable(s1_batch.cuda()), Variable(s2_batch.cuda())
		tgt_label_batch = Variable(torch.LongTensor(label[i:i + params.eval_batch_size])).cuda()
		
		# print example
		if i % params.print_every == 0:
			print current_run_dir, '\n'
			print "SNLI DEV example"
			print "Sentence1:  ", ' '.join(s1[i]), " LENGTH: ", s1_len[0] 
			print "Sentence2:  ", ' '.join(s2[i]), " LENGTH: ", s2_len[0] 
			print "Gold label:  ", get_key_from_val(label[i], NLI_DIC_LABELS)

		out_lbl = [0, 1, 2]
		for index in range(1, 4):
			expl = eval("expl_" + str(index))
			input_expl_batch, _ = get_batch(expl[i:i + params.eval_batch_size], word_vec)
			input_expl_batch = Variable(input_expl_batch[:-1].cuda())
			if i % params.print_every == 0:
				print "Explanation " + str(index) + " :  ", ' '.join(expl[i])
			tgt_expl_batch, lens_tgt_expl = get_target_expl_batch(expl[i:i + params.eval_batch_size], word_index)
			assert tgt_expl_batch.dim() == 2, "tgt_expl_batch.dim()=" + str(tgt_expl_batch.dim())
			tgt_expl_batch = Variable(tgt_expl_batch).cuda()
			if i % params.print_every == 0:
				print "Target expl " + str(index) + " :  ", get_sentence_from_indices(word_index, tgt_expl_batch[:, 0]), " LENGHT: ", lens_tgt_expl[0]
			
			# model forward, tgt_label is None for both v1 and v2 bcs it's test time for v2
			out_expl, out_lbl[index-1] = esnli_net((s1_batch, s1_len), (s2_batch, s2_len), input_expl_batch, 'teacher')
			# ppl
			loss_expl = criterion_expl(out_expl.view(out_expl.size(0) * out_expl.size(1), -1), tgt_expl_batch.view(tgt_expl_batch.size(0) * tgt_expl_batch.size(1)))
			cum_dev_n_words += lens_tgt_expl.sum()
			cum_dev_ppl += loss_expl.data[0]
			answer_idx = torch.max(out_expl, 2)[1]
			if i % params.print_every == 0:
				print "Decoded explanation " + str(index) + " :  ", get_sentence_from_indices(word_index, answer_idx[:, 0])
				print "\n"

		assert torch.equal(out_lbl[0], out_lbl[1]), "out_lbl[0]: " + str(out_lbl[0]) + " while " + "out_lbl[1]: " + str(out_lbl[1]) 
		assert torch.equal(out_lbl[1], out_lbl[2]), "out_lbl[1]: " + str(out_lbl[1]) + " while " + "out_lbl[2]: " + str(out_lbl[2]) 
		# accuracy
		pred = out_lbl[0].data.max(1)[1]
		if i % params.print_every == 0:
			print "Predicted label:  ", get_key_from_val(pred[0], NLI_DIC_LABELS), "\n\n\n"
		correct += pred.long().eq(tgt_label_batch.data.long()).cpu().sum()

	total_dev_points = len(s1)

	
	# accuracy
	eval_acc = round(100 * correct / total_dev_points, 2)
	print 'togrep : results : epoch {0} ; mean accuracy {1} '.format(epoch, eval_acc)

	dev_ppl.append(math.exp(cum_dev_ppl/cum_dev_n_words))
	current_best_model_path = None
	current_best_model_state_dict_path = None

	if eval_acc > val_acc_best or dev_ppl[-1] < val_ppl_best:
		last_improvement_epoch = epoch
		
		# if alpha > 0 we only save the model if increase in ACC
		if params.alpha > 0.01 and eval_acc > val_acc_best:
			print('saving model at epoch {0}'.format(epoch))
			# save with torch.save
			best_model_prefix = os.path.join(current_run_dir, 'best_devacc_')
			current_best_model_path = best_model_prefix + '_devACC{0:.3f}_devppl{1:.3f}__epoch_{2}_model.pt'.format(eval_acc, dev_ppl[-1], epoch)
			torch.save(esnli_net, current_best_model_path)
			for f in glob.glob(best_model_prefix + '*'):
				if f != current_best_model_path:
					os.remove(f)
			# also save model.state_dict()
			best_state_dict_prefix = os.path.join(current_run_dir, 'state_dict_best_devacc_')
			current_best_model_state_dict_path = best_state_dict_prefix + '_devACC{0:.3f}_devppl{1:.3f}__epoch_{2}_model.pt'.format(eval_acc, dev_ppl[-1], epoch)
			state = {'model_state': esnli_net.state_dict(), 'config_model': config_nli_model, 'params':params}
			torch.save(state, current_best_model_state_dict_path)
			for f in glob.glob(best_state_dict_prefix + '*'):
				if f != current_best_model_state_dict_path:
					os.remove(f)
			val_acc_best = eval_acc
			if dev_ppl[-1] < val_ppl_best:
				val_ppl_best = dev_ppl[-1]
			
		# if alpha = 0 (EXPL_ONLY) we only save the model if decrease in PPL
		elif params.alpha < 0.01 and dev_ppl[-1] < val_ppl_best:
			print('saving model at epoch {0}'.format(epoch))
			# save with torch.save
			best_model_prefix = os.path.join(current_run_dir, 'best_devppl_')
			current_best_model_path = best_model_prefix + '_devPPL{0:.3f}__epoch_{1}_model.pt'.format(dev_ppl[-1], epoch)
			torch.save(esnli_net, current_best_model_path)
			for f in glob.glob(best_model_prefix + '*'):
				if f != current_best_model_path:
					os.remove(f)
			# save model.state_dict()
			best_state_dict_prefix = os.path.join(current_run_dir, 'state_dict_best_devppl_')
			current_best_model_state_dict_path = best_state_dict_prefix + '_devPPL{0:.3f}__epoch_{1}_model.pt'.format(dev_ppl[-1], epoch)
			state = {'model_state': esnli_net.state_dict(), 'config_model': config_nli_model, 'params':params}
			torch.save(state, current_best_model_state_dict_path)
			for f in glob.glob(best_state_dict_prefix + '*'):
				if f != current_best_model_state_dict_path:
					os.remove(f)
			val_ppl_best = dev_ppl[-1]

	else: # no improvement at all, regardless whether it's in PPL or ACC
		if 'sgd' in params.optimizer:
			optimizer.param_groups[0]['lr'] = optimizer.param_groups[0]['lr'] / params.lrshrink
			print('Shrinking lr by : {0}. New lr = {1}'.format(params.lrshrink, optimizer.param_groups[0]['lr']))
			if optimizer.param_groups[0]['lr'] < params.minlr:
				stop_training = True
				print "Stopping training because LR < ", params.minlr

		
		# for any optimizer early stopping
		if (epoch - last_improvement_epoch > params.early_stopping_epochs):
			stop_training = True
			print "Stopping training because no more improvement done in the last ", params.early_stopping_epochs, " epochs"
		

	return eval_acc, current_best_model_state_dict_path



"""
Train model
"""
epoch = 1
dev_accuracies = []
dev_ppl = []
best_model_path = None

val_acc_best = -1e10
val_ppl_best = 100000000
adam_stop = False
stop_training = False
last_improvement_epoch = 0

train_label_costs = []
train_expl_costs = []
train_all_losses = []
train_ppl = []
total_norms = []
enc_norms = []

while not stop_training and epoch <= params.n_epochs:    
	start = time.time()
	train_acc = trainepoch(epoch)
	print "Train epoch " + str(epoch) + " took " + pretty_duration(time.time() - start)

	# All losses in normal scale
	all_loss_line, = plt.plot(train_all_losses, "r-", label="full loss")
	lbl_loss_line, = plt.plot(train_label_costs, "b-", label="label loss")
	expl_loss_line, = plt.plot(train_expl_costs, "g-", label= str(params.lmbda * (1 - params.alpha)) + " * expl loss")
	plt.legend([all_loss_line, lbl_loss_line, expl_loss_line], ['full loss', str(params.lmbda * params.alpha) + ' * (lbl loss)', str(params.lmbda * (1 - params.alpha)) + ' * expl loss'])
	plt.savefig(current_run_dir + "/train_loss.png")
	plt.close()

	# All losses in log scale
	all_loss_line, = plt.semilogy(train_all_losses, "r-", label="full loss")
	lbl_loss_line, = plt.semilogy(train_label_costs, "b-", label="label loss")
	expl_loss_line, = plt.semilogy(train_expl_costs, "g-", label= str(params.lmbda * (1 - params.alpha)) + " * expl loss")
	plt.legend([all_loss_line, lbl_loss_line, expl_loss_line], ['full loss', str(params.lmbda * params.alpha) + ' * (lbl loss)', str(params.lmbda * (1 - params.alpha)) + ' * expl loss'])
	plt.savefig(current_run_dir + "/train_loss.png")
	plt.close()

	# Labels loss separately
	train_lbl_line, = plt.plot(train_label_costs, "g-", label="train_label_loss")
	plt.legend([train_lbl_line], ['train_label'])
	plt.savefig(current_run_dir + "/train_label.png")
	plt.close()

	# PPL train
	train_ppl_line, = plt.semilogy(train_ppl, "g-", label="train_ppl")
	plt.legend([train_ppl_line], ['train ppl'])
	plt.savefig(current_run_dir + "/train_ppl.png")
	plt.close()

	# Total grad norms after average by batch size 
	total_grads_norm_line, = plt.plot(total_norms, "g-", label="total_grads_norm")
	plt.legend([total_grads_norm_line], ['total_grads'])
	plt.savefig(current_run_dir + "/total_grads.png")
	plt.close()

	# RNN encoder grad norms
	enc_grads_norm_line, = plt.plot(enc_norms, "g-", label="enc_grads_norm")
	plt.legend([enc_grads_norm_line], ['enc_grads_norm'])
	plt.savefig(current_run_dir + "/enc_grads_norm.png")
	plt.close()

	eval_acc, current_best_model_path = evaluate_dev(epoch)
	dev_accuracies.append(eval_acc)
	if not (current_best_model_path is None):
		best_model_path = current_best_model_path

	dev_ppl_line, = plt.plot(dev_ppl, "g-", label="dev_ppl")
	plt.legend([dev_ppl_line], ['dev ppl'])
	plt.savefig(current_run_dir + "/dev_ppl.png")
	plt.close()

	dev_acc_line, = plt.plot(dev_accuracies, "g-", label="dev_acc")
	plt.legend([dev_acc_line], ['dev accuracies'])
	plt.savefig(current_run_dir + "/dev_acc.png")
	plt.close()

	epoch += 1

print 'grads norms before clipping ', total_norms

# Eval the best model
print "best_model_path", best_model_path
file = os.path.join(current_run_dir, 'TRAINED.txt')
f = open(file,'w')
f.write(best_model_path)
f.close()

# load best model from state
state_best_model = torch.load(best_model_path)['model_state']
esnli_net.load_state_dict(state_best_model)
eval_all(esnli_net, criterion_expl, params)


file = os.path.join(current_run_dir, 'DONE.txt')
f = open(file,'w')
f.write("DONE")
f.close()


