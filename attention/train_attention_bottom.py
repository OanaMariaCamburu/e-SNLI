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

from data_attention_bottom import get_train, get_batch, build_vocab, get_word_dict, get_target_expl_batch, get_dev_test_with_expl, get_dev_or_test_without_expl, NLI_DIC_LABELS
from models_attention_bottom_separate import eSNLIAttention
from eval_attention import eval_all

sys.path.append("..")
from utils.mutils import get_optimizer, makedirs, pretty_duration, get_sentence_from_indices, get_key_from_val, n_parameters, remove_file, assert_sizes

import streamtologger

import matplotlib
matplotlib.use("agg")
import matplotlib.pyplot as plt


GLOVE_PATH = '../dataset/GloVe/glove.840B.300d.txt'


parser = argparse.ArgumentParser(description='NLI training')
# paths
parser.add_argument("--train_set", type=str, default='eSNLI', help="eSNLI or ALLeNLI")
parser.add_argument("--esnli_path", type=str, default='../dataset/eSNLI/', help="eSNLI data path")
parser.add_argument("--n_train", type=int, default=-1)
parser.add_argument("--save_title", type=str, default='')
parser.add_argument("--results_dir", type=str, default='results_attention')
parser.add_argument("--print_every", type=int, default=500)
parser.add_argument("--avg_every", type=int, default=100)


# training
parser.add_argument("--max_norm", type=float, default=5., help="max norm (grad clipping)")
parser.add_argument("--n_epochs", type=int, default=30)
parser.add_argument("--early_stopping_epochs", type=int, default=50)
parser.add_argument("--dpout_enc", type=float, default=0., help="encoder dropout")
parser.add_argument("--dpout_dec", type=float, default=0.5, help="decoder dropout")
parser.add_argument("--dpout_fc", type=float, default=0., help="classifier dropout")
parser.add_argument("--optimizer", type=str, default="sgd,lr=0.1", help="adam or sgd,lr=0.1")
parser.add_argument("--lrshrink", type=float, default=5, help="shrink factor for sgd, 1 = no shrink, default infersent = 5")
parser.add_argument("--decay", type=float, default=0.99, help="lr decay")
parser.add_argument("--minlr", type=float, default=1e-5, help="minimum lr")


# vocab preproc on explanations
parser.add_argument("--preproc_expl", type=str, default="preproc1")
parser.add_argument("--min_freq", type=int, default=15) # 0 for using all words


# model
parser.add_argument("--not_separate_att", action='store_false', dest='separate_att')
parser.add_argument("--att_hid_dim", type=int, default=512, help="for attention projections")

parser.add_argument("--n_layers_dec", type=int, default=1)
parser.add_argument("--encoder_type", type=str, default='BLSTMEncoder', help="see list of encoders")
parser.add_argument("--decoder_type", type=str, default='lstm', help="lstm or gru")

parser.add_argument("--not_use_init", action='store_false', dest='use_init')
parser.add_argument("--att_type", type=str, default='dot', help="dot or lin")

parser.add_argument("--batch_size", type=int, default=64)
parser.add_argument("--eval_batch_size", type=int, default=64)
parser.add_argument("--max_T_encoder", type=int, default=84)
parser.add_argument("--max_T_decoder", type=int, default=40)
parser.add_argument("--enc_rnn_dim", type=int, default=2048, help="encoder nhid dimension")
parser.add_argument("--dec_rnn_dim", type=int, default=1024, help="explanation decoder nhid dimension")
parser.add_argument("--fc_dim", type=int, default=512, help="nhid of fc layers")
parser.add_argument("--n_classes", type=int, default=3, help="entailment/neutral/contradiction")
parser.add_argument("--pool_type", type=str, default='max', help="max or mean")


# gpu
parser.add_argument("--gpu_id", type=int, default=0, help="GPU ID")
parser.add_argument("--gpu", type=int, default=None, help="for printing purposes only")
parser.add_argument("--seed", type=int, default=1234, help="seed")
parser.add_argument("--cudnn_nondeterministic", action='store_false', dest='cudnn_deterministic')

# for eval
parser.add_argument("--directory_expl_to_labels", type=str, default='../expl_to_labels/results_expl_to_labels/23:05_20:17:40_sgd,lr=0.1_Enc2048_bs256_gpu2_LRdecay0.99_MAXpool_lrshrink5')
parser.add_argument("--state_path_expl_to_labels", type=str, default='state_dict_best_devacc__devACC96.780__epoch_12_model.pt')


params = parser.parse_args()
assert(params.gpu is not None)
encoder_types = ['BLSTMEncoder', 'BLSTMprojEncoder', 'BGRUlastEncoder', 'InnerAttentionYANGEncoder', 'InnerAttentionNAACLEncoder', 'ConvNetEncoder', 'LSTMEncoder'] # 'InnerAttentionMILAEncoder', 
assert params.encoder_type in encoder_types, "encoder_type must be in " + str(encoder_types)
assert params.decoder_type in ['gru', 'lstm']
assert params.train_set in ['eSNLI', 'ALLeNLI']
assert params.att_type in ['dot', 'lin']


# CUDNN deterministic
torch.backends.cudnn.deterministic = params.cudnn_deterministic

params.results_dir = params.results_dir + "_" + params.train_set
makedirs(params.results_dir)


params.save_title += "_dec" + str(params.decoder_type.upper()) + "_" + params.optimizer + "_Enc" + str(params.enc_rnn_dim) + "_Dec" + str(params.dec_rnn_dim) + "_att_hid" + str(params.att_hid_dim) + "_bs" + str(params.batch_size) + "_gpu" + str(params.gpu) + "__encT" + str(params.max_T_encoder) + "__decT" + str(params.max_T_decoder)


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

if params.lrshrink != 1:
	params.save_title += "_lrshrink" + str(params.lrshrink)

if params.seed != 1234:
	params.save_title += "_seed" + str(params.seed)

if params.use_init:
	params.save_title += "_init"

if params.max_norm > 5:
	params.save_title += "__max_norm" + str(params.max_norm)

if params.n_layers_dec > 1:
	params.save_title += "__nlayersDec" + str(params.n_layers_dec)

if params.early_stopping_epochs < params.n_epochs:
	params.save_title += "__earlystop" + str(params.early_stopping_epochs)

if not params.cudnn_deterministic:
	params.save_title += "__NONdeterministicCUDNN"


current_run_dir = params.results_dir + "/" + time.strftime("%d:%m") + "_" + time.strftime("%H:%M:%S") + params.save_title
params.current_run_dir = current_run_dir
makedirs(current_run_dir)
copy2('models_attention_bottom_separate.py', current_run_dir)
copy2('train_attention_bottom.py', current_run_dir)
copy2('data_attention_bottom.py', current_run_dir)
copy2('eval_attention.py', current_run_dir)
copy2(os.path.join(params.directory_expl_to_labels, "models_expl_to_labels.py"), '.')
from models_expl_to_labels import ExplToLabelsNet
copy2("models_expl_to_labels.py", current_run_dir)

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
train_path = os.path.join('../dataset', params.train_set)
train = get_train(train_path, preproc, params.min_freq, params.n_train)

snli_dev = get_dev_test_with_expl(params.esnli_path, 'dev', preproc, params.min_freq)

all_sentences = train['s1'] + train['s2'] + train['expl_1'] + snli_dev['s1'] + snli_dev['s2'] + snli_dev['expl_1'] + snli_dev['expl_2'] + snli_dev['expl_3']
word_vec = build_vocab(all_sentences, GLOVE_PATH)

expl_sentences = train['expl_1'] + snli_dev['expl_1'] + snli_dev['expl_2'] + snli_dev['expl_3']
word_index = get_word_dict(expl_sentences)

params.word_index = word_index
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
	'encoder_type'   :  params.encoder_type   ,
	'decoder_type'   :  params.decoder_type   ,
	'use_cuda'       :  True                  ,
	'n_vocab'        :  params.n_vocab        ,
	'word_vec'       :  word_vec              ,
	'word_index'     :  word_index            ,
	'max_T_encoder'  :  params.max_T_encoder  ,
	'max_T_decoder'  :  params.max_T_decoder  ,
	'use_init'       :  params.use_init       ,
	'n_layers_dec'   :  params.n_layers_dec   ,
	'att_type'       :  params.att_type       ,
	'att_hid_dim'    :  params.att_hid_dim    ,
}


# model
esnli_net = eSNLIAttention(config_nli_model)
print(esnli_net)
print("Number of trainable paramters: ", n_parameters(esnli_net))

# loss expl
criterion_expl = nn.CrossEntropyLoss(ignore_index=word_index["<p>"])
criterion_expl.size_average = False

# optimizer
optim_fn, optim_params = get_optimizer(params.optimizer)
optimizer = optim_fn(esnli_net.parameters(), **optim_params)

# cuda by default
esnli_net.cuda()
criterion_expl.cuda()

"""
TRAIN
"""
def trainepoch(epoch):
	print('\nTRAINING : Epoch ' + str(epoch))
	esnli_net.train()

	expl_costs = []
	cum_n_words = 0
	cum_ppl = 0

	# shuffle the data
	permutation = np.random.permutation(len(train['s1']))

	s1 = train['s1'][permutation]
	s2 = train['s2'][permutation]
	expl_1 = train['expl_1'][permutation]
	label = train['label'][permutation]

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

		# make them variables and set them on cuda
		s1_batch, s2_batch, input_expl_batch = Variable(s1_batch.cuda()), Variable(s2_batch.cuda()), Variable(input_expl_batch.cuda())

		# taget expl is a translation of one timestep
		tgt_expl_batch, lens_tgt_expl = get_target_expl_batch(expl_1[stidx:stidx + params.batch_size], word_index)
		assert tgt_expl_batch.dim() == 2, "tgt_expl_batch.dim()=" + str(tgt_expl_batch.dim())
		tgt_expl_batch = Variable(tgt_expl_batch).cuda()
		
		# model forward train
		out_expl = esnli_net((s1_batch, s1_len), (s2_batch, s2_len), input_expl_batch, 'teacher', visualize=False)
		answer_idx = torch.max(out_expl, 2)[1]

		# print example
		if stidx % params.print_every == 0:
			print current_run_dir, '\n'
			print 'epoch: ', epoch
			print "Sentence1:  ", ' '.join(s1[stidx]), " LENGTH: ", s1_len[0]
			print "Sentence2:  ", ' '.join(s2[stidx]), " LENGTH: ", s2_len[0]
			print "Gold label:  ", get_key_from_val(label[stidx], NLI_DIC_LABELS)
			print "Explanation:  ", ' '.join(expl_1[stidx])
			print "Target expl:  ", get_sentence_from_indices(word_index, tgt_expl_batch[:, 0]), " LENGTH: ", lens_tgt_expl[0]
			print "Decoded explanation:  ", get_sentence_from_indices(word_index, answer_idx[:, 0]), "\n\n\n"


		# loss expl; out_expl is T x bs x vocab_sizes, tgt_expl_batch is T x bs
		loss_expl = criterion_expl(out_expl.view(out_expl.size(0) * out_expl.size(1), -1), tgt_expl_batch.view(tgt_expl_batch.size(0) * tgt_expl_batch.size(1)))
		expl_costs.append(loss_expl.data[0])
		cum_n_words += lens_tgt_expl.sum()
		cum_ppl += loss_expl.data[0]

		# backward
		optimizer.zero_grad()
		loss_expl.backward()

		# infersent version of gradient clipping
		shrink_factor = 1
		current_bs = len(s1_len)
		# total grads norm
		total_norm = 0
		for name, p in esnli_net.named_parameters():
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
		if len(expl_costs) == params.avg_every:
			train_expl_costs.append(np.mean(expl_costs))
			train_ppl.append(math.exp(cum_ppl/cum_n_words))
			print '{0} ; epoch: {1}, loss : {2} ; train ppl : {3}'.format(stidx, epoch, round(train_expl_costs[-1], 2), round(train_ppl[-1], 2))
			expl_costs = []
			cum_n_words = 0
			cum_ppl = 0



def evaluate_dev(epoch):
	esnli_net.eval()
	global val_ppl_best, stop_training, last_improvement_epoch

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
		
		# print example
		if i % params.print_every == 0:
			print current_run_dir, '\n'
			print "SNLI DEV example"
			print "Sentence1:  ", ' '.join(s1[i]), " LENGTH: ", s1_len[0] 
			print "Sentence2:  ", ' '.join(s2[i]), " LENGTH: ", s2_len[0] 
			print "Gold label:  ", get_key_from_val(label[i], NLI_DIC_LABELS)

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
			
			# model forward
			out_expl = esnli_net((s1_batch, s1_len), (s2_batch, s2_len), input_expl_batch, 'teacher', visualize=False)
			# ppl
			loss_expl = criterion_expl(out_expl.view(out_expl.size(0) * out_expl.size(1), -1), tgt_expl_batch.view(tgt_expl_batch.size(0) * tgt_expl_batch.size(1)))
			cum_dev_n_words += lens_tgt_expl.sum()
			cum_dev_ppl += loss_expl.data[0]
			answer_idx = torch.max(out_expl, 2)[1]
			if i % params.print_every == 0:
				print "Decoded explanation " + str(index) + " :  ", get_sentence_from_indices(word_index, answer_idx[:, 0])
				print "\n"

	dev_ppl.append(math.exp(cum_dev_ppl/cum_dev_n_words))
	current_best_model_path = None

	if dev_ppl[-1] < val_ppl_best:
		last_improvement_epoch = epoch
		print('saving model at epoch {0}'.format(epoch))
		# save with torch.save
		best_model_prefix = os.path.join(current_run_dir, 'best_devppl_')
		current_best_model_path = best_model_prefix + '_devPPL{0:.3f}__epoch_{1}_model.pt'.format(dev_ppl[-1], epoch)
		torch.save(esnli_net, current_best_model_path)
		for f in glob.glob(best_model_prefix + '*'):
			if f != current_best_model_path:
				os.remove(f)
		# also save model.state_dict()
		best_state_dict_prefix = os.path.join(current_run_dir, 'state_dict_best_devppl_')
		current_best_model_state_dict_path = best_state_dict_prefix + '_devPPL{0:.3f}__epoch_{1}_model.pt'.format(dev_ppl[-1], epoch)
		state = {'model_state': esnli_net.state_dict(), 'config_model': config_nli_model, 'params':params}
		torch.save(state, current_best_model_state_dict_path)
		for f in glob.glob(best_state_dict_prefix + '*'):
			if f != current_best_model_state_dict_path:
				os.remove(f)
		val_ppl_best = dev_ppl[-1]

	else: 
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
		

	return current_best_model_path



"""
Train model
"""
epoch = 1
dev_ppl = []
best_model_path = None


val_ppl_best = 100000000
adam_stop = False
stop_training = False
last_improvement_epoch = 0


train_expl_costs = []
train_ppl = []
total_norms = []
enc_norms = []

#while not stop_training and epoch <= params.n_epochs: 
while epoch <= params.n_epochs:    
	start = time.time()
	trainepoch(epoch)
	print "Train epoch " + str(epoch) + " took " + pretty_duration(time.time() - start)

	# All losses in normal scale
	expl_loss_line, = plt.plot(train_expl_costs, "g-", label="expl loss")
	plt.savefig(current_run_dir + "/train_expl_loss.png")
	plt.close()

	# All losses in log scale 
	expl_loss_line, = plt.semilogy(train_expl_costs, "g-", label="expl loss")
	plt.savefig(current_run_dir + "/train_expl_loss_logscale.png")
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

	
	current_best_model_path = evaluate_dev(epoch)
	if not (current_best_model_path is None):
		best_model_path = current_best_model_path

	dev_ppl_line, = plt.plot(dev_ppl, "g-", label="dev_ppl")
	plt.legend([dev_ppl_line], ['dev ppl'])
	plt.savefig(current_run_dir + "/dev_ppl.png")
	plt.close()
	

	epoch += 1

print 'grads norms before clipping ', total_norms

# Eval the best model
print "best_model_path", best_model_path
file = os.path.join(current_run_dir, 'TRAINED.txt')
f = open(file,'w')
f.write(best_model_path)
f.close()

del esnli_net
esnli_net = torch.load(best_model_path)

state_expl_to_labels = torch.load(os.path.join(params.directory_expl_to_labels, params.state_path_expl_to_labels))
model_config_expl_to_label = state_expl_to_labels['config_model']
model_state_expl_to_label = state_expl_to_labels['model_state']
expl_net = ExplToLabelsNet(model_config_expl_to_label).cuda()
expl_net.load_state_dict(model_state_expl_to_label)
eval_all(esnli_net, expl_net, criterion_expl, params)


file = os.path.join(current_run_dir, 'DONE.txt')
f = open(file,'w')
f.write("DONE")
f.close()


