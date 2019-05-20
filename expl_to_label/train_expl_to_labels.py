import os
import sys
import time
import argparse
import glob
import math
import csv
import numpy as np
from shutil import copy2

import torch
from torch.autograd import Variable
import torch.nn as nn

from data_expl_to_labels import get_train, get_batch, build_vocab, get_dev_test_with_expl, NLI_DIC_LABELS
from models_expl_to_labels import ExplToLabelsNet

sys.path.append("..")
from utils.mutils import get_optimizer, makedirs, pretty_duration, get_key_from_val, n_parameters, remove_file, assert_sizes


import streamtologger

import matplotlib
matplotlib.use("agg")
import matplotlib.pyplot as plt

GLOVE_PATH = '../dataset/GloVe/glove.840B.300d.txt'


parser = argparse.ArgumentParser(description='NLI training')
# paths
parser.add_argument("--esnli_path", type=str, default='../dataset/eSNLI/', help="eSNLI data path")
parser.add_argument("--n_train", type=int, default=-1)
parser.add_argument("--save_title", type=str, default='')
parser.add_argument("--results_dir", type=str, default='results_expl_to_labels')
parser.add_argument("--print_every", type=int, default=500)
parser.add_argument("--avg_every", type=int, default=100)


# training
parser.add_argument("--max_norm", type=float, default=5., help="max norm (grad clipping)")
parser.add_argument("--n_epochs", type=int, default=20)
parser.add_argument("--early_stopping_epochs", type=int, default=50)
parser.add_argument("--dpout_enc", type=float, default=0., help="encoder dropout")
parser.add_argument("--dpout_fc", type=float, default=0., help="classifier dropout")
parser.add_argument("--linear_fc", action='store_false', dest='nonlinear_fc', help="use nonlinearity in fc")
parser.add_argument("--optimizer", type=str, default="sgd,lr=0.1", help="adam or sgd,lr=0.1")
parser.add_argument("--lrshrink", type=float, default=5, help="shrink factor for sgd, 1 = no shrink, default infersent = 5")
parser.add_argument("--decay", type=float, default=0.99, help="lr decay")
parser.add_argument("--minlr", type=float, default=1e-5, help="minimum lr")


# model
parser.add_argument("--encoder_type", type=str, default='BLSTMEncoder', help="see list of encoders")
parser.add_argument("--batch_size", type=int, default=256)
parser.add_argument("--eval_batch_size", type=int, default=256)
parser.add_argument("--enc_rnn_dim", type=int, default=2048, help="encoder nhid dimension")
parser.add_argument("--fc_dim", type=int, default=512, help="nhid of fc layers")
parser.add_argument("--n_classes", type=int, default=3, help="entailment/neutral/contradiction")
parser.add_argument("--pool_type", type=str, default='max', help="max or mean")


# gpu
parser.add_argument("--gpu_id", type=int, default=0, help="GPU ID")
parser.add_argument("--gpu", type=int, default=None, help="for printing purposes only")
parser.add_argument("--seed", type=int, default=1234, help="seed")
parser.add_argument("--cudnn_nondeterministic", action='store_false', dest='cudnn_deterministic')


params = parser.parse_args()
assert(params.gpu is not None)
encoder_types = ['BLSTMEncoder', 'BLSTMprojEncoder', 'BGRUlastEncoder', 'InnerAttentionYANGEncoder', 'InnerAttentionNAACLEncoder', 'ConvNetEncoder', 'LSTMEncoder'] # 'InnerAttentionMILAEncoder', 
assert params.encoder_type in encoder_types, "encoder_type must be in " + str(encoder_types)
params.word_emb_dim = 300

# CUDNN deterministic
torch.backends.cudnn.deterministic = params.cudnn_deterministic


makedirs(params.results_dir)

params.save_title += "_" + params.optimizer + "_Enc" + str(params.enc_rnn_dim) + "_bs" + str(params.batch_size) + "_gpu" + str(params.gpu)

if not params.nonlinear_fc:
	print '\n\n\n WARNING: Classifier is linear only \n\n\n'
	params.save_title += "___LINEAR_CLASSIF___"

if params.fc_dim != 512:
	params.save_title += "_MLP_dim" + str(params.fc_dim)

if params.encoder_type != 'BLSTMEncoder':
	params.save_title += '_' + params.encoder_type

if "sgd" in params.optimizer:
	params.save_title += "_LRdecay" + str(params.decay)

if params.dpout_enc > 0:
	params.save_title += "_dpout_enc" + str(params.dpout_enc)

if params.dpout_fc > 0:
	params.save_title += "_dpout_classif" + str(params.dpout_fc)

if params.encoder_type in ["BLSTMEncoder", "BGRUlastEncoder", "BLSTMprojEncoder"]:
	params.save_title += "_" + params.pool_type.upper() + "pool"

if params.lrshrink != 1:
	params.save_title += "_lrshrink" + str(params.lrshrink)

if params.seed != 1234:
	params.save_title += "_seed" + str(params.seed)

if params.max_norm > 5:
	params.save_title += "__max_norm" + str(params.max_norm)

if params.early_stopping_epochs < params.n_epochs:
	params.save_title += "__earlystop" + str(params.early_stopping_epochs)

if not params.cudnn_deterministic:
	params.save_title += "__NONdeterministicCUDNN"

current_run_dir = params.results_dir + "/" + time.strftime("%d:%m") + "_" + time.strftime("%H:%M:%S") + params.save_title
params.current_run_dir = current_run_dir
makedirs(current_run_dir)
copy2('models_expl_to_labels.py', current_run_dir)
copy2('train_expl_to_labels.py', current_run_dir)
copy2('data_expl_to_labels.py', current_run_dir)

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
torch.manual_seed(params.seed)
torch.cuda.manual_seed(params.seed)


"""
ALL DATA, some will only be needed for eval for we want to build glove vocab once
"""

train = get_train(params.esnli_path, params.n_train)

snli_dev = get_dev_test_with_expl(params.esnli_path, 'dev')
snli_test = get_dev_test_with_expl(params.esnli_path, 'test')

all_sentences = train['expl_1'] + snli_dev['expl_1'] + snli_dev['expl_2'] + snli_dev['expl_3'] + snli_test['expl_1'] + snli_test['expl_2'] + snli_test['expl_3']
word_vec = build_vocab(all_sentences, GLOVE_PATH)

for split in ['expl_1']:
	train[split] = np.array([['<s>'] + [word for word in sent.split() if word in word_vec] + ['</s>'] for sent in train[split]])

for split in ['expl_1', 'expl_2', 'expl_3']:
	snli_dev[split] = np.array([['<s>'] + [word for word in sent.split() if word in word_vec] + ['</s>'] for sent in snli_dev[split]])
	snli_test[split] = np.array([['<s>'] + [word for word in sent.split() if word in word_vec] + ['</s>'] for sent in snli_test[split]])


"""
CREATE MODEL
"""
# model config
config_nli_model = {
	'word_emb_dim'   :  params.word_emb_dim   ,
	'enc_rnn_dim'    :  params.enc_rnn_dim    ,
	'dpout_enc'      :  params.dpout_enc      ,
	'dpout_fc'       :  params.dpout_fc       ,
	'fc_dim'         :  params.fc_dim         ,
	'bsize'          :  params.batch_size     ,
	'n_classes'      :  params.n_classes      ,
	'pool_type'      :  params.pool_type      ,
	'nonlinear_fc'   :  params.nonlinear_fc   ,
	'encoder_type'   :  params.encoder_type   ,
	'use_cuda'       :  True                  ,
	'word_vec'       :  word_vec              ,
}


# model
esnli_net = ExplToLabelsNet(config_nli_model)
print(esnli_net)
print("Number of trainable paramters: ", n_parameters(esnli_net))

# loss labels
criterion_labels = nn.CrossEntropyLoss()
criterion_labels.size_average = False

# optimizer
optim_fn, optim_params = get_optimizer(params.optimizer)
optimizer = optim_fn(esnli_net.parameters(), **optim_params)

# cuda by default
esnli_net.cuda()
criterion_labels.cuda()

"""
TRAIN
"""
def trainepoch(epoch):
	print('\nTRAINING : Epoch ' + str(epoch))
	esnli_net.train()

	label_costs = []
	correct = 0.

	# shuffle the data
	permutation = np.random.permutation(len(train['expl_1']))
	expl_1 = train['expl_1'][permutation]
	label = train['label'][permutation]

	optimizer.param_groups[0]['lr'] = optimizer.param_groups[0]['lr'] * params.decay if epoch>1\
		and 'sgd' in params.optimizer else optimizer.param_groups[0]['lr']
	print('Learning rate : {0}'.format(optimizer.param_groups[0]['lr']))

	for stidx in range(0, len(expl_1), params.batch_size):
		# prepare batch
		expl_batch, expl_len = get_batch(expl_1[stidx:stidx + params.batch_size], word_vec)
		expl_batch = Variable(expl_batch.cuda())

		tgt_label_batch = Variable(torch.LongTensor(label[stidx:stidx + params.batch_size])).cuda()

		
		# model forward train
		out_lbl = esnli_net((expl_batch, expl_len))
		pred = out_lbl.data.max(1)[1]
		current_bs = len(pred)
		correct += pred.long().eq(tgt_label_batch.data.long()).cpu().sum()

		# print example
		if stidx % params.print_every == 0:
			print current_run_dir, '\n'
			print 'epoch: ', epoch
			print "Explanation:  ", ' '.join(expl_1[stidx])
			print "Gold label:  ", get_key_from_val(label[stidx], NLI_DIC_LABELS)
			print "Predicted label:  ", get_key_from_val(pred[0], NLI_DIC_LABELS)
			
		# loss labels
		loss_labels = criterion_labels(out_lbl, tgt_label_batch)
		label_costs.append(loss_labels.data[0])

		# backward
		optimizer.zero_grad()
		loss_labels.backward()

		# infersent version of gradient clipping
		# TODO: set correct division by number of decoded tokens for the optimizer of the decoder
		shrink_factor = 1

		# total grads norm
		total_norm = 0
		for p in esnli_net.parameters():
			if p.requires_grad:
				p.grad.data.div_(current_bs)  # divide by the actual batch size
				total_norm += p.grad.data.norm() ** 2
		total_norm = np.sqrt(total_norm)
		total_norms.append(total_norm)

		# encoder grads norm
		enc_norm = 0
		for p in esnli_net.encoder.parameters():
			if p.requires_grad:
				#p.grad.data.div_(current_bs)  # divide by the actual batch size
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
		if len(label_costs) == params.avg_every:
			train_label_costs.append(np.mean(label_costs))
			print '{0} ; epoch: {1}, total loss : {2} ; accuracy train expl_to_labels : {3}'.format(stidx, epoch, round(train_label_costs[-1], 2), round(100.*correct/(stidx+expl_batch.size(1)), 2))
			label_costs = []

	train_acc = round(100 * correct/len(expl_1), 2)
	print('results : epoch {0} ; mean accuracy train esnli : {1}'.format(epoch, train_acc))
	return train_acc


def evaluate_dev(epoch):
	esnli_net.eval()
	global val_acc_best, stop_training, last_improvement_epoch

	correct = 0.

	print('\DEV : Epoch {0}'.format(epoch))

	# eSNLI
	expl_1 = snli_dev['expl_1']
	expl_2 = snli_dev['expl_2']
	expl_3 = snli_dev['expl_3']
	label = snli_dev['label']

	for i in range(0, len(expl_1), params.eval_batch_size):
		# prepare batch
		tgt_label_batch = Variable(torch.LongTensor(label[i:i + params.eval_batch_size])).cuda()
		
		# print example
		if i % params.print_every == 0:
			print current_run_dir, '\n'
			print "SNLI DEV example" 
			print "Gold label:  ", get_key_from_val(label[i], NLI_DIC_LABELS)

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

	total_dev_points = 3 * len(expl_1)

	# accuracy
	eval_acc = round(100 * correct / total_dev_points, 2)
	print 'togrep : results : epoch {0} ; mean accuracy {1} '.format(epoch, eval_acc)

	current_best_model_path = None

	if eval_acc > val_acc_best:
		last_improvement_epoch = epoch

		print('saving model at epoch {0}'.format(epoch))
		# save with torch.save
		best_model_prefix = os.path.join(current_run_dir, 'best_devacc_')
		current_best_model_path = best_model_prefix + '_devACC{0:.3f}__epoch_{1}_model.pt'.format(eval_acc, epoch)
		torch.save(esnli_net, current_best_model_path)
		for f in glob.glob(best_model_prefix + '*'):
			if f != current_best_model_path:
				os.remove(f)
		# also save model.state_dict()
		best_state_dict_prefix = os.path.join(current_run_dir, 'state_dict_best_devacc_')
		current_best_model_state_dict_path = best_state_dict_prefix + '_devACC{0:.3f}__epoch_{1}_model.pt'.format(eval_acc, epoch)
		state = {'model_state': esnli_net.state_dict(), 'config_model': config_nli_model, 'params':params}
		torch.save(state, current_best_model_state_dict_path)
		for f in glob.glob(best_state_dict_prefix + '*'):
			if f != current_best_model_state_dict_path:
				os.remove(f)
		val_acc_best = eval_acc

	else: # no improvement 
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
		

	return eval_acc, current_best_model_path



"""
Train model
"""
epoch = 1
dev_accuracies = []
best_model_path = None

val_acc_best = -1e10
adam_stop = False
stop_training = False
last_improvement_epoch = 0

train_label_costs = []
total_norms = []
enc_norms = []

while not stop_training and epoch <= params.n_epochs:    
	start = time.time()
	train_acc = trainepoch(epoch)
	print "Train epoch " + str(epoch) + " took " + pretty_duration(time.time() - start)

	# All losses in normal scale
	lbl_loss_line, = plt.plot(train_label_costs, "b-", label="label loss")
	plt.savefig(current_run_dir + "/train_loss.png")
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


	dev_acc_line, = plt.plot(dev_accuracies, "g-", label="dev_acc")
	plt.legend([dev_acc_line], ['dev accuracies'])
	plt.savefig(current_run_dir + "/dev_acc.png")
	plt.close()

	epoch += 1

print 'grads norms before clipping ', total_norms

# Eval the best model
# Run best model on SNLI test.
print "best_model_path", best_model_path
file = os.path.join(current_run_dir, 'TRAINED.txt')
f = open(file,'w')
f.write(best_model_path)
f.close()


def evaluate_test():
	esnli_net.eval()
	
	correct = 0.

	print('TEST')

	# eSNLI
	expl_1 = snli_test['expl_1']
	expl_2 = snli_test['expl_2']
	expl_3 = snli_test['expl_3']
	label = snli_test['label']

	for i in range(0, len(expl_1), params.eval_batch_size):
		# prepare batch
		tgt_label_batch = Variable(torch.LongTensor(label[i:i + params.eval_batch_size])).cuda()
		
		# print example
		if i % params.print_every == 0:
			print current_run_dir, '\n'
			print "SNLI TEST example" 
			print "Gold label:  ", get_key_from_val(label[i], NLI_DIC_LABELS)

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
	print 'test accuracy ', eval_acc

evaluate_test()
