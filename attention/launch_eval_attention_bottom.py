import os
import argparse
import numpy as np
import time
import shutil
from shutil import copy2

import torch
from torch.autograd import Variable
import torch.nn as nn


from eval_attention import eval_all
from models_expl_to_labels import ExplToLabelsNet
from data_attention_bottom import get_train, get_batch, build_vocab, get_word_dict, get_target_expl_batch, get_dev_test_with_expl, get_dev_or_test_without_expl, NLI_DIC_LABELS

import streamtologger

parser = argparse.ArgumentParser(description='eval')

# saved trained models
parser.add_argument("--directory", type=str, default='')
parser.add_argument("--state_path", type=str, default='')
parser.add_argument("--eval_batch_size", type=int, default=32)

parser.add_argument("--directory_expl_to_labels", type=str, default='')
parser.add_argument("--state_path_expl_to_labels", type=str, default='')

eval_params = parser.parse_args()

if not os.path.exists("copy_models_attention_bottom_separate.py"):
	shutil.copy(os.path.join(eval_params.directory, "models_attention_bottom_separate.py"), "copy_models_attention_bottom_separate.py")
from copy_models_attention_bottom_separate import eSNLIAttention
	

streamtologger.redirect(target=os.path.join(eval_params.directory, time.strftime("%d:%m") + "_" + time.strftime("%H:%M:%S") + 'log_eval.txt'))


# attention model
state_att = torch.load(os.path.join(eval_params.directory, eval_params.state_path))
model_config_att = state_att['config_model']
model_state_dict = state_att['model_state']
att_net = eSNLIAttention(model_config_att).cuda()
att_net.load_state_dict(model_state_dict)
params = state_att['params']
assert params.separate_att == eval_params.separate_att, "params.separate_att " + str(params.separate_att)
params.word_vec_expl = model_config_att['word_vec']
params.current_run_dir = eval_params.directory
params.eval_batch_size = eval_params.eval_batch_size
params.eval_just_snli  = eval_params.eval_just_snli

# expl_to_label model
state_expl_to_labels = torch.load(os.path.join(eval_params.directory_expl_to_labels, eval_params.state_path_expl_to_labels))
model_config_expl_to_label = state_expl_to_labels['config_model']
model_state_expl_to_label = state_expl_to_labels['model_state']
expl_net = ExplToLabelsNet(model_config_expl_to_label).cuda()
expl_net.load_state_dict(model_state_expl_to_label)


# set gpu device
torch.cuda.set_device(0)

# criterion
pad_idx = params.word_index["<p>"]
criterion_expl = nn.CrossEntropyLoss(ignore_index=pad_idx).cuda()
criterion_expl.size_average = False

eval_all(att_net, expl_net, criterion_expl, params)


txt_file = 'DONE_eval_att.txt'
file = os.path.join(params.current_run_dir, txt_file)
f = open(file,'w')
f.write("DONE")
f.close()

os.remove("copy_models_attention_bottom_separate.py")



