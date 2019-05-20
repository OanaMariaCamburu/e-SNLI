"""
The initialization only uses the premise and hypothesis embeddings but not the diff_product
"""

import numpy as np
import time
import torch
import torch.nn as nn
from torch.autograd import Variable
import sys
#sys.path.append("..")
from mutils import get_keys_from_vals, assert_sizes, get_key_from_val
from data_label_in_expl import NLI_DIC_LABELS

def array_all_true(arr):
	for i in arr:
		if i == False:
			return False
	return True

"""
Decoder for the explanation
"""
class Decoder(nn.Module):
	def __init__(self, config):
		super(Decoder, self).__init__()
		
		self.decoder_type = config['decoder_type']
		self.word_emb_dim = config['word_emb_dim']
		self.dec_rnn_dim = config['dec_rnn_dim']
		self.dpout_dec = config['dpout_dec']
		self.n_vocab = config['n_vocab']
		self.word_index = config['word_index']
		self.word_vec = config['word_vec']
		self.max_T_decoder = config['max_T_decoder']
		self.use_vocab_proj = config['use_vocab_proj']
		self.vocab_proj_dim = config['vocab_proj_dim']
		self.use_init = config['use_init']
		self.n_layers_dec = config['n_layers_dec']
		self.use_smaller_inp_dec_dim = config['use_smaller_inp_dec_dim']
		self.smaller_inp_dec_dim = config['smaller_inp_dec_dim']
		self.use_diff_prod_sent_embed = config['use_diff_prod_sent_embed']
		self.only_diff_prod = config['only_diff_prod']
		

		self.sent_dim = 2 * config['enc_rnn_dim']
		if config['encoder_type'] in ["ConvNetEncoder", "InnerAttentionMILAEncoder"]:
			self.sent_dim = 4 * self.sent_dim 
		if config['encoder_type'] == "LSTMEncoder":
			self.sent_dim = self.sent_dim / 2 

		assert self.sent_dim == 4096

		self.context_mutiply_coef = 2
		if self.use_diff_prod_sent_embed:
			self.context_mutiply_coef = 4
			
		# initialization only uses the premise and hypothesis embeddings but not the diff_product
		if self.use_init and 2 * self.sent_dim != self.dec_rnn_dim:
			self.proj_init = nn.Linear(2 * self.sent_dim, self.dec_rnn_dim)

		self.inp_dec_dim = self.dec_rnn_dim
		if self.use_smaller_inp_dec_dim:
			assert self.smaller_inp_dec_dim < self.dec_rnn_dim, str(self.smaller_inp_dec_dim) + " should be smaller than " + str(self.dec_rnn_dim)
			self.inp_dec_dim = self.smaller_inp_dec_dim
		self.proj_inp_dec = nn.Linear(self.context_mutiply_coef * self.sent_dim + self.word_emb_dim, self.inp_dec_dim)
		
		if self.decoder_type == 'gru':
			self.decoder_rnn = nn.GRU(self.inp_dec_dim, self.dec_rnn_dim, self.n_layers_dec, bidirectional=False, dropout=self.dpout_dec)
		else: # 'lstm'
			self.decoder_rnn = nn.LSTM(self.inp_dec_dim, self.dec_rnn_dim, self.n_layers_dec, bidirectional=False, dropout=self.dpout_dec)

		if self.use_vocab_proj:
			self.vocab_proj = nn.Linear(self.dec_rnn_dim, self.vocab_proj_dim)
			self.vocab_layer = nn.Linear(self.vocab_proj_dim, self.n_vocab)
		else:
			self.vocab_layer = nn.Linear(self.dec_rnn_dim, self.n_vocab)


	def forward(self, expl, s1_embed, s2_embed, mode, classif_lbl):
		# expl: Variable(seqlen x bsize x worddim)
		# s1/2_embed: Variable(bsize x sent_dim)
		
		assert mode in ['forloop', 'teacher'], mode

		batch_size = expl.size(1)
		assert_sizes(s1_embed, 2, [batch_size, self.sent_dim])
		assert_sizes(s2_embed, 2, [batch_size, self.sent_dim])
		assert_sizes(expl, 3, [expl.size(0), batch_size, self.word_emb_dim])
		
		context = torch.cat([s1_embed, s2_embed], 1).unsqueeze(0)
		if self.use_diff_prod_sent_embed:
			context = torch.cat([s1_embed, s2_embed, torch.abs(s1_embed - s2_embed), s1_embed * s2_embed], 1).unsqueeze(0)
		if self.only_diff_prod:
			context = torch.cat([torch.abs(s1_embed - s2_embed), s1_embed * s2_embed], 1).unsqueeze(0)

		assert_sizes(context, 3, [1, batch_size, self.context_mutiply_coef * self.sent_dim])

		# init decoder
		context_init = torch.cat([s1_embed, s2_embed], 1).unsqueeze(0)
		if self.use_init:
			if 2 * self.sent_dim != self.dec_rnn_dim:
				init_0 = self.proj_init(context_init.expand(self.n_layers_dec, batch_size, 2 * self.sent_dim))
			else:
				init_0 = context_init
		else:
			init_0 = Variable(torch.zeros(self.n_layers_dec, batch_size, self.dec_rnn_dim)).cuda()

		init_state = init_0
		if self.decoder_type == 'lstm':
			init_state = (init_0, init_0)

		self.decoder_rnn.flatten_parameters()
		
		if mode == "teacher":
			input_dec = torch.cat([expl, context.expand(expl.size(0), batch_size, self.context_mutiply_coef * self.sent_dim)], 2)
			input_dec = self.proj_inp_dec(nn.Dropout(self.dpout_dec)(input_dec))

			out, _ = self.decoder_rnn(input_dec, init_state)
			dp_out = nn.Dropout(self.dpout_dec)(out)

			if not self.use_vocab_proj:
				return self.vocab_layer(dp_out)
			return self.vocab_layer(self.vocab_proj(dp_out))
		
		else:
			assert classif_lbl is not None
			assert_sizes(classif_lbl, 1, [batch_size])
			pred_expls = []
			finished = []
			for i in range(batch_size):
				pred_expls.append("")
				finished.append(False)
			
			dec_inp_t = torch.cat([expl[0, :, :].unsqueeze(0), context], 2)
			dec_inp_t = self.proj_inp_dec(dec_inp_t)
			
			ht = init_state
			t = 0
			while t < self.max_T_decoder and not array_all_true(finished):
				t += 1
				word_embed = torch.zeros(1, batch_size, self.word_emb_dim)
				assert_sizes(dec_inp_t, 3, [1, batch_size, self.inp_dec_dim])
				dec_out_t, ht = self.decoder_rnn(dec_inp_t, ht) 
				assert_sizes(dec_out_t, 3, [1, batch_size, self.dec_rnn_dim])
				if self.use_vocab_proj:
					out_t_proj = self.vocab_proj(dec_out_t)
					out_t =self.vocab_layer(out_t_proj).data
				else:
					out_t = self.vocab_layer(dec_out_t).data # TODO: Use torch.stack with variables instead
				assert_sizes(out_t, 3, [1, batch_size, self.n_vocab])
				i_t = torch.max(out_t, 2)[1]
				assert_sizes(i_t, 2, [1, batch_size])
				pred_words = get_keys_from_vals(i_t, self.word_index) # array of bs of words at current timestep
				assert len(pred_words) == batch_size, "pred_words " + str(len(pred_words)) + " batch_size " + str(batch_size)
				for i in range(batch_size):
					if pred_words[i] == '</s>':
						finished[i] = True
					if not finished[i]:
						pred_expls[i] += " " + pred_words[i]
					if t > 1:
						#print "self.word_vec[pred_words[i]]", type(self.word_vec[pred_words[i]]) 
						word_embed[0, i] = torch.from_numpy(self.word_vec[pred_words[i]])
						#print "type(word_embed[0, i]) ", word_embed[0, i]
						#assert False
					else:
						# put label predicted by classifier
						classif_label = get_key_from_val(classif_lbl[i], NLI_DIC_LABELS)
						assert classif_label in ['entailment', 'contradiction', 'neutral'], classif_label
						word_embed[0, i] = torch.from_numpy(self.word_vec[classif_label])
				word_embed = Variable(word_embed.cuda())
				assert_sizes(word_embed, 3, [1, batch_size, self.word_emb_dim])
				dec_inp_t = self.proj_inp_dec(torch.cat([word_embed, context], 2))
			return pred_expls



"""
BLSTM (max/mean) encoder
"""
class BLSTMEncoder(nn.Module):

	def __init__(self, config):
		super(BLSTMEncoder, self).__init__()
		self.bsize = config['bsize']
		self.word_emb_dim = config['word_emb_dim']
		self.enc_rnn_dim = config['enc_rnn_dim']
		self.pool_type = config['pool_type']
		self.dpout_enc = config['dpout_enc']
		self.relu_before_pool = config['relu_before_pool']

		self.enc_lstm = nn.LSTM(self.word_emb_dim, self.enc_rnn_dim, 1,
								bidirectional=True, dropout=self.dpout_enc)

	def is_cuda(self):
		# either all weights are on cpu or they are on gpu
		return 'cuda' in str(type(self.enc_lstm.bias_hh_l0.data))

	def forward(self, sent_tuple):
		# sent_len: [max_len, ..., min_len] (bsize)
		# sent: Variable(seqlen x bsize x worddim)
		sent, sent_len = sent_tuple

		# Sort by length (keep idx)
		sent_len, idx_sort = np.sort(sent_len)[::-1], np.argsort(-sent_len)
		idx_unsort = np.argsort(idx_sort)

		idx_sort = torch.from_numpy(idx_sort).cuda() if self.is_cuda() \
			else torch.from_numpy(idx_sort)
		sent = sent.index_select(1, Variable(idx_sort))

		# Handling padding in Recurrent Networks
		sent_packed = nn.utils.rnn.pack_padded_sequence(sent, sent_len)
		self.enc_lstm.flatten_parameters()
		sent_output = self.enc_lstm(sent_packed)[0]  # seqlen x batch x 2*nhid
		padding_value = 0.0
		if self.pool_type == "max":
			padding_value = -100
		sent_output = nn.utils.rnn.pad_packed_sequence(sent_output, False, padding_value)[0]

		# Un-sort by length
		idx_unsort = torch.from_numpy(idx_unsort).cuda() if self.is_cuda() \
			else torch.from_numpy(idx_unsort)
		sent_output = sent_output.index_select(1, Variable(idx_unsort))
		if self.relu_before_pool:
			sent_output = nn.ReLU()(sent_output)
		sent_len=sent_len[idx_unsort]

		# Pooling
		if self.pool_type == "mean":
			sent_len = Variable(torch.FloatTensor(sent_len)).unsqueeze(1).cuda()
			emb = torch.sum(sent_output, 0).squeeze(0)
			emb = emb / sent_len.expand_as(emb)
		elif self.pool_type == "max":
			emb = torch.max(sent_output, 0)[0]
			if emb.ndimension() == 3:
				emb = emb.squeeze(0)
				assert emb.ndimension() == 2, "emb.ndimension()=" + str(emb.ndimension())

		return emb

	def set_glove_path(self, glove_path):
		self.glove_path = glove_path

	def get_word_dict(self, sentences, tokenize=True):
		# create vocab of words
		word_dict = {}
		if tokenize:
			from nltk.tokenize import word_tokenize
		sentences = [s.split() if not tokenize else word_tokenize(s)
					 for s in sentences]
		for sent in sentences:
			for word in sent:
				if word not in word_dict:
					word_dict[word] = ''
		word_dict['<s>'] = ''
		word_dict['</s>'] = ''
		return word_dict

	def get_glove(self, word_dict):
		assert hasattr(self, 'glove_path'), \
			   'warning : you need to set_glove_path(glove_path)'
		# create word_vec with glove vectors
		word_vec = {}
		with open(self.glove_path) as f:
			for line in f:
				word, vec = line.split(' ', 1)
				if word in word_dict:
					word_vec[word] = np.fromstring(vec, sep=' ')
		print('Found {0}(/{1}) words with glove vectors'.format(
					len(word_vec), len(word_dict)))
		return word_vec

	def get_glove_k(self, K):
		assert hasattr(self, 'glove_path'), 'warning : you need \
											 to set_glove_path(glove_path)'
		# create word_vec with k first glove vectors
		k = 0
		word_vec = {}
		with open(self.glove_path) as f:
			for line in f:
				word, vec = line.split(' ', 1)
				if k <= K:
					word_vec[word] = np.fromstring(vec, sep=' ')
					k += 1
				if k > K:
					if word in ['<s>', '</s>']:
						word_vec[word] = np.fromstring(vec, sep=' ')

				if k > K and all([w in word_vec for w in ['<s>', '</s>']]):
					break
		return word_vec

	def build_vocab(self, sentences, tokenize=True):
		assert hasattr(self, 'glove_path'), 'warning : you need \
											 to set_glove_path(glove_path)'
		word_dict = self.get_word_dict(sentences, tokenize)
		self.word_vec = self.get_glove(word_dict)
		print('Vocab size from within BLSTMEncoder : {0}'.format(len(self.word_vec)))

	# build GloVe vocab with k most frequent words
	def build_vocab_k_words(self, K):
		assert hasattr(self, 'glove_path'), 'warning : you need \
											 to set_glove_path(glove_path)'
		self.word_vec = self.get_glove_k(K)
		print('Vocab size : {0}'.format(K))

	def update_vocab(self, sentences, tokenize=True):
		assert hasattr(self, 'glove_path'), 'warning : you need \
											 to set_glove_path(glove_path)'
		assert hasattr(self, 'word_vec'), 'build_vocab before updating it'
		word_dict = self.get_word_dict(sentences, tokenize)

		# keep only new words
		for word in self.word_vec:
			if word in word_dict:
				del word_dict[word]

		# udpate vocabulary
		if word_dict:
			new_word_vec = self.get_glove(word_dict)
			self.word_vec.update(new_word_vec)
		print('New vocab size : {0} (added {1} words)'.format(
						len(self.word_vec), len(new_word_vec)))

	def get_batch(self, batch):
		# sent in batch in decreasing order of lengths
		# batch: (bsize, max_len, word_dim)
		embed = np.zeros((len(batch[0]), len(batch), self.word_emb_dim))

		for i in range(len(batch)):
			for j in range(len(batch[i])):
				embed[j, i, :] = self.word_vec[batch[i][j]]

		return torch.FloatTensor(embed)

	def prepare_samples(self, sentences, bsize, tokenize, verbose):
		if tokenize:
			from nltk.tokenize import word_tokenize
		sentences = [['<s>'] + s.split() + ['</s>'] if not tokenize else
					 ['<s>']+word_tokenize(s)+['</s>'] for s in sentences]
		n_w = np.sum([len(x) for x in sentences])

		# filters words without glove vectors
		for i in range(len(sentences)):
			s_f = [word for word in sentences[i] if word in self.word_vec]
			if not s_f:
				import warnings
				warnings.warn('No words in "{0}" (idx={1}) have glove vectors. \
							   Replacing by "</s>"..'.format(sentences[i], i))
				s_f = ['</s>']
			sentences[i] = s_f

		lengths = np.array([len(s) for s in sentences])
		n_wk = np.sum(lengths)
		if verbose:
			print('Nb words kept : {0}/{1} ({2} %)'.format(
						n_wk, n_w, round((100.0 * n_wk) / n_w, 2)))

		# sort by decreasing length
		lengths, idx_sort = np.sort(lengths)[::-1], np.argsort(-lengths)
		sentences = np.array(sentences)[idx_sort]

		return sentences, lengths, idx_sort

	def encode(self, sentences, bsize=64, tokenize=True, verbose=False):
		tic = time.time()
		sentences, lengths, idx_sort = self.prepare_samples(
						sentences, bsize, tokenize, verbose)

		embeddings = []
		for stidx in range(0, len(sentences), bsize):
			batch = Variable(self.get_batch(
						sentences[stidx:stidx + bsize]), volatile=True)
			if self.is_cuda():
				batch = batch.cuda()
			batch = self.forward(
				(batch, lengths[stidx:stidx + bsize])).data.cpu().numpy()
			embeddings.append(batch)
		embeddings = np.vstack(embeddings)

		# unsort
		idx_unsort = np.argsort(idx_sort)
		embeddings = embeddings[idx_unsort]

		if verbose:
			print('Speed : {0} sentences/s ({1} mode, bsize={2})'.format(
					round(len(embeddings)/(time.time()-tic), 2),
					'gpu' if self.is_cuda() else 'cpu', bsize))
		return embeddings

	def visualize(self, sent, tokenize=True):
		if tokenize:
			from nltk.tokenize import word_tokenize

		sent = sent.split() if not tokenize else word_tokenize(sent)
		sent = [['<s>'] + [word for word in sent if word in self.word_vec] +
				['</s>']]

		if ' '.join(sent[0]) == '<s> </s>':
			import warnings
			warnings.warn('No words in "{0}" have glove vectors. Replacing \
						   by "<s> </s>"..'.format(sent))
		batch = Variable(self.get_batch(sent), volatile=True)

		if self.is_cuda():
			batch = batch.cuda()
		output = self.enc_lstm(batch)[0]
		output, idxs = torch.max(output, 0)
		# output, idxs = output.squeeze(), idxs.squeeze()
		idxs = idxs.data.cpu().numpy()
		argmaxs = [np.sum((idxs == k)) for k in range(len(sent[0]))]

		# visualize model
		import matplotlib.pyplot as plt
		x = range(len(sent[0]))
		y = [100.0*n/np.sum(argmaxs) for n in argmaxs]
		plt.xticks(x, sent[0], rotation=45)
		plt.bar(x, y)
		plt.ylabel('%')
		plt.title('Visualisation of words importance')
		plt.show()

		return output, idxs


"""
Main module for Natural Language Inference
"""
class eSNLINet(nn.Module):
	def __init__(self, config):
		super(eSNLINet, self).__init__()

		# classifier
		self.nonlinear_fc = config['nonlinear_fc']
		self.fc_dim = config['fc_dim']
		self.n_classes = 3
		self.enc_rnn_dim = config['enc_rnn_dim']
		self.dec_rnn_dim = config['dec_rnn_dim']
		self.encoder_type = config['encoder_type']
		self.dpout_fc = config['dpout_fc']
		self.n_vocab = config['n_vocab']

		self.encoder = eval(self.encoder_type)(config)
		self.inputdim = 4*2*self.enc_rnn_dim
		self.inputdim = 4*self.inputdim if self.encoder_type in \
						["ConvNetEncoder", "InnerAttentionMILAEncoder"] else self.inputdim
		self.inputdim = self.inputdim/2 if self.encoder_type == "LSTMEncoder" \
										else self.inputdim
		if self.nonlinear_fc:
			self.classifier = nn.Sequential(
				nn.Dropout(p=self.dpout_fc),
				nn.Linear(self.inputdim, self.fc_dim),
				nn.Tanh(),
				nn.Dropout(p=self.dpout_fc),
				nn.Linear(self.fc_dim, self.fc_dim),
				nn.Tanh(),
				nn.Dropout(p=self.dpout_fc),
				nn.Linear(self.fc_dim, self.n_classes),
				)
		else:
			self.classifier = nn.Sequential(
				nn.Linear(self.inputdim, self.fc_dim),
				nn.Linear(self.fc_dim, self.fc_dim),
				nn.Linear(self.fc_dim, self.n_classes)
				)

		self.decoder = Decoder(config)


	def forward(self, s1, s2, expl, mode):
		# s1 : (s1, s1_len)
		# s2 : (s2, s2_len)
		# expl : Variable(T x bs x 300)

		u = self.encoder(s1)
		v = self.encoder(s2)

		features = torch.cat((u, v, torch.abs(u-v), u*v), 1)
		out_label = self.classifier(features)

		# expl is none for the dev of multinli during validation of the model
		if expl is None:
			return out_label

		if mode == "forloop":
			pred = out_label.data.max(1)[1]
			out_expl = self.decoder(expl, u, v, mode, pred)
		else:
			out_expl = self.decoder(expl, u, v, mode, None)
		
		return out_expl, out_label

	def encode(self, s1):
		emb = self.encoder(s1)
		return emb