import numpy as np
import time
import torch
import torch.nn as nn
from torch.autograd import Variable



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
class ExplToLabelsNet(nn.Module):
	def __init__(self, config):
		super(ExplToLabelsNet, self).__init__()

		# classifier
		self.nonlinear_fc = config['nonlinear_fc']
		self.fc_dim = config['fc_dim']
		self.n_classes = 3
		self.enc_rnn_dim = config['enc_rnn_dim']
		self.encoder_type = config['encoder_type']
		self.dpout_fc = config['dpout_fc']

		self.encoder = eval(self.encoder_type)(config)
		self.inputdim = 2*self.enc_rnn_dim
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

	def forward(self, expl):
		# expl : ( Variable(T x bs x 300), lens_expl)

		enc_out_expl = self.encoder(expl)
		out_label = self.classifier(enc_out_expl)

		return out_label

	def encode(self, s1):
		emb = self.encoder(s1)
		return emb






