
import cPickle
import gzip
import os
import sys
sys.setrecursionlimit(6000)
import time

import numpy
import theano
import theano.tensor as T

from logistic_sgd import LogisticRegression
from mlp import HiddenLayer
from WPDefined import Full_Convolayer,Conv_Fold_DynamicK_PoolLayer, dropout_from_layer, shared_dataset, load_training_file, SoftMaxlayer, load_data_for_training
from word2embeddings.nn.layers import BiasedHiddenLayer, SerializationLayer, \
    IndependendAttributesLoss, SquaredErrorLossLayer
from word2embeddings.nn.util import zero_value, random_value_normal, \
    random_value_GloBen10
from word2embeddings.tools.theano_extensions import MRG_RandomStreams2
from cis.deep.utils.theano import debug_print

class CNN_LM(object):
    def __init__(self, learning_rate=0.2, n_epochs=2000, nkerns=[6, 14], batch_size=10, useAllSamples=True, ktop=4, filter_size=[7,5],
                    L2_weight=0.00005, useEmb=0, maxSentLength=60, sentEm_length=48, window=3, 
                    k=5, nce_seeds=2345, only_left_context=False, wait_iter=20, embedding_size=48, newd=[100, 100], train_file_style=1, from_scratch=False, stop=1e-2):
        self.write_file_name_suffix='_lr'+str(learning_rate)+'_nk'+str(nkerns[0])+'&'+str(nkerns[1])+'_bs'+str(batch_size)+'_fs'+str(filter_size[0])+'&'+str(filter_size[1])\
        +'_maxSL'+str(maxSentLength)+'_win'+str(window)+'_noi'+str(k)+'_wait'+str(wait_iter)+'_wdEm'+str(embedding_size)\
        +'_stEm'+str(sentEm_length)+'_ts'+str(from_scratch)+'_newd'+str(newd[0])+'&'+str(newd[1])+'_trFi'+str(train_file_style)+'stop'+str(stop)
        model_options = locals().copy()
        print "model options", model_options
        self.ini_learning_rate=learning_rate
        self.n_epochs=n_epochs
        self.nkerns=nkerns
        self.batch_size=batch_size
        self.useAllSamples=useAllSamples
        
        self.ktop=ktop
        self.filter_size=filter_size
        self.L2_weight=L2_weight
        self.useEmb=useEmb
        self.maxSentLength=maxSentLength
        self.kmax=self.maxSentLength/2+5
        self.sentEm_length=sentEm_length
        self.window=window
        self.k=k
        self.only_left_context=only_left_context
        if self.only_left_context:
            self.context_size=self.window
        else:
            self.context_size=2*self.window
        self.nce_seed=nce_seeds
        self.embedding_size=0
        self.train_file_style=train_file_style
        #we define "train_file_style" as: 0 (wiki), 11(sent_train), 12 (senti_dev), 13 (senti_test)
        
        senti_trainfile="/mounts/data/proj/wenpeng/Dataset/StanfordSentiment/stanfordSentimentTreebank/2classes/2train.txt"
        senti_devfile="/mounts/data/proj/wenpeng/Dataset/StanfordSentiment/stanfordSentimentTreebank/2classes/2dev.txt"
        senti_testfile="/mounts/data/proj/wenpeng/Dataset/StanfordSentiment/stanfordSentimentTreebank/2classes/2test.txt"
        wiki_path="/mounts/data/proj/wenpeng/PhraseEmbedding/enwiki-20130503-pages-articles-cleaned-tokenized"
        embeddingPath='/mounts/data/proj/wenpeng/Downloads/hlbl-embeddings-original.EMBEDDING_SIZE=50.txt'
        embeddingPath2='/mounts/data/proj/wenpeng/MC/src/released_embedding.txt'
        root='/mounts/data/proj/wenpeng/Thang/'
        if self.train_file_style !=0:
            datasets, unigram, train_lengths, word_count, self.id2word=load_training_file(senti_trainfile,self.maxSentLength,self.train_file_style)
        elif self.train_file_style == 0:
            #datasets, unigram, train_lengths, word_count, self.id2word=load_training_file(root+'train.txt',self.maxSentLength,self.train_file_style)
            datasets, unigram, train_lengths, dev_lengths, word_count, self.id2word=load_data_for_training(root+'train.txt', root+'test_eval92-93.txt',self.maxSentLength)
        

        
        self.unigram=unigram # we use the average of unigram as probability of new word in dev set
        self.datasets=datasets
        self.embedding_size=embedding_size
        self.vocab_size=word_count
        self.rand_values_R=random_value_normal((self.vocab_size+1, self.embedding_size), theano.config.floatX, numpy.random.RandomState(1234))
        self.rand_values_R[0]=numpy.array(numpy.zeros(self.embedding_size))
                                                           
        self.rand_values_Q=random_value_normal((self.vocab_size+1, self.embedding_size), theano.config.floatX, numpy.random.RandomState(4321))
        self.rand_values_Q[0]=numpy.array(numpy.zeros(self.embedding_size))
          
        self.from_scratch=from_scratch
        if not self.from_scratch:
            self.load_pretrained_embeddings_and_w2v50d()
        self.embeddings_R=theano.shared(value=self.rand_values_R) 
        self.embeddings_Q=theano.shared(value=self.rand_values_Q) 
        
        self.extend_unigram=numpy.append(unigram, [sum(unigram)/len(unigram)])
        #print 'unigram, p_n length:', len(unigram), len(self.extend_unigram)
        self.p_n=theano.shared(value=self.extend_unigram)
        self.train_lengths=train_lengths
        self.vali_lengths=dev_lengths
        b_values = zero_value((len(unigram)+1,), dtype=theano.config.floatX)#the last bias is for new words in dev data
        #print 'bias length:', len(b_values)
        self.bias = theano.shared(value=b_values, name='bias')
        self.wait_iter=wait_iter
        self.newd=newd
        self.stop=stop
        

    def load_pretrained_embeddings_and_w2v50d(self):
        
        word2embeddings_R={}
        word2embeddings_Q={}
        #read_file=open('/mounts/data/proj/wenpeng/Sent2Vec/context_target_embeddings.txt')  # should be changed according to...
        read_file=open('/mounts/data/proj/wenpeng/Thang/context_target_embeddings_lr0.005_nk1&1_bs50_fs7&5_maxSL60_win5_noi10_wait10_wdEm50_stEm50_tsTrue_newd50&50_trFi0stop0.001.txt')
        for line in read_file:
            tokens=line.strip().split('\t')
            word2embeddings_R[tokens[0]]=tokens[1].split() #transfer str list into float list
            word2embeddings_Q[tokens[0]]=tokens[2].split()
        print 'Pretrained embeddings loaded over.'
        read_file.close()
        #load 50d word2vec
        read_file=open('/mounts/data/proj/wenpeng/Dataset/word2vec_50d_Heike.txt')
        word2emb={}
        for line in read_file:
            emb=[]
            tokens=line.strip().split()
            if len(tokens)>2:
                for i in range(50):
                    emb.append(float(tokens[i+1]))
                word2emb[tokens[0]]=emb
        read_file.close()
        print 'word2vec 50d loaded over.'
        
        words_number=len(self.id2word)
        training_size=len(self.unigram)
        print 'Totally, ', words_number, ' words, ', training_size, ' in training data.'
        for index in range(1, words_number+1):
            if index <= training_size:
                embed_R=word2embeddings_R.get(self.id2word[index], -1)
                embed_Q=word2embeddings_Q.get(self.id2word[index], -1)
                if embed_R!=-1 and embed_Q!=-1:
                    self.rand_values_R[index]=numpy.array(embed_R)
                    #T.set_subtensor(self.embeddings_R[index], theano.shared(value=numpy.array(embed_R), dtype=theano.config.floatX))
                    self.rand_values_Q[index]=numpy.array(embed_Q)
                    #T.set_subtensor(self.embeddings_Q[index], theano.shared(value=numpy.array(embed_Q), dtype=theano.config.floatX))
                else:
                    print 'error: word '+self.id2word[index]+' doesnt find embedding in pretrained set.'
            else: #unknown words in test set
                emb=word2emb.get(self.id2word[index], -1)
                if emb!=-1:
                    self.rand_values_R[index]=numpy.array(emb)
                    #T.set_subtensor(self.embeddings_R[index], theano.shared(value=numpy.array(embed_R), dtype=theano.config.floatX))
                    self.rand_values_Q[index]=numpy.array(emb) 
                else:
                    self.rand_values_R[index]=numpy.array(numpy.random.rand(self.embedding_size))
                    #T.set_subtensor(self.embeddings_R[index], theano.shared(value=numpy.array(embed_R), dtype=theano.config.floatX))
                    self.rand_values_Q[index]=numpy.array(numpy.random.rand(self.embedding_size))                                 
            
        print 'Embedding initialize over.'
        #exit(0)
   
   
    def get_noise(self):
            # Create unigram noise distribution.
        srng = MRG_RandomStreams2(seed=self.nce_seed)
    
        # Get the indices of the noise samples.
        random_noise = srng.multinomial(size=(self.batch_size, self.k), pvals=self.unigram)
        #random_noise=theano.printing.Print('random_noise')(random_noise)
        noise_indices_flat = random_noise.reshape((self.batch_size * self.k,))
        p_n_noise = self.p_n[noise_indices_flat].reshape((self.batch_size, self.k))
        return random_noise+1, p_n_noise   # for word index starts from 1 in our embedding matrix
    
    def concatenate_sent_context(self,sent_matrix, context_matrix):
        return T.concatenate([sent_matrix, context_matrix], axis=1)
        
    def calc_r_h(self, h_indices):
        return self.embed_context(h_indices)
    

    
    def embed_context(self,indices):
        #indices is a matrix with (batch_size, context_size)
        embedded=self.embed_word_indices(indices, self.embeddings_R)
        '''
        flattened_embedded=embedded.flatten()
        batch_size=indices.shape[0]
        context_size=indices.shape[1]
        embedding_size=self.embeddings_R.shape[1]
        '''
        #return embedded.reshape((self.batch_size, self.context_size*self.context_embedding_size))
        #now, we change it to satisfy HS_simplified.py
        tensor_embed=embedded.reshape((self.batch_size, self.context_size, self.embedding_size))
        tensor_sum=T.sum(tensor_embed, axis=1)
        return tensor_sum.reshape((self.batch_size, self.embedding_size)) #now, this matrix can be added directly to the sentence embedding matrix
    def embed_noise(self, indices):
        embedded=self.embed_word_indices(indices, self.embeddings_Q)
        '''
        flattened_embedded=embedded.flatten()
        return flattened_embedded.reshape((self.batch_size, self.k, self.embedding_size ))  
        '''
        return embedded.reshape((self.batch_size, self.k, self.embedding_size ))
    def embed_target(self,indices):
        embedded=self.embed_word_indices(indices, self.embeddings_Q)
        return embedded.reshape((self.batch_size, self.embedding_size ))       
    def embed_word_indices(self, indices, embeddings):
        indices2vector=indices.flatten()
        #return a matrix
        return embeddings[indices2vector]
    def extract_contexts_targets(self, indices_matrix, sentLengths, leftPad):
        #first pad indices_matrix with zero indices on both side
        left_padding = T.zeros((indices_matrix.shape[0], self.window), dtype=theano.config.floatX)
        right_padding = T.zeros((indices_matrix.shape[0], self.window), dtype=theano.config.floatX)
        matrix_padded = T.concatenate([left_padding, indices_matrix, right_padding], axis=1)  
        
        leftPad=leftPad+self.window   #a vector plus a number
           
        # x, y indices
        max_length=T.max(sentLengths)
        x=T.repeat(T.arange(self.batch_size), max_length)
        y=[]
        for row in range(self.batch_size):
            y.append(T.repeat((T.arange(leftPad[row], leftPad[row]+sentLengths[row]),), max_length, axis=0).flatten()[:max_length])
        y=T.concatenate(y, axis=0)   
        #construct xx, yy for context matrix
        context_x=T.repeat(T.arange(self.batch_size), max_length*self.context_size)
        #wenpeng=theano.printing.Print('context_x')(context_x)
        context_y=[]
        for i in range(self.window, 0, -1): # first consider left window
            context_y.append(y-i)
        if not self.only_left_context:
            for i in range(self.window): # first consider left window
                context_y.append(y+i+1)
        context_y_list=T.concatenate(context_y, axis=0)       
        new_shape = T.cast(T.join(0, 
                               T.as_tensor([self.context_size]),
                               T.as_tensor([self.batch_size*max_length])),
                               'int64')
        context_y_vector=T.reshape(context_y_list, new_shape, ndim=2).transpose().flatten()
        new_shape = T.cast(T.join(0, 
                               T.as_tensor([self.batch_size]),
                               T.as_tensor([self.context_size*max_length])),
                               'int64')
        
        context_matrix = T.reshape(matrix_padded[context_x,context_y_vector], new_shape, ndim=2)  
        new_shape = T.cast(T.join(0, 
                               T.as_tensor([self.batch_size]),
                               T.as_tensor([max_length])),
                               'int64') 
        target_matrix = T.reshape(matrix_padded[x,y], new_shape, ndim=2)
        return    T.cast(context_matrix, 'int64'),  T.cast(target_matrix, 'int64')
    def store_model_to_file(self):
        save_file = open('/mounts/data/proj/wenpeng/Thang/model_params'+self.write_file_name_suffix, 'wb')  # this will overwrite current contents
        for para in self.best_params:           
            cPickle.dump(para.get_value(borrow=True), save_file, -1)  # the -1 is for HIGHEST_PROTOCOL
        save_file.close()
    def load_model_from_file(self):
        save_file = open('/mounts/data/proj/wenpeng/Thang/model_params_lr0.005_nk1&1_bs50_fs7&5_maxSL60_win5_noi10_wait10_wdEm50_stEm50_tsTrue_newd50&50_trFi0stop0.001')
        for para in self.params_to_store: #note that this doesnt include context and target embeddings
            para.set_value(cPickle.load(save_file), borrow=True)
        print 'Model params loaded over.'
        save_file.close()
    
    def evaluate_lenet5(self):
    #def evaluate_lenet5(learning_rate=0.1, n_epochs=2000, nkerns=[6, 12], batch_size=70, useAllSamples=0, kmax=30, ktop=5, filter_size=[10,7],
    #                    L2_weight=0.000005, dropout_p=0.5, useEmb=0, task=5, corpus=1):
        rng = numpy.random.RandomState(23455)
        
        #datasets, embedding_size, embeddings=read_data(root+'2classes/train.txt', root+'2classes/dev.txt', root+'2classes/test.txt', embeddingPath,60)

        #datasets = load_data(dataset)
        indices_train, trainLengths, trainLeftPad, trainRightPad= self.datasets[0]
        indices_dev, devLengths, devLeftPad, devRightPad= self.datasets[1]  #use itself as validation set

        #create embedding matrix to store the final embeddings
        sentences_embs=numpy.zeros((indices_train.shape[0],self.sentEm_length), dtype=theano.config.floatX)

        n_train_batches=indices_train.shape[0]/self.batch_size
        n_valid_batches=indices_dev.shape[0]/self.batch_size
        remain_train=indices_train.shape[0]%self.batch_size
        
        train_batch_start=[]
        dev_batch_start=[]
        if self.useAllSamples:
            train_batch_start=list(numpy.arange(n_train_batches)*self.batch_size)+[indices_train.shape[0]-self.batch_size]
            dev_batch_start=list(numpy.arange(n_valid_batches)*self.batch_size)+[indices_dev.shape[0]-self.batch_size]
            n_train_batches=n_train_batches+1
            n_valid_batches=n_valid_batches+1
        else:
            train_batch_start=list(numpy.arange(n_train_batches)*self.batch_size)
            dev_batch_start=list(numpy.arange(n_valid_batches)*self.batch_size)
    
        indices_train_theano=theano.shared(numpy.asarray(indices_train, dtype=theano.config.floatX), borrow=True)
        indices_dev_theano=theano.shared(numpy.asarray(indices_dev, dtype=theano.config.floatX), borrow=True)
        indices_train_theano=T.cast(indices_train_theano, 'int32')
        indices_dev_theano=T.cast(indices_dev_theano, 'int32')
        
        # allocate symbolic variables for the data
        index = T.lscalar()  # index to a [mini]batch
        x_index = T.imatrix('x_index')   # now, x is the index matrix, must be integer
        #y = T.ivector('y')  
        z = T.ivector('z')   # sentence length
        left=T.ivector('left')
        right=T.ivector('right')
        iteration= T.lscalar()
        
        #newd=(6, 7)
        x=self.embeddings_R[x_index.flatten()].reshape((self.batch_size,self.maxSentLength, self.embedding_size)).transpose(0, 2, 1).flatten()
        ishape = (self.embedding_size, self.maxSentLength)  # this is the size of MNIST images
        filter_size1=(self.embedding_size,self.filter_size[0])
        filter_size2=(self.newd[0],self.filter_size[1])
        poolsize1=(1, ishape[1]-filter_size1[1]+1) #?????????????????????????????
        #poolsize1=(1, ishape[1]+filter_size1[1]-1)
        
        #now, we use valid convolution, so, it's -filter_width+1
        left_after_conv=T.maximum(0,left-filter_size1[1]+1)
        right_after_conv=T.maximum(0, right-filter_size1[1]+1)
        

        
        #kmax=30 # this can not be too small, like 20
        #ktop=6
        poolsize2=(1, self.kmax-filter_size2[1]+1) #(1,6)
        #poolsize2=(1, self.kmax+filter_size2[1]-1) #(1,6)
        dynamic_lengths=T.maximum(4,z/2+1)  # dynamic k-max pooling
        
        ######################
        # BUILD ACTUAL MODEL #
        ######################
        print '... building the model'
    
        # Reshape matrix of rasterized images of shape (batch_size,28*28)
        # to a 4D tensor, compatible with our LeNetConvPoolLayer
        layer0_input = debug_print(x.reshape((self.batch_size, 1, ishape[0], ishape[1])), 'layer0_input')

        # Construct the first convolutional pooling layer:
        # filtering reduces the image size to (28-5+1,28-5+1)=(24,24)
        # maxpooling reduces this further to (24/2,24/2) = (12,12)
        # 4D output tensor is thus of shape (batch_size,nkerns[0],12,12)
        layer0 = Full_Convolayer(rng, input=layer0_input,
                image_shape=(self.batch_size, 1, ishape[0], ishape[1]),                 
                filter_shape=(self.nkerns[0], 1, filter_size1[0], filter_size1[1]), poolsize=poolsize1, k=dynamic_lengths, unifiedWidth=self.kmax, left=left_after_conv, right=right_after_conv, newd=self.newd[0],firstLayer=True)

        # Construct the second convolutional pooling layer
        # filtering reduces the image size to (12-5+1,12-5+1)=(8,8)
        # maxpooling reduces this further to (8/2,8/2) = (4,4)
        # 4D output tensor is thus of shape (nkerns[0],nkerns[1],4,4)
        '''
        layer1 = LeNetConvPoolLayer(rng, input=layer0.output,
                image_shape=(batch_size, nkerns[0], ishape[0], kmax),
                filter_shape=(nkerns[1], nkerns[0], filter_size2[0], filter_size2[1]), poolsize=poolsize2, k=ktop)
        '''
        
        left_after_conv=T.maximum(0, layer0.leftPad-filter_size2[1]+1)
        right_after_conv=T.maximum(0, layer0.rightPad-filter_size2[1]+1)
        
        '''
        left_after_conv=layer0.leftPad
        right_after_conv=layer0.rightPad
        '''
        dynamic_lengths=T.repeat([self.ktop],self.batch_size)  # dynamic k-max pooling
        '''
        layer1 = ConvFoldPoolLayer(rng, input=layer0.output,
                image_shape=(batch_size, nkerns[0], ishape[0]/2, kmax),
                filter_shape=(nkerns[1], nkerns[0], filter_size2[0], filter_size2[1]), poolsize=poolsize2, k=ktop, left=left_after_conv, right=right_after_conv)
        '''
        layer1 = Full_Convolayer(rng, input=layer0.output,
                image_shape=(self.batch_size, self.nkerns[0], self.newd[0], self.kmax), newd=self.newd[1],
                filter_shape=(self.nkerns[1], self.nkerns[0], filter_size2[0], filter_size2[1]), poolsize=poolsize2, k=dynamic_lengths, unifiedWidth=self.ktop, left=left_after_conv, right=right_after_conv, firstLayer=False)    
        
        # the HiddenLayer being fully-connected, it operates on 2D matrices of
        # shape (batch_size,num_pixels) (i.e matrix of rasterized images).
        # This will generate a matrix of shape (20,32*4*4) = (20,512)
        
        
        layer2_input = debug_print(layer1.output.flatten(2), 'layer2_input')
        #layer2_input=theano.printing.Print('layer2_input')(layer2_input)
        #produce sentence embeddings, then add up with context embeddings
        #layer2 = HiddenLayer(rng, input=layer2_input, n_in=self.nkerns[1] * self.newd[1] * self.ktop, n_out=self.sentEm_length, activation=T.tanh)
        layer2_output=layer2_input
        context_matrix,  target_matrix=self.extract_contexts_targets(indices_matrix=x_index, sentLengths=z, leftPad=left)
        #note that context indices might be zero embeddings
        h_indices=debug_print(context_matrix[:, self.context_size*iteration:self.context_size*(iteration+1)],'h_indices')
        w_indices=debug_print(target_matrix[:, iteration:(iteration+1)],'w_indices')
        #r_h is the concatenation of context embeddings
        r_h=self.embed_context(h_indices)  #(batch_size, context_size*embedding_size)
        q_w=self.embed_target(w_indices)
        #q_hat: concatenate sentence embeddings and context embeddings
        '''
        q_hat=self.concatenate_sent_context(layer2.output, r_h)
        layer3 = HiddenLayer(rng, input=q_hat, n_in=self.sentEm_length+self.context_size*self.embedding_size, n_out=self.embedding_size, activation=T.tanh)
        '''
        layer3_output=layer2_output+r_h
        
        noise_indices, p_n_noise=self.get_noise()
        #noise_indices=theano.printing.Print('noise_indices')(noise_indices)
        s_theta_data=T.sum(layer3_output * q_w, axis=1).reshape((self.batch_size,1)) + self.bias[T.minimum(w_indices-1, len(self.unigram))]  #bias[0] should be the bias of word index 1
        #s_theta_data=theano.printing.Print('s_theta_data')(s_theta_data)
        p_n_data = self.p_n[T.minimum(w_indices-1, len(self.unigram))] #p_n[0] indicates the probability of word indexed 1
        delta_s_theta_data = s_theta_data - T.log(self.k * p_n_data)
        log_sigm_data = T.log(T.nnet.sigmoid(delta_s_theta_data))
        
        #create the noise, q_noise has shape(self.batch_size, self.k, self.embedding_size )
        q_noise = self.embed_noise(noise_indices)
        q_hat_res = layer3_output.reshape((self.batch_size, 1, self.embedding_size))
        s_theta_noise = T.sum(q_hat_res * q_noise, axis=2) + self.bias[noise_indices-1] #(batch_size, k)
        delta_s_theta_noise = s_theta_noise - T.log(self.k * p_n_noise)  # it should be matrix (batch_size, k)
        log_sigm_noise = T.log(1 - T.nnet.sigmoid(delta_s_theta_noise))
        sum_noise_per_example =T.sum(log_sigm_noise, axis=1)   #(batch_size, 1)
        # Calc objective function
        J = -T.mean(log_sigm_data) - T.mean(sum_noise_per_example)
        L2_reg = (layer1.W** 2).sum()+(layer0.W** 2).sum()+(self.embeddings_R**2).sum()+( self.embeddings_Q**2).sum()#+(self.bias**2).sum()
        self.cost = J + self.L2_weight*L2_reg
        
        validate_model = theano.function([index,iteration], [self.cost],
                givens={
                    x_index: indices_dev_theano[index: index + self.batch_size],
                    z: devLengths[index: index + self.batch_size],
                    left: devLeftPad[index: index + self.batch_size],
                    right: devRightPad[index: index + self.batch_size]})
        
        # create a list of all model parameters to be fit by gradient descent
        self.params = layer1.params + layer0.params+[self.embeddings_R, self.embeddings_Q]
        self.params_to_store = layer1.params + layer0.params
        #params = layer3.params + layer2.params + layer0.params+[embeddings]
        
        accumulator=[]
        for para_i in self.params:
            eps_p=numpy.zeros_like(para_i.get_value(borrow=True),dtype=theano.config.floatX)
            accumulator.append(theano.shared(eps_p, borrow=True))
          
        # create a list of gradients for all model parameters
        grads = T.grad(self.cost, self.params)
        updates = []
        for param_i, grad_i, acc_i in zip(self.params, grads, accumulator):
            acc = acc_i + T.sqr(grad_i)
            if param_i == self.embeddings_R or param_i == self.embeddings_Q:
                updates.append((param_i, T.set_subtensor((param_i - self.ini_learning_rate * grad_i / T.sqrt(acc))[0], theano.shared(numpy.zeros(self.embedding_size)))))   #AdaGrad
            else:
                updates.append((param_i, param_i - self.ini_learning_rate * grad_i / T.sqrt(acc)))   #AdaGrad
            updates.append((acc_i, acc))    
           
        train_model = theano.function([index,iteration], self.cost, updates=updates,
              givens={
                x_index: indices_train_theano[index: index + self.batch_size],
                z: trainLengths[index: index + self.batch_size],
                left: trainLeftPad[index: index + self.batch_size],
                right: trainRightPad[index: index + self.batch_size]})
    
        ###############
        # TRAIN MODEL #
        ###############
        print '... predicting'
        if not self.from_scratch:
            self.load_model_from_file()#reset the model parameters
        

        validation_losses=[]           
        for batch_start in dev_batch_start:
            total_iteration=max(self.vali_lengths[batch_start: batch_start + self.batch_size])
            for iteration in range(total_iteration):
                vali_loss_i=validate_model(batch_start, iteration)
                validation_losses.append(vali_loss_i)       
        this_validation_loss = numpy.mean(validation_losses)
        print 'test loss: ', this_validation_loss


    def store_sentence_embeddings(self):
        save_file = open('/mounts/data/proj/wenpeng/Thang/sentence_embeddings'+self.write_file_name_suffix+'.txt', 'w')  # this will overwrite current contents # this will overwrite current contents
        rows=self.best_sentence_embs.shape[0]
        cols=self.best_sentence_embs.shape[1]
        for row in range(rows):
            for col in range(cols):
                save_file.write(str(self.best_sentence_embs[row, col])+" ")
            save_file.write("\n")
        save_file.close() 
        print 'Sentence embeddings stored over.'
    
    def store_context_target_embeddings(self):
        save_file=open('/mounts/data/proj/wenpeng/Thang/context_target_embeddings'+self.write_file_name_suffix+'.txt', 'w')
        fake_word_count=len(self.id2word)+1
        embeddings_R=self.best_embeddings_R.get_value()
        embeddings_Q=self.best_embeddings_Q.get_value()
        for id in range(1, fake_word_count):
            save_file.write(self.id2word[id]+'\t')
            for col in range(self.embedding_size-1):
                save_file.write(str(embeddings_R[id][col])+' ')
            save_file.write(str(embeddings_R[id][self.embedding_size-1])+'\t')
            for col in range(self.embedding_size-1):
                save_file.write(str(embeddings_Q[id][col])+' ')
            save_file.write(str(embeddings_Q[id][self.embedding_size-1])+'\n')
        print 'Storing context & target embeddings over.'
        save_file.close()
        
        

def minimal_of_list(list_of_ele):
    if len(list_of_ele) ==0:
        return 1e10
    else:
        return min(list_of_ele)

   
    

if __name__ == '__main__':

    #train_file_style: 1 senti training; 2 wiki
    network=CNN_LM(learning_rate=0.005, nkerns=[1, 1], batch_size=50, ktop=1, filter_size=[7,5],
                    L2_weight=0.00005, maxSentLength=60, sentEm_length=50, window=5, 
                    k=10, only_left_context=True, wait_iter=10, embedding_size=50, newd=[50, 50], train_file_style=0, from_scratch=False, stop=1e-3)
    network.evaluate_lenet5()

