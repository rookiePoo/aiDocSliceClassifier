from keras.models import Sequential, load_model, Model
from keras.layers import Embedding, LSTM, Dropout, Dense, Bidirectional, GRU, Conv1D, MaxPooling1D, Flatten, Input, \
    Concatenate
from keras.callbacks import ModelCheckpoint, TensorBoard
from keras.optimizers import RMSprop, Adam

from config import Config
from keras.utils import plot_model
from data_helper import data_generator
import os



class BiLSTM(object):
    def __init__(self):
        self.sequenceMaxLength = Config.sequenceMaxLength
        self.numChars = Config.numChars
        self.embeddingDim = Config.modelConfig.embeddingDim
        self.lstmOutputDim = Config.modelConfig.lstmOutputDim
        self.checkpointPath = Config.checkpointPath
        self.lstmCheckpointPath = Config.lstmCheckpointPath
        self.charcnnCheckpointPath = Config.charcnnCheckpointPath

        self.logDir = Config.logDir
        self.lr = Config.trainingConfig.learningRate
        # self.build_cnn_model()
        # self.build_bi_lstm_model()
        # self.bilstm_merge_loc_model()
        # self.charcnn_merge_loc_model()

    def build_cnn_model(self, filters=256):
        # Inputs
        char_seq_input = Input(shape=(self.sequenceMaxLength,), dtype='int32', name='char_input')

        # Embedding layers
        embedding_layer = Embedding(self.numChars,
                                    self.embeddingDim,
                                    input_length=self.sequenceMaxLength)
        embedded_sequences = embedding_layer(char_seq_input)

        # conv layers
        convs = []
        filter_sizes = [3, 4, 5]
        for fsz in filter_sizes:
            # 该卷积层的输入为emb
            conv1 = Conv1D(filters, kernel_size=fsz, activation='relu')(embedded_sequences)
            # 将conv1作为下一层Pooling的输入
            pool1 = MaxPooling1D(self.sequenceMaxLength - fsz + 1)(conv1)
            pool1 = Flatten()(pool1)
            convs.append(pool1)
        conv_merge = Concatenate(axis=1)(convs)
        #print(conv_merge.shape)
        dropout = Dropout(0.5)(conv_merge)
        output = Dense(64, activation='relu')(dropout)

        class_pred = Dense(1, activation='sigmoid', name='class_pred')(output)

        self.charcnn_model = Model(inputs=char_seq_input, outputs=class_pred, name='charcnn')
        plot_model(self.charcnn_model, to_file='charcnn_model.png')

    def charcnn_merge_loc_model(self, checkpoint_path=None):
        self.build_cnn_model()
        # char_input = self.charcnn_model.get_layer('char_input')
        class_pred = self.charcnn_model.get_layer('class_pred')

        if checkpoint_path and os.path.exists(checkpoint_path):
            self.charcnn_model.load_weights(checkpoint_path)
            print("Load model weights...")

        loc_input = Input(shape=[3, ], dtype='float32', name='loc_input')

        merge = Concatenate(axis=-1)([loc_input, class_pred.output])
        merge_dense = Dense(128, activation='relu')(merge)
        res_pred = Dense(1, activation='sigmoid', name='res_pred')(merge_dense)
        self.merge_model = Model(inputs=[self.charcnn_model.input, loc_input], outputs=[class_pred.output, res_pred])
        plot_model(self.merge_model, to_file='merge_model.png')

    def build_bi_lstm_model(self):
        char_seq_input = Input(shape=(self.sequenceMaxLength,), dtype='int32', name='char_input')

        embedding_layer = Embedding(self.numChars,
                                    self.embeddingDim,
                                    input_length=self.sequenceMaxLength)
        embedded_sequences = embedding_layer(char_seq_input)
        bilstm = Bidirectional(LSTM(self.lstmOutputDim),
                               input_shape=(self.sequenceMaxLength, self.embeddingDim))
        bilstm_encode = bilstm(embedded_sequences)
        bilstm_encode_drop = Dropout(0.5)(bilstm_encode)
        class_pred = Dense(1, activation='sigmoid', name='class_pred')(bilstm_encode_drop)

        self.bilstm_model = Model(inputs=char_seq_input, outputs=class_pred, name='bilstm')


    def bilstm_merge_loc_model(self):
        self.build_bi_lstm_model()
        # char_input = self.bilstm_model.get_layer('char_input')
        class_pred = self.bilstm_model.get_layer('class_pred')


        plot_model(self.bilstm_model, to_file='bilstm_model.png')
        loc_input = Input(shape=[3, ], dtype='float32', name='loc_input')

        # print(loc_input.shape)
        # print(class_pred.output.shape)

        merge = Concatenate(axis=-1)([loc_input, class_pred.output])
        merge_dense = Dense(128, activation='relu')(merge)
        res_pred = Dense(1, activation='sigmoid', name='res_pred')(merge_dense)
        #res_pred = Dense(1, activation='sigmoid', name='res_pred')(merge)
        self.merge_model = Model(inputs=[self.bilstm_model.input, loc_input], outputs=[class_pred.output, res_pred])

        plot_model(self.merge_model, to_file='merge_model.png')


    def train_lstm(self, excel_dir, checkpoint_path=None):
        train_dict, test_dict = data_generator(excel_dir)
        optimizer = RMSprop(lr=self.lr, rho=0.9, epsilon=1e-06)
        self.bilstm_model.compile(loss='binary_crossentropy',
                                  optimizer=optimizer,
                                  metrics=['accuracy'])

        if checkpoint_path and os.path.exists(checkpoint_path):
            self.bilstm_model.load_weights(checkpoint_path)
            print("Load model weights...")

        checkPointer = ModelCheckpoint(filepath=self.lstmCheckpointPath,
                                       monitor='val_acc',
                                       mode='max',
                                       verbose=1,
                                       save_best_only=True)
        tensorboard = TensorBoard(log_dir=self.logDir)

        x_train = train_dict['char_idx_inputs']
        y_train = train_dict['char_labels']
        x_test = test_dict['char_idx_inputs']
        y_test = test_dict['char_labels']

        self.bilstm_model.fit(x_train, y_train,
                              batch_size=Config.trainingConfig.batchSize,
                              validation_data=[x_test, y_test],
                              epochs=Config.trainingConfig.epoches,
                              callbacks=[checkPointer, tensorboard])

        # self.bilstm_model.save('bilstm.h5')

    def train_charcnn(self, excel_dir, checkpoint_path=None):
        train_dict, test_dict = data_generator(excel_dir)
        optimizer = Adam()
        self.charcnn_model.compile(loss='binary_crossentropy',
                                  optimizer=optimizer,
                                  metrics=['accuracy'])

        if checkpoint_path and os.path.exists(checkpoint_path):
            self.charcnn_model.load_weights(checkpoint_path)
            print("Load model weights...")

        checkPointer = ModelCheckpoint(filepath=self.charcnnCheckpointPath,
                                       monitor='val_acc',
                                       mode='max',
                                       verbose=1,
                                       save_best_only=True)
        tensorboard = TensorBoard(log_dir=self.logDir)

        x_train = train_dict['char_idx_inputs']
        y_train = train_dict['char_labels']
        x_test = test_dict['char_idx_inputs']
        y_test = test_dict['char_labels']

        self.charcnn_model.fit(x_train, y_train,
                              batch_size=Config.trainingConfig.batchSize,
                              validation_data=[x_test, y_test],
                              epochs=Config.trainingConfig.epoches,
                              callbacks=[checkPointer, tensorboard])

        # self.charcnn_model.save('charcnn.h5')

    def train(self, excel_dir):
        train_dict, test_dict = data_generator(excel_dir)

        self.merge_model.compile(optimizer='adam',
                                 loss={'class_pred': 'binary_crossentropy', 'res_pred': 'binary_crossentropy'},
                                 metrics=['accuracy'],
                                 loss_weights={'class_pred': 0.5, 'res_pred': 5.0})


        checkPointer = ModelCheckpoint(filepath=self.checkpointPath,
                                       monitor='val_res_pred_acc',
                                       mode='max',
                                       verbose=1,
                                       save_best_only=True,
                                       save_weights_only=False
                                       #period=5
                                       )
        tensorboard = TensorBoard(log_dir=self.logDir)

        self.merge_model.fit({'char_input': train_dict['char_idx_inputs'], 'loc_input': train_dict['loc_inputs']},
                             {'class_pred': train_dict['char_labels'], 'res_pred': train_dict['res_labels']},
                              epochs=Config.trainingConfig.epoches,
                              batch_size=Config.trainingConfig.batchSize,
                              validation_data=(
                                  {'char_input': test_dict['char_idx_inputs'], 'loc_input': test_dict['loc_inputs']},
                                  {'class_pred': test_dict['char_labels'], 'res_pred': test_dict['res_labels']}),
                              callbacks=[checkPointer, tensorboard])

    def predict_lstm(self, excel_dir, checkpoint_path):
        if checkpoint_path and os.path.exists(checkpoint_path):
            self.bilstm_model.load_weights(checkpoint_path)
            print("Load model weights...")
        train_dict, test_dict = data_generator(excel_dir)
        res = self.bilstm_model.predict_on_batch(test_dict['char_idx_inputs'])
        for i in range(len(res)):
            print(test_dict['char_inputs'][i], res[i], test_dict['char_labels'][i])

    def predict_merge(self, excel_dir, checkpoint_path):
        if checkpoint_path and os.path.exists(checkpoint_path):
            self.merge_model.load_weights(checkpoint_path)
            print("Load model weights...")
        train_dict, test_dict = data_generator(excel_dir)
        res = self.merge_model.predict_on_batch({'char_input': test_dict['char_idx_inputs'], 'loc_input': test_dict['loc_inputs']})
        print(len(res), len(res[0]))
        count = 0
        for i in range(len(res[1])):
            if round(res[1][i][0]) == int(test_dict['res_labels'][i]):
                count += 1
            print(test_dict['char_inputs'][i], res[0][i], round(res[1][i][0]), test_dict['res_labels'][i])
        print(count)

    def load_weights(self, checkpoint_path):
        if checkpoint_path and os.path.exists(checkpoint_path):
            self.merge_model.load_weights(checkpoint_path)
            print("Load model weights...")
    def predict_one(self, char_input, loc_input):

        res = self.merge_model.predict({'char_input': char_input, 'loc_input': loc_input})
        return res



if __name__ == "__main__":
    import numpy as np
    bilstm = BiLSTM()
    excel_dir = '/Users/peng_ji/Desktop/labeled01'

    bilstm.charcnn_merge_loc_model()
    #bilstm.train(excel_dir)
    bilstm.predict_merge(excel_dir, './saved_model/model-weights-55-0.96.hdf5')
    #bilstm.predict_one(np.array([[20]*50]), np.array([[0.1,0.1,0.1]]), 'model-weights-55-0.96.hdf5')


