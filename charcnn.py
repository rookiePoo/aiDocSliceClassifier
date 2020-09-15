from keras.models import Sequential, load_model, Model
from keras.layers import Embedding, LSTM, Dropout, Dense, Bidirectional, GRU, Conv1D, MaxPooling1D, Flatten, Input, \
    Concatenate
from keras.callbacks import ModelCheckpoint, TensorBoard
from keras.optimizers import RMSprop, Adam

from config import Config
from keras.utils import plot_model
from data_helper import data_generator, data_generator2, my_smote
import os



class CHARCNN(object):
    def __init__(self):
        self.sequenceMaxLength = Config.sequenceMaxLength
        self.numChars = Config.numChars
        self.embeddingDim = Config.modelConfig.embeddingDim
        self.lstmOutputDim = Config.modelConfig.lstmOutputDim
        self.checkpointPath = Config.checkpointPath
        self.charcnnCheckpointPath = Config.charcnnCheckpointPath
        self.filters = Config.modelConfig.cnn_filters
        self.cross_dim = Config.modelConfig.loc_dim + Config.modelConfig.class_dim

        self.logDir = Config.logDir
        self.lr = Config.trainingConfig.learningRate

    def build_cnn_model(self, ):
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
            conv1 = Conv1D(self.filters, kernel_size=fsz, activation='relu')(embedded_sequences)
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

        loc_input = Input(shape=[self.cross_dim, ], dtype='float32', name='loc_input')

        merge = Concatenate(axis=-1)([loc_input, class_pred.output])
        # dropout = Dropout(0.5)(merge)
        # merge_dense = Dense(64, activation='relu')(dropout)
        merge_dense = Dense(512, activation='relu')(merge)
        res_pred = Dense(1, activation='sigmoid', name='res_pred')(merge_dense)
        self.merge_model = Model(inputs=[self.charcnn_model.input, loc_input], outputs=[class_pred.output, res_pred])
        plot_model(self.merge_model, to_file='merge_model.png')

    def train_charcnn(self, excel_dir, checkpoint_path=None):
        train_dict, test_dict = data_generator2(excel_dir)
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

    def train(self, excel_dir, is_enhance_pos = True, initCheckpoint=None):
        train_dict, test_dict = data_generator2(excel_dir, train_rate=0.90)

        if is_enhance_pos:
            train_dict = my_smote(train_dict)

        if initCheckpoint and os.path.exists(initCheckpoint):
            self.merge_model.load_weights(initCheckpoint)
            print("Load model weights...")

        self.merge_model.compile(optimizer='adam',
                                 loss={'class_pred': 'binary_crossentropy', 'res_pred': 'binary_crossentropy'},
                                 metrics=['accuracy'],
                                 loss_weights={'class_pred': 1, 'res_pred': 5})


        checkPointer = ModelCheckpoint(filepath=self.checkpointPath,
                                       monitor='val_res_pred_acc',
                                       mode='max',
                                       verbose=1,
                                       save_best_only=True,
                                       save_weights_only=False
                                       #period=5
                                       )
        tensorboard = TensorBoard(log_dir=self.logDir)
        print(train_dict['loc_inputs'])
        print(train_dict['char_idx_inputs'])

        self.merge_model.fit({'char_input': train_dict['char_idx_inputs'], 'loc_input': train_dict['loc_inputs']},
                             {'class_pred': train_dict['char_labels'], 'res_pred': train_dict['res_labels']},
                              shuffle=True,
                              epochs=Config.trainingConfig.epoches,
                              batch_size=Config.trainingConfig.batchSize,
                              validation_data=(
                                  {'char_input': test_dict['char_idx_inputs'], 'loc_input': test_dict['loc_inputs']},
                                  {'class_pred': test_dict['char_labels'], 'res_pred': test_dict['res_labels']}),
                              callbacks=[checkPointer, tensorboard])


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
    charcnn = CHARCNN()
    #excel_dir = '/Users/peng_ji/codeHub/rookieCode/AB_result'
    excel_dir = '/Users/peng_ji/codeHub/rookieCode/result_train_0804050607_neg'
    #excel_dir = '/Users/peng_ji/codeHub/rookieCode/result_train0804-10_neg'


    #先训练charcnn部分
    charcnn.build_cnn_model()
    #charcnn.train_charcnn(excel_dir)

    # 注释上面👆训练，再训练完整模型
    # bilstm.charcnn_merge_loc_model(checkpoint_path='./saved_model/charcnn-weights-17-0.97.hdf5')
    # bilstm.train(excel_dir)
    #bilstm.predict_merge(excel_dir, './saved_model/modelnopunc-weights-55-0.96.hdf5')
    #bilstm.predict_one(np.array([[20]*50]), np.array([[0.1,0.1,0.1]]), 'modelnopunc-weights-55-0.96.hdf5')

    charcnn.charcnn_merge_loc_model(checkpoint_path=None)
    #charcnn.train(excel_dir)
    charcnn.train(excel_dir, is_enhance_pos=True, initCheckpoint='./saved_model/model-dense08-512-1-5-st4d-neg-weights-01-0.98.hdf5')


