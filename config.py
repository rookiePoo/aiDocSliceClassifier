# 配置参数
import string
import pickle
def load_tokenizer(tokenizer_path):
    with open(tokenizer_path, 'rb') as f:
        print("tokenizer has been loaded.")
        tokenizer = pickle.load(f)
    return tokenizer

class TrainingConfig(object):
    epoches = 128
    batchSize = 64
    evaluateStep = 100
    checkpointSaveStep = 100
    learningRate = 0.001


class ModelConfig(object):
    embeddingDim = 64
    lstmOutputDim = 128  # 单层LSTM结构的神经元个数
    denseOutputDim = 1  # 输出层维度，二分类设置为1，多分类设置为类别的数目
    dropoutKeepProb = 0.5


class Config(object):
    # alphabet = string.ascii_letters
    # digit = '0123456789'
    # whitespace = ' \t\n$'
    # numChars = len(alphabet + digit + whitespace) + 1
    tokenizerPath = './tokenizer/tokenizer.pickle'
    tokenizer = load_tokenizer(tokenizerPath)
    numChars = len(tokenizer.word_index) + 1
    sequenceMaxLength = 50  # 取了所有序列长度的均值

    checkpointPath = "./saved_model/model-weights-{epoch:02d}-{val_res_pred_acc:.2f}.hdf5"
    lstmCheckpointPath = "./saved_model/lstm-weights-{epoch:02d}-{val_acc:.2f}.hdf5"
    charcnnCheckpointPath = "./saved_model/charcnn-weights-{epoch:02d}-{val_acc:.2f}.hdf5"
    logDir = './logs/' #tensorboard --logdir=/full_path_to_your_logs


    numClasses = 1  # 二分类设置为1，多分类设置为类别的数目

    rate = 0.8  # 训练集的比例

    trainingConfig = TrainingConfig()

    modelConfig = ModelConfig()
