# 配置参数
import string
import pickle
def load_tokenizer(tokenizer_path):
    with open(tokenizer_path, 'rb') as f:
        print("tokenizer has been loaded.")
        tokenizer = pickle.load(f)
    return tokenizer

FIELD_TYPE_DICT = {"装运": 0,
                   "卸货": 1,
                   "船名": 2,
                   "合同": 3,
                   "LC": 4,
                   "BL": 5,
                   "发票": 6}

class TrainingConfig(object):
    epoches = 100
    batchSize = 64
    evaluateStep = 10
    checkpointSaveStep = 100
    learningRate = 0.001


class ModelConfig(object):
    embeddingDim = 64
    cnn_filters = 256
    lstmOutputDim = 128  # 单层LSTM结构的神经元个数
    denseOutputDim = 1  # 输出层维度，二分类设置为1，多分类设置为类别的数目
    dropoutKeepProb = 0.5
    loc_dim = 11
    class_dim = len(FIELD_TYPE_DICT)+1

class Config(object):
    # alphabet = string.ascii_letters
    # digit = '0123456789'
    # whitespace = ' \t\n$'
    # numChars = len(alphabet + digit + whitespace) + 1

    tokenizerPath = './tokenizer/tokenizer.pickle'
    tokenizer = load_tokenizer(tokenizerPath)
    numChars = len(tokenizer.word_index) + 1
    sequenceMaxLength = 128  # 取了所有序列长度的均值

    checkpointPath = "./saved_model/model-dense08-512-1-5-st4d-neg+ori-weights-{epoch:02d}-{val_res_pred_acc:.2f}.hdf5"
    lstmCheckpointPath = "./saved_model/lstm-weights-{epoch:02d}-{val_acc:.2f}.hdf5"
    charcnnCheckpointPath = "./saved_model/charcnn-weights-{epoch:02d}-{val_acc:.2f}.hdf5"
    logDir = './logs/' #tensorboard --logdir=/full_path_to_your_logs


    numClasses = 1  # 二分类设置为1，多分类设置为类别的数目

    rate = 0.8  # 训练集的比例

    trainingConfig = TrainingConfig()

    modelConfig = ModelConfig()

if __name__ == "__main__":
    tokenizerPath = './tokenizer/tokenizer.pickle'
    tokenizer = load_tokenizer(tokenizerPath)
    numChars = len(tokenizer.word_index) + 1
    print(numChars)
