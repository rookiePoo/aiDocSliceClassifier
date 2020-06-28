from sklearn.metrics import roc_auc_score, accuracy_score, precision_score, recall_score

def pred_probability2label(proba_list, threshold=0.5):
    pred_label_list = []
    for proba in proba_list:
        if proba >= threshold:
            pred_label_list.append(1)
        else:
            pred_label_list.append(0)
    return pred_label_list


def genMetrics(trueY, binaryPredY):
    """
    生成acc和auc值
    """
    auc = roc_auc_score(trueY, binaryPredY)
    accuracy = accuracy_score(trueY, binaryPredY)
    precision = precision_score(trueY, binaryPredY, average='macro')
    recall = recall_score(trueY, binaryPredY, average='macro')

    return round(accuracy, 4), round(auc, 4), round(precision, 4), round(recall, 4)