from sklearn.metrics import f1_score, classification_report, accuracy_score

def api_metric(true_labels, pred_labels):
    # 计算并打印分类报告
    print(classification_report(true_labels, pred_labels))
    
    # 计算平均F1（宏平均）
    avg_f1 = f1_score(true_labels, pred_labels, average='macro')
    
    # 计算加权F1
    weighted_f1 = f1_score(true_labels, pred_labels, average='weighted')
    
    # 计算准确率
    acc = accuracy_score(true_labels, pred_labels)
    
    # 返回多个指标
    return {
        'accuracy': acc,
        'avg_f1': avg_f1,
        'weighted_f1': weighted_f1
    }