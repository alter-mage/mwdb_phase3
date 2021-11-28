#  False Positive Rate = FP / (FP + TN)
#  Miss Rate = FNR = False negatives / (False negatives + True positives) 
total = {1: 12, 2: 40, 3: 10}

def metrics(real, obs, task):
    '''
    Input: Real Label Data Matrix, Predicted Label Data Matrix, Task Number

    Output: False Positive Rate Matrix, False Negative Rate Matrix
    '''
    tot = total[task] + 1
    fpr, fnr = [], []
    for val in range(1, tot):
        fp, tp, fn, tn = 0, 0, 0, 0
        for j in range(len(real)):
            if val != real and val != obs:
                tn += 1
            elif val == obs and val != real:
                fp += 1
            elif val == obs and val == real:
                tp += 1
            else:
                fn += 1
                continue
        fpr_1 = fp/(fp+tn)
        fnr_1 = fn/(fn+tp)
        fpr.append(fpr_1)
        fnr.append(fnr_1)
    for i in range(1, tot):
        print(fpr[i], fnr[i])
    return fpr, fnr
