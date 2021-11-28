#  False Positive Rate = FP / (FP + TN)
#  Miss Rate = FNR = False negatives / (False negatives + True positives) 
import utilities
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
        t = 0
        ci = 0
        for j in range(len(real)):
            if val != real[j] and val != obs[j]:
                tn += 1
            elif val == obs[j] and val != real[j]:
                fp += 1
            elif val == obs[j] and val == real[j]:
                tp += 1
            else:
                fn += 1
        #         continue

        
        # False Positive Rate = #(images_considered - correct_images)/t
        # miss rate = (t - #correct_images)/t

        # print(tp, fp, fn, tn)
        fpr_1 = fp/(fp+tn)
        # fnr_1 = 1 - (tp+tn)/tot
        # fnr_1 = fn / (fn+tp)
        if fn == 0 and tp == 0:
            fnr_1 = 1.0
        else:
            fnr_1 = fn/(fn+tp)
        fpr_1 = round(fpr_1, 2)
        fnr_1 = round(fnr_1, 2)
        fpr.append(fpr_1)
        fnr.append(fnr_1)
    return fpr, fnr
