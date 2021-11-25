#  False Positive Rate = FP / (FP + TN)
#  Miss Rate = No. of misses / Total no of requests
total = {1: 12, 2: 40, 3: 10}

def fp_rate(real, obs, task):
    tot = total[task]
    result = []
    for i in range(1, tot):
        fp = 0
        tn = 0
        for i in range
