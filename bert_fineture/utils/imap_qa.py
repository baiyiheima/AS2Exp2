def calc_one_map(data):
    relcnt = 0
    score = 0.0
    data = sorted(data, key=lambda d: d[1], reverse=True)
    for idx, item in enumerate(data):
        if float(item[0][2]) == 1:
            relcnt = relcnt + 1
            score = score + 1.0 * relcnt / (idx + 1)
    if relcnt == 0:
        return 0
    return score / relcnt

def calc_one_mrr(data):
    score = 0
    data = sorted(data, key=lambda d: d[1], reverse=True)
    for idx, item in enumerate(data):
        if float(item[0][2]) == 1:
            score = 1.0 / (idx + 1)
            break
    return score

def calc_map1(testfile, preds):
        pred = []
        data=testfile
        for p in preds:
            pred.append(float(p))
        oneq = []
        pre = "BEGIN"
        mapscore = 0.0
        excnt = 0
        for item in zip(data, pred):
            if item[0][0] == pre or pre == "BEGIN":
                oneq.append(item)
            else :
                excnt = excnt + 1
                sc = calc_one_map(oneq)
                if sc == 0:
                    excnt = excnt - 1
                mapscore = mapscore + sc
                oneq = []
                oneq.append(item)
            pre = item[0][0]
        sc = calc_one_map(oneq)
        if sc != 0:
            excnt = excnt + 1
        mapscore = mapscore + sc
        return mapscore / excnt

def calc_mrr1(testfile, preds):
        pred = []
        data = testfile
        for p in preds:
            pred.append(float(p))
        oneq = []
        pre = "BEGIN"
        mrrscore = 0.0
        excnt = 0
        for item in zip(data, pred):

            if item[0][0] == pre or pre == "BEGIN":
                oneq.append(item)
            else :
                excnt = excnt + 1
                sc = calc_one_mrr(oneq)
#                 print sc,
                if sc == 0:
                    excnt = excnt - 1
                mrrscore = mrrscore + sc
                oneq = []
                oneq.append(item)
            pre = item[0][0]

        sc = calc_one_mrr(oneq)
        if sc != 0:
            excnt = excnt + 1
        mrrscore = mrrscore + sc
        return mrrscore / excnt