def OfflineEval(predict_path, result_path):
    count = 0
    pre_text = {}
    with open(predict_path) as f:
        for text in f:
            data = eval(text)
            textstr = data['text']

            count += 1
            pre_text[textstr] = set()
            for i in range(len(data['spo_list'])):
                subject_type = data['spo_list'][i]['subject_type']
                predicate = data['spo_list'][i]['predicate']
                object_type = data['spo_list'][i]['object_type']
                spo = subject_type + '|' + predicate + '|' + object_type
                subj = data['spo_list'][i]['subject']
                obj = data['spo_list'][i]['object']
                k = str(subj) + ' ' + str(spo) + ' ' + str(obj)
                pre_text[textstr].add(k)

    sumcount = 0
    r = 0
    s = 0
    p = 0
    with open(result_path) as f:
        for text in f:
            data = eval(text)
            textstr = data['text']
            sumcount += 1
            actual = set()

            for i in range(len(data['spo_list'])):
                subject_type = data['spo_list'][i]['subject_type']
                predicate = data['spo_list'][i]['predicate']
                object_type = data['spo_list'][i]['object_type']
                spo = subject_type + '|' + predicate + '|' + object_type
                subj = data['spo_list'][i]['subject']
                obj = data['spo_list'][i]['object']
                k = subj + ' ' + spo + ' ' + obj
                actual.add(k)
                r += 1

            if (textstr in pre_text):
                pset = pre_text[textstr]
                rset = actual
                p += len(pset)

                s += len(pset.intersection(rset))

    recall = s / r
    precise = s / p
    f1_base = 2 * recall * precise / (recall + precise)

    print('precise: ', precise)
    print('recall: ', recall)
    print('f1: ', f1_base)




if __name__ == '__main__':
    # 预测文件路径
    predict_path = "data/test_demo.res"
    # predict_path = "data/submit1.json"
    # 答案路径
    result_path = "/home/rentc/project/比赛/2019CCF/信息抽取/data/dev_data.json"
    OfflineEval(predict_path, result_path)
