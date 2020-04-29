import torch

##max feature id 136
##min feature id 1

qddict={}
def qd_feature(fs):
    d={}
    for f in fs:
        kvpair=f.split(':')
        d[int(kvpair[0])]=float(kvpair[1])
    return d

def one_qd(qd_line):
    ss=qd_line.rstrip('\n').split(' ')
    label=int(ss[0])
    qid=int(ss[1][4:])
    feature=qd_feature(ss[2:])
    return qid,label,feature

def to_torch(qddict):
    def to_dense(feats,max_feature_id=136):
        result=torch.zeros(len(feats),max_feature_id)
        for i,fs in enumerate(feats):
            for k,v in fs.items():
                result[i,k-1]=v
        return result
    result={}
    for k,v in qddict.items():
        tlable=torch.tensor(v[0])
        tfeature=to_dense(v[1])
        result[k]=(tlable,tfeature)
    return result

def read_qdfile(filename):
    qddict={}
    with open(filename) as f:
        for qd_line in f:
            qid,label,feature=one_qd(qd_line)
            if qid not in qddict:
                qddict[qid]=([label,],[feature,])
            else:
                qddict[qid][0].append(label)
                qddict[qid][1].append(feature)
    qddict = {k: v for k, v in qddict.items() if len(v[0]) > 1}
    tqddict = to_torch(qddict)
    return tqddict
