import pandas as pd
import numpy as np
from tkinter import _flatten
from sklearn.metrics import roc_auc_score
from pyod.utils.utility import precision_n_scores
from pyod.utils.data import evaluate_print
from pyod.utils.example import visualize
import os
import warnings
warnings.filterwarnings("ignore")
from pyod.models.pca import PCA
from pyod.models.mcd import MCD
from pyod.models.lof import LOF
from pyod.models.knn import KNN
from pyod.models.loda import LODA
from sklearn import decomposition

def train(doc_list,dataset_name,clf_name):
    model_roc=[]
    model_prc=[]
    if clf_name=="PCA":
        clf = PCA()
    elif clf_name=="MCD":
        clf = MCD()
    elif clf_name=="LOF":
        clf = LOF()
    elif clf_name=="KNN":
        clf = KNN()
    elif clf_name=="LODA":
        clf = LODA()
    for i in range(10): 
        data = pd.read_csv(doc_list[i], header=0, index_col=0)
        train_x = data.drop(drop+ground_truth, axis=1).values
        train_y=np.array([transfor[x] for x in list(_flatten(data[ground_truth].values.tolist()))])
        clf.fit(train_x)
        predict=clf.decision_scores_
        roc=roc_auc_score(train_y, predict)
        prc=precision_n_scores(train_y, predict)
        if((i+1)%200==0):
            print("第"+str(i+1)+"个文件结果:")
            evaluate_print(clf_name, train_y, predict)
        model_roc.append(roc)
        model_prc.append(prc)
    model_roc_avg = np.mean(model_roc)
    model_prc_avg = np.mean(model_prc)
    print("模型"+clf_name+"在数据集"+dataset_name+"的平均roc_auc为"+str(round(model_roc_avg,4))+",平均prc为"+str(round(model_prc_avg,4))+"。")  

    return model_roc_avg,model_prc_avg

#处理数据集pageb
doc_list=[]
for root,dirs,files in os.walk(r"D:\研一下学期\数据挖掘\作业4\pageb\benchmarks"):
    for file in files:
        doc_list.append(os.path.join(root,file))
        
transfor={'nominal':0,'anomaly':1} 
drop=['motherset','origin']
ground_truth=['ground.truth']

model1_roc_avg,model1_prc_avg=train(doc_list,"pageb","PCA")
model2_roc_avg,model2_prc_avg=train(doc_list,"pageb","MCD")
model3_roc_avg,model3_prc_avg=train(doc_list,"pageb","LOF")
model4_roc_avg,model4_prc_avg=train(doc_list,"pageb","KNN")
model5_roc_avg,model5_prc_avg=train(doc_list,"pageb","LODA")

dataset1_result_roc=[model1_roc_avg,model2_roc_avg,model3_roc_avg,model4_roc_avg,model5_roc_avg]
dataset1_result_prc=[model1_prc_avg,model2_prc_avg,model3_prc_avg,model4_prc_avg,model5_prc_avg]

#处理数据集abalone
doc_list=[]
for root,dirs,files in os.walk(r"D:\研一下学期\数据挖掘\作业4\abalone\benchmarks"):
    for file in files:
        doc_list.append(os.path.join(root,file))
        
transfor={'nominal':0,'anomaly':1} 
drop=['motherset','origin']
ground_truth=['ground.truth']

model1_roc_avg,model1_prc_avg=train(doc_list,"abalone","PCA")
model2_roc_avg,model2_prc_avg=train(doc_list,"abalone","MCD")
model3_roc_avg,model3_prc_avg=train(doc_list,"abalone","LOF")
model4_roc_avg,model4_prc_avg=train(doc_list,"abalone","KNN")
model5_roc_avg,model5_prc_avg=train(doc_list,"abalone","LODA")

dataset2_result_roc=[model1_roc_avg,model2_roc_avg,model3_roc_avg,model4_roc_avg,model5_roc_avg]
dataset2_result_prc=[model1_prc_avg,model2_prc_avg,model3_prc_avg,model4_prc_avg,model5_prc_avg]

#统计每种方法的结果
result_roc=pd.DataFrame([["pageb",12,dataset1_result_roc[0],dataset1_result_roc[1],dataset1_result_roc[2],dataset1_result_roc[3],dataset1_result_roc[4]],["abalone",9,dataset2_result_roc[0],dataset2_result_roc[1],dataset2_result_roc[2],dataset2_result_roc[3],dataset2_result_roc[4]]],columns=["Dataset","Dimensions","PCA","MCD","LOF","KNN","LODA"])
result_prc=pd.DataFrame([["pageb",12,dataset1_result_prc[0],dataset1_result_prc[1],dataset1_result_prc[2],dataset1_result_prc[3],dataset1_result_prc[4]],["abalone",9,dataset2_result_prc[0],dataset2_result_prc[1],dataset2_result_prc[2],dataset2_result_prc[3],dataset2_result_prc[4]]],columns=["Dataset","Dimensions","PCA","MCD","LOF","KNN","LODA"])
result_roc
result_prc

#对全集csv文件进行训练并可视化结果
clf = PCA()
clf_name="PCA"
read=r"D:\研一下学期\数据挖掘\作业4\pageb\meta_data\pageb.preproc.csv"
data = pd.read_csv(read, header=0, index_col=0)
train_x = data.drop(drop+ground_truth+["original.label"], axis=1).values
train_y=np.array([transfor[x] for x in list(_flatten(data[ground_truth].values.tolist()))])
clf.fit(train_x)
label=clf.labels_
predict=clf.decision_scores_
evaluate_print(clf_name, train_y, predict)
pca = decomposition.PCA(n_components=2)
X = pca.fit_transform(train_x)
visualize(clf_name, X, train_y, X, train_y, label,
          train_y, show_figure=True, save_figure=True)

clf = MCD()
clf_name="PCA"
read=r"D:\研一下学期\数据挖掘\作业4\abalone\meta_data\abalone.preproc.csv"
data = pd.read_csv(read, header=0, index_col=0)
train_x = data.drop(drop+ground_truth+["original.label"], axis=1).values
train_y=np.array([transfor[x] for x in list(_flatten(data[ground_truth].values.tolist()))])
clf.fit(train_x)
label=clf.labels_
predict=clf.decision_scores_
evaluate_print(clf_name, train_y, predict)
pca = decomposition.PCA(n_components=2)
X = pca.fit_transform(train_x)
visualize(clf_name, X, train_y, X, train_y, label,
          train_y, show_figure=True, save_figure=True)

