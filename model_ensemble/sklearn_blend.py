from datasets import load_dataset, load_metric
import csv
from sklearn.linear_model import LinearRegression
from sklearn.svm import SVR
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import make_scorer
from sklearn import preprocessing

def spearman_score(pred,label):
    metric = load_metric("glue", "stsb")
    val_spearmanr = metric.compute(predictions=pred, references=label)['spearmanr']
    print('  spearman: ',val_spearmanr)
    return val_spearmanr

def blending(model,save_path):

    file_names = [
                './bert-base-uncased-model-out',
                './robert-base-out',
                './deberta-base-out',
                './stsb-mpnet-base-out',
                './stsb-roberta-base-out',
                './stsb-roberta-large-out',
                './bert-base-uncased-stsb-out',
                './stsb-distilroberta-base-out',
                ]

    val_results = []
    for filename in file_names:
        temp_result = []
        with open(filename+'/output_validation.csv','r') as f:
            reader = csv.reader(f)
            for row in reader:
                temp_result.append(float(row[0]))
        val_results.append(temp_result)

    val_datas = []
    for i in range(len(val_results[0])):
        val_datas.append([val_results[j][i] for j in range(len(val_results))])
    print(len(val_datas))

    test_results = []
    for filename in file_names:
        temp_result = []
        with open(filename+'/output_test.csv','r') as f:
            reader = csv.reader(f)
            for row in reader:
                temp_result.append(float(row[0]))
        test_results.append(temp_result)

    test_datas = []
    for i in range(len(test_results[0])):
        test_datas.append([test_results[j][i] for j in range(len(test_results))])
    print(len(test_datas))

    sts_data = load_dataset("glue", "stsb")
    validation = list(sts_data['validation'])

    val_labels = [data["label"] for data in validation]

    scaler = preprocessing.StandardScaler().fit(val_datas)

    val_datas = scaler.transform(val_datas)
    test_datas = scaler.transform(test_datas)
    
    model.fit(val_datas,val_labels)

    val_pred = model.predict(val_datas)

    test_pred = model.predict(test_datas)

    print('finally, spearmanr: ',spearman_score(val_pred,val_labels))

    with open(save_path,'w',newline="") as f:
        writer = csv.writer(f)
        for i in range(len(test_pred)):
            writer.writerow([str(test_pred[i])])

if __name__ == "__main__":
    # lin_reg = LinearRegression()
    # blending(lin_reg,'lin_results.csv')


    svr_reg = SVR()
    parameters = {
        'kernel':('linear','rbf'), 
        'tol':[1e-5,1e-4,1e-3],
        'C':[1.0,2.0,3.0],
        # 'max_iter':[10,50,100,500,-1]
        }
    # parameters = {
    #     'kernel':('linear','rbf'), 
    #     'tol':[1e-5,1e-4,1e-3]
    #     }
    spearman_scorer = make_scorer(spearman_score)
    reg = GridSearchCV(svr_reg, parameters, scoring=spearman_scorer, cv=3)

    blending(reg,'svr_grid_search_results.csv')

    print(reg.best_params_)
