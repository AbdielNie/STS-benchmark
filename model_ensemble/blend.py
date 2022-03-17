import torch
from torch import nn
import torch.utils.data as Data 
import csv

from datasets import load_dataset, load_metric



class BlendingNeuralNetwork(nn.Module):
    def __init__(self,input_size:int):
        super(BlendingNeuralNetwork, self).__init__()
        self.linear_stack = nn.Sequential(
            nn.Linear(input_size, 512),
            #nn.ReLU(),
            nn.Linear(512, 1)
        )

    def forward(self, x):
        logits = self.linear_stack(x)
        return logits

def train(loader,device,optimizer,criterion):
    total_loss = 0
    metric = load_metric("glue", "stsb")

    for i,(data,target) in enumerate(loader):
        data,target = data.to(device),target.to(device)

        pred = model(data)
        loss = criterion(pred,target)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        total_loss += loss.item()

        metric.add_batch(predictions=pred, references=target)

    spearmanr = metric.compute()['spearmanr']
    print('spearmanr:',spearmanr)

    print('total_loss:',total_loss)
    

if __name__ == "__main__":
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

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Using {device} device")
    model = BlendingNeuralNetwork(8).to(device)

    results = []
    for filename in file_names:
        temp_result = []
        with open(filename+'/output_validation.csv','r') as f:
            reader = csv.reader(f)
            for row in reader:
                temp_result.append(float(row[0]))
        results.append(temp_result)

    datas = []
    for i in range(len(results[0])):
        datas.append([results[j][i] for j in range(len(results))])
    datas=torch.tensor(datas)
    print(datas.shape)

    sts_data = load_dataset("glue", "stsb")
    validation = list(sts_data['validation'])
    test = list(sts_data['test'])

    labels = [data["label"] for data in validation]


    for i,model_name in enumerate(file_names):
        metric = load_metric("glue", "stsb")
        spearmanr = metric.compute(predictions=results[i], references=labels)['spearmanr']
        print(f'{model_name} spearmanr on valid:',spearmanr)

    

    # with open('validation_label.csv','w',newline="") as f:
    #     writer = csv.writer(f)
    #     for i in range(len(labels)):
    #         writer.writerow([str(labels[i])])

    labels = torch.tensor(labels)
    print(labels.shape)
    
    torch_dataset = Data.TensorDataset(datas,labels)
    loader = Data.DataLoader(
        dataset=torch_dataset,
        batch_size = 1,
        shuffle=True,
    )

    optimizer = torch.optim.SGD(model.parameters(),lr=5e-4)
    criterion = nn.MSELoss()

    for epoch in range(100):
        print('epoch:',epoch)
        train(loader,device,optimizer,criterion)
        
    
    results = []
    for filename in file_names:
        temp_result = []
        with open(filename+'/output_test.csv','r') as f:
            reader = csv.reader(f)
            for row in reader:
                temp_result.append(float(row[0]))
        results.append(temp_result)

    datas = []
    for i in range(len(results[0])):
        datas.append([results[j][i] for j in range(len(results))])
    datas=torch.tensor(datas)
    print(datas.shape)
    
    torch_dataset = Data.TensorDataset(datas)
    loader = Data.DataLoader(
        dataset=torch_dataset,
        batch_size = 1,
        shuffle=False,
    )
    model.eval()
    predictions = []

    

    for data in loader:
        data = data[0]
        data = data.to(device)
        predictions.append(model(data).item())

        

    print(len(predictions))

    with open('results.csv','w',newline="") as f:
        writer = csv.writer(f)
        for i in range(len(predictions)):
            writer.writerow([str(predictions[i])])

    # print([p for p in model.parameters()])
