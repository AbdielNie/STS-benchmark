import csv
result_filenames = ['./stsb-mpnet-base-out',
                './stsb-roberta-base-out',
                './stsb-roberta-large-out',
                './bert-base-uncased-stsb-out',]
weights = [0.05,0.25,0.65,0.05]

results = []
for filename in result_filenames:
    temp_result = []
    with open(filename+'/output_test.csv','r') as f:
        reader = csv.reader(f)
        for row in reader:
            temp_result.append(float(row[0]))
    results.append(temp_result)

test_results = []
for i in range(len(results[0])):
    final_value = 0.0
    for j in range(len(weights)):
        final_value += weights[j] * results[j][i]
    test_results.append(final_value)

with open(f'weighted_average_results.csv','w',newline="") as f:
    writer = csv.writer(f)
    for i in range(len(test_results)):
        writer.writerow([str(test_results[i])])