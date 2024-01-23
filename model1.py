import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
import math

def count_probability(row, angle):
    data_dict = X_train.to_dict(orient='records')

    # remove decision attribute

    data_dict[0].popitem()

    values = []

    for attribute_id, attribute in enumerate(data_dict[0].keys()):
        nominator = count_occurence(attribute, row.get(attribute), angle)
        denominator = count_target_occurence("num", angle)

        # Laplace

        if nominator == 0:
            nominator += 1
            denominator += count_target_occurence(attribute, row.get(attribute))

        values.append(nominator / denominator)

    return np.prod(values)


def count_occurence(attribute, value, angle):
    data_dict = X_train.to_dict(orient='records')

    count = 0
    for row in data_dict:
        if row.get(attribute) == value and row.get("num") == angle:
            count += 1

    return count


def count_target_occurence(attribute, value):
    data_dict = X_train.to_dict(orient='records')

    count = 0
    for row in data_dict:
        if row.get(attribute) == value:
            count += 1

    return count


def get_accuracy(classified):
    train_classified = list(X_test["num"])
    test_classified = classified

    length = len(test_classified)

    count = 0

    for i, angle in enumerate(test_classified):
        if train_classified[i] == test_classified[i]:
            count += 1

    accuracy = (count / length) * 100

    return accuracy


def get_confussion_matrix(classified):
    train_classified = list(X_test["num"])
    test_classified = classified

    data = {'y_actual': train_classified, 'y_predicted': test_classified}

    matrix_data = pd.DataFrame(data)

    confusion_matrix = pd.crosstab(matrix_data['y_actual'], matrix_data['y_predicted'], rownames=['Actual'], colnames=['Predicted'])

    return confusion_matrix


def get_avg_acc(acc):
    sum = 0
    for val in acc:
        sum += val

    avg = sum / len(acc)
    return int(avg)


def get_deviation(acc):
    avg = get_avg_acc(acc)
    sum = 0

    for val in acc:
        sum += pow((int(val) - avg), 2)

    deviation = math.sqrt(sum / len(acc))

    return deviation


df = pd.read_csv("./files/processedcleveland.csv")

df.replace('?', pd.NA, inplace=True)
df.dropna(inplace=True)

# dichotomization to 0 - healthy, 1,2,3,4 - sick

replacements = {2: 1, 3: 1, 4: 1}
df['num'] = df['num'].map(replacements).fillna(df['num'])

# binning

df["age"] = pd.cut(x=df["age"], bins=[25, 40, 45, 50, 55, 60, 65, 70, 75, 80], labels=[1, 2, 3, 4, 5, 6, 7, 8, 9])
df["thalach"] = pd.cut(x=df["thalach"], bins=[70, 90, 110, 130, 150, 170, 190, 210], labels=[1, 2, 3, 4, 5, 6, 7])
df["chol"] = pd.cut(x=df["chol"], bins=[120, 170, 220, 270, 320, 370, 420, 470, 520, 570],
                    labels=[1, 2, 3, 4, 5, 6, 7, 8, 9])
df["trestbps"] = pd.cut(x=df["trestbps"], bins=[90, 105, 120, 135, 150, 165, 180, 195, 210],
                        labels=[1, 2, 3, 4, 5, 6, 7, 8])
df["oldpeak"] = pd.cut(x=df["oldpeak"], bins=[-1, 0, 1, 2, 3, 4, 5, 6, 7], labels=[1, 2, 3, 4, 5, 6, 7, 8])

# remove higher correlation

corr_matrix = df.corr().abs()
upper = corr_matrix.where(np.triu(np.ones(corr_matrix.shape), k=1).astype(bool))
to_drop = [column for column in upper.columns if any(upper[column] > 0.8) and column != "num"]
df.drop(to_drop, axis=1, inplace=True)

X = df
y = df['num']

avg_acc = []

for cycle in range(5):
    print("Processing...")
    print()

    # split to train and test

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.25)

    X_test_no_num = X_test.iloc[:, :-1]

    test_data = X_test_no_num.to_dict(orient='records')

    classified = []

    for row in test_data:
        prob_list = {}

        # healthy angle

        negative_prob = count_probability(row, 0)
        prob_list.update({0: negative_prob})

        # sick angle

        low_prob = count_probability(row, 1)
        prob_list.update({1: low_prob})

        max_key = max(prob_list, key=prob_list.get)

        classified.append(max_key)

    avg_acc.append(get_accuracy(classified))
    print("-----------")
    print("Cycle", cycle, '\n')
    print("Accuracy:" + str(int(get_accuracy(classified))) + "%")
    print()
    print("[Confussion matrix]")
    print(get_confussion_matrix(classified), '\n')

print()
print("[Summary]:")
print("Average accuracy: " + str(get_avg_acc(avg_acc)) + "%")
print("Standard deviation: " + str(get_deviation(avg_acc)))
