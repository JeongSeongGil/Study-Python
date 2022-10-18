import cv2
import pickle, gzip


with gzip.open("data/mnist.pkl.gz", "rb") as f:
    train_set, valid_set, test_set = pickle.load(f, encoding="latin1")

train_data, train_label = train_set
test_data, test_label = test_set

print("train_set=", train_set[0].shape)
print("valid_set=", valid_set[0].shape)
print("test_set=", test_set[0].shape)

print("training...")

knn = cv2.ml.KNearest_create()
train_idx = 1000

for_cnt = int(len(train_data) / train_idx) + 1

for i in range(1, for_cnt):
    train_cnt = i * train_idx
    knn.train(train_data[:train_cnt], cv2.ml.ROW_SAMPLE, train_label[:train_cnt])

    sample_cnt = 1000
    ret, results, neighbours, dist = knn.findNearest(test_data[:sample_cnt], k=5)
    accur = sum(test_label[:sample_cnt] == results.flatten()) / len(results)

    print(train_cnt, "건 학습 후, 정확도 : ", accur * 100, '%')



