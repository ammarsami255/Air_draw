import pickle
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score

# تحميل البيانات
with open('./left.pickle', 'rb') as f:
    data_dict = pickle.load(f)

# تأكد من أن كل العناصر NumPy arrays
data_list = [np.array(item) for item in data_dict['data']]
max_length = max(len(item) for item in data_list)

# توحيد الأطوال بـ padding
data_padded = np.array([np.pad(item, (0, max_length - len(item)), mode='constant') for item in data_list])
labels = np.asarray(data_dict['labels'])

# تقسيم البيانات
x_train, x_test, y_train, y_test = train_test_split(data_padded, labels, test_size=0.2, shuffle=True, stratify=labels)

# تدريب النموذج
model = RandomForestClassifier()
model.fit(x_train, y_train)

# التقييم
y_predict = model.predict(x_test)
score = accuracy_score(y_predict, y_test)
print(f'{score * 100:.2f}% of samples were classified correctly!')

# حفظ النموذج
with open('left.p', 'wb') as f:
    pickle.dump({'model': model}, f)
