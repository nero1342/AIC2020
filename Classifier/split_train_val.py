
from sklearn.model_selection import train_test_split
from sklearn.metrics import f1_score
import tensorflow as tf

random_split = False
if random_split:
  train_csv_path = 'train_vehicle_type.csv'
  df = pd.read_csv(train_csv_path, sep = ',', dtype = {'img_id': str})
  X = df
  y = np.array(df['vehicle_type'])
  X_train_df, X_val_df, y_train, y_val = train_test_split(X, y, test_size=0.3, stratify = y)
  os.makedirs('list')
  X_train_df.to_csv('list/train.csv', index = False)
  X_val_df.to_csv('list/val.csv', index = False)