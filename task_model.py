import pandas as pd
from sklearn.preprocessing import LabelEncoder
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans


def one_hot_encode_columns(df, columns_to_encode):
    df_encoded = pd.get_dummies(df, columns=columns_to_encode, prefix=columns_to_encode)
    df_encoded = df_encoded.astype(int)
    return df_encoded


dataframe = pd.read_csv('./datasets/customers_mall.csv')
print(dataframe.head())


dataframe_2 = one_hot_encode_columns(dataframe, columns_to_encode=['Gender'])
df_train=dataframe_2.drop('CustomerID',axis=1)

dataframe_3 = dataframe.drop(columns=['CustomerID'],axis=1)

dataframe_kmeans=dataframe_3.copy(deep=True)

# Label Encoding
le = LabelEncoder()
# Get a list of categorical columns
categorical_cols = dataframe_kmeans.select_dtypes(include='object').columns
# Apply the label encoder to each categorical column
for col in categorical_cols:
    dataframe_kmeans[col] = le.fit_transform(dataframe_kmeans[col])

# select the features
X = dataframe_kmeans
#Scaling Data
scaler = StandardScaler()
X = scaler.fit_transform(X)

# create a k-means object with the optimal number of clusters
optimal_k = 4 # number of clusters where the elbow is
kmeans = KMeans(n_clusters=optimal_k, init='k-means++', max_iter=300, n_init=10, random_state=0)
kmeans.fit(X)
y_kmeans = kmeans.predict(X)
dataframe_kmeans['cluster'] = y_kmeans

dataframe_4 = dataframe.drop(columns=['CustomerID'],axis=1)
df_hr = dataframe_4.copy(deep=True)
categorical_cols = df_hr.select_dtypes(include='object').columns
for col in categorical_cols:
    df_hr[col] = le.fit_transform(df_hr[col])
