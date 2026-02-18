import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import warnings

from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.svm import SVR
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score

# Uyarıları ve görsel ayarlarını yapılandırma
warnings.filterwarnings('ignore')
sns.set_style("whitegrid")
pd.options.display.max_columns = None

def load_and_clean_data(filepath):
    """
    Veriyi yükler, gereksiz indeks sütununu atar ve
    fiziksel olarak imkansız olan (0 boyutlu) verileri temizler.
    """
    # Unnamed sütunu genellikle gereksiz indekstir, okurken index_col=0 diyerek kurtulabiliriz
    df = pd.read_csv(filepath, index_col=0)
    
    # x, y, z boyutlarının 0 olması fiziksel olarak imkansızdır, bunları temizliyoruz
    # (Bu işlem yaklaşık 20-30 satırı siler)
    df = df[(df[['x', 'y', 'z']] != 0).all(axis=1)]
    
    return df

def remove_outliers(df):
    """
    Veri setindeki uç değerleri (outliers) belirli eşik değerlerine göre filtreler.
    Bu eşikler EDA (Keşifçi Veri Analizi) aşamasında belirlenmiştir.
    """
    # Depth ve Table için filtreleme
    df = df[(df["depth"] < 75) & (df["depth"] > 45)]
    df = df[(df["table"] < 75) & (df["table"] > 40)]
    
    # Boyutlar (x, y, z) için filtreleme
    df = df[(df["z"] < 30) & (df["z"] > 2)]
    df = df[(df["y"] < 75)]
    
    return df

def preprocess_features(df):
    """
    Kategorik verileri (Cut, Color, Clarity) sayısal hale getirir
    ve veriyi ölçeklendirir.
    """
    X = df.drop("price", axis=1)
    y = df["price"]
    
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.25, random_state=15)
    
    # Kategorik Dönüşüm (Label Encoding)
    categorical_cols = ["cut", "color", "clarity"]
    encoders = {}
    for col in categorical_cols:
        encoders[col] = LabelEncoder()
        X_train[col] = encoders[col] .fit_transform(X_train[col])
        X_test[col] = encoders[col] .transform(X_test[col])
        
    # Standardizasyon (Scaling)
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)
    
    return X_train_scaled, X_test_scaled, y_train, y_test,encoders,scaler

def evaluate_model(X_train,X_test,y_train,y_test):
    svr = SVR(C=1000,gamma =0.1,kernel="rbf")
    svr.fit(X_train,y_train)
    y_pred=svr.predict(X_test)
    score = r2_score(y_test,y_pred)
    print("R2 Score: ",score)
    return svr
# --- ANA UYGULAMA AKIŞI ---

if __name__ == "__main__":
    # 1. Veri Yükleme ve Temizlik
    print("Loading data...")
    df = load_and_clean_data("10-diamonds.csv")
    
    # 2. Aykırı Değerlerin Temizlenmesi
    df = remove_outliers(df)
    
    # 3. Ön İşleme (Encoding & Scaling)
    X_train, X_test, y_train, y_test ,encoders,scaler= preprocess_features(df)
    svr=evaluate_model(X_train,X_test,y_train,y_test)
    
    import pickle
    with open("diamond_model_complete.pkl","wb") as f:
        pickle.dump(
            {'model': svr,
            'encoders':encoders,
            'scaler':scaler
            },f
            )
    pd.DataFrame(X_test).to_csv("testdata.csv",index=False)