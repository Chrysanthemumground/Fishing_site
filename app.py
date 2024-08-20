from flask import Flask, render_template, request
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error
app = Flask(__name__) 

data = {
    "時間": ["06:00〜13:15", "07:00〜14:00", "08:00〜15:00", "06:00〜13:15", "07:00〜14:00"],
    "釣果": [1, 2, 3, 2, 1],
    "気温": [25.0, 26.0, 27.0, 24.5, 25.5],
    "風向": ["北", "東", "西", "北", "東"],
    "風速": [1.4, 1.6, 1.2, 1.3, 1.5],
    "気圧": [1010, 1012, 1013, 1009, 1011],
    "潮": ["小潮", "中潮", "大潮", "小潮", "中潮"],
    "潮の高さ": [22.6, 24.0, 26.0, 22.0, 23.5]
}

# データフレームを作成
df = pd.DataFrame(data)

# 特徴量とターゲットに分ける
X = df.drop(columns=["釣果"])
y = df["釣果"]

# カテゴリカルデータをダミー変数に変換
X = pd.get_dummies(X, drop_first=True)

# 訓練データとテストデータに分割
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# モデルの構築と訓練
model = RandomForestRegressor(n_estimators=100, random_state=42)
model.fit(X_train, y_train)

@app.route('/')
def hello_world():
    return render_template('index.html')


   

    

@app.route('/calc', methods=['POST'])
def calc():
    kiatu = request.form['kiatu']
    time = request.form['time']
    wing = request.form['wing']
    huko = request.form['huko']
    shio = request.form['shio']
    temperture = request.form['temperture']
    shioshio = request.form['shioshio']


    new_data = {
    "時間": [time],
    "気温": [temperture],
    "風向": [huko],
    "風速": [wing],
    "気圧": [kiatu],
    "潮": [shioshio],
    "潮の高さ": [shio]
    } 

    new_df = pd.DataFrame(new_data)

    # トレーニング時と同じダミー変数を適用
    new_df = pd.get_dummies(new_df, drop_first=True)

    # トレーニング時の全ての特徴量を揃える
    new_df = new_df.reindex(columns=X_train.columns, fill_value=0)



    result = model.predict(new_df)
    print(new_df)
    print(result)
    return render_template('index.html',result = result )

if __name__ == "__main__":
    app.run(debug=True)
