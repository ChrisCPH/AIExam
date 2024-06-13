from flask import Flask, request, render_template
import pickle
import pandas as pd

app = Flask(__name__)

RandomForestModel = pickle.load(open('models/RandomForestModel.pkl','rb'))

# Getting kommune feature
df = pd.read_csv('../Data/Data_processed/X_train_data.csv', sep=';', encoding='latin1')
valid_kommunes = {col for col in df.columns if col.startswith('Kommune_')}

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/RF',methods=['POST'])
def RF():

    boligtype = request.form.get("Boligtype")
    parcel = 1 if boligtype == "Parcel" else 0
    fritidshus = 1 if boligtype == "Fritidshus" else 0
    ejerlejlighed = 1 if boligtype == "Ejerlejlighed" else 0

    kommune_input = request.form.get("Kommune")
    kommune_feature = f"Kommune_{kommune_input}"
    kommune_features = {f: 0 for f in valid_kommunes}
    if kommune_feature in valid_kommunes:
        kommune_features[kommune_feature] = 1
    else:
        return render_template('index.html', RF_result="Invalid Kommune")
    
    år = request.form.get("År")

    måned = request.form.get("Måned")

    # Sorting it so it should be alphebetical
    sorted_kommune_features = [kommune_features[key] for key in sorted(kommune_features)]

    features = [ejerlejlighed, fritidshus, parcel] + sorted_kommune_features + [år, måned]
    
    prediction = RandomForestModel.predict([features])
    
    result = prediction[0]

    return render_template('index.html', RF_result=result)

if __name__ == "__main__":
    app.run(debug=True)