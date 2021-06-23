import pickle
import string
from nltk.corpus import stopwords
from flask import Flask, render_template, request, url_for

app = Flask(__name__)

# stopwords = pickle.load(open('stp.pkl','rb'))
lem = pickle.load(open('lem.pkl','rb'))
ps = pickle.load(open('pm.pkl','rb'))
vect = pickle.load(open('vect.pkl','rb'))


def func3(text):
    no_punc = [char for char in text if char not in string.punctuation]
    no_punc = ''.join(no_punc)
    stop_words = [word for word in no_punc.split() if word not in stopwords.words('english')]
    stem_words = [word for word in stop_words if ps.stem(word)]
    lem_words = [word for word in stem_words if lem.lemmatize(word)]
    str_words = [word for word in lem_words]
    str_words = ' '.join(str_words)
    return str_words

model = pickle.load(open('model.pkl','rb'))
# # f1 = pickle.load(open('f1.pkl','rb'))
# f2 = pickle.load(open('f2.pkl','rb'))
# f3 = pickle.load(open('pm.pkl','rb'))
# f4 = pickle.load(open('lem.pkl','rb'))
# f5 = pickle.load(open('stp.pkl','rb'))



@app.route('/')
def home():
    return render_template('index.html')

@app.route('/predict', methods = ['POST'])
def subj():
    sub = request.form['sub']
    bod = request.form['body']
    mail = sub+' '+bod
    mail = mail.lower()
    mail_new = func3(mail)
    x = vect.transform([mail_new])
    y = model.predict(x)

    if (y[0] == 0):
        return(render_template('index.html', ans = 'NOT SPAM'))
    else:
        return(render_template('index.html', ans = 'SPAM'))



    # return render_template('index.html', ans = 1)

if __name__ == '__main__':
    app.run(debug = True)