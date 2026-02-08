# apps\service_provider\views.py

from django.db.models import Count, Avg, Q
from django.shortcuts import render, redirect
from django.http import HttpResponse
import xlwt
import pandas as pd
import numpy as np

from sklearn.feature_extraction.text import CountVectorizer
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix, accuracy_score, classification_report
from sklearn.ensemble import VotingClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.linear_model import LogisticRegression, SGDClassifier
from sklearn.naive_bayes import MultinomialNB
from sklearn import svm

from apps.service_provider.models import (
    ClientRegister_Model,
    Spam_Prediction,
    detection_ratio,
    detection_accuracy
)

def serviceproviderlogin(request):
    if request.method  == "POST":
        admin = request.POST.get('username')
        password = request.POST.get('password')
        if admin == "Admin" and password =="Admin":
            detection_accuracy.objects.all().delete()
            return redirect('View_remote_users')

    return render(request,'SProvider/serviceproviderlogin.html')

def View_IOTMessage_Type_Ratio(request):
    detection_ratio.objects.all().delete()

    total_count = Spam_Prediction.objects.count()

    if total_count == 0:
        return render(request, 'SProvider/View_IOTMessage_Type_Ratio.html', {'objs': []})

    spam_count = Spam_Prediction.objects.filter(Prediction='Spam').count()
    normal_count = Spam_Prediction.objects.filter(Prediction='Normal').count()

    spam_ratio = (spam_count / total_count) * 100
    normal_ratio = (normal_count / total_count) * 100

    detection_ratio.objects.create(names='Spam', ratio=spam_ratio)
    detection_ratio.objects.create(names='Normal', ratio=normal_ratio)

    obj = detection_ratio.objects.all()
    return render(request, 'SProvider/View_IOTMessage_Type_Ratio.html', {'objs': obj})

def View_remote_users(request):
    obj = ClientRegister_Model.objects.all()
    return render(request, 'SProvider/View_Remote_Users.html'
, {'objects': obj})

def ViewTrendings(request):
    topic = Spam_Prediction.objects.values('Prediction').annotate(
        dcount=Count('Prediction')
    ).order_by('-dcount')

    return render(request, 'SProvider/ViewTrendings.html', {'objects': topic})

def charts(request,chart_type):
    chart1 = detection_ratio.objects.values('names').annotate(dcount=Avg('ratio'))
    return render(request,"SProvider/charts.html", {'form':chart1, 'chart_type':chart_type})

def charts1(request,chart_type):
    chart1 = detection_accuracy.objects.values('names').annotate(dcount=Avg('ratio'))
    return render(request,"SProvider/charts1.html", {'form':chart1, 'chart_type':chart_type})

def View_Prediction_Of_IOTMessage_Type(request):
    obj =Spam_Prediction.objects.all()
    return render(request, 'SProvider/View_Prediction_Of_IOTMessage_Type.html', {'list_objects': obj})

def likeschart(request,like_chart):
    charts =detection_accuracy.objects.values('names').annotate(dcount=Avg('ratio'))
    return render(request,"SProvider/likeschart.html", {'form':charts, 'like_chart':like_chart})


def Download_Trained_DataSets(request):

    response = HttpResponse(content_type='application/ms-excel')
    # decide file name
    response['Content-Disposition'] = 'attachment; filename="Predicted_Data.xls"'
    # creating workbook
    wb = xlwt.Workbook(encoding='utf-8')
    # adding sheet
    ws = wb.add_sheet("sheet1")
    # Sheet header, first row
    row_num = 0
    font_style = xlwt.XFStyle()
    # headers are bold
    font_style.font.bold = True
    # writer = csv.writer(response)
    obj = Spam_Prediction.objects.all()
    data = obj  # dummy method to fetch data.
    for my_row in data:
        row_num = row_num + 1
        ws.write(row_num, 0, my_row.Message_Id, font_style)
        ws.write(row_num, 1, my_row.Message_Date, font_style)
        ws.write(row_num, 2, my_row.IOT_Message, font_style)
        ws.write(row_num, 3, my_row.Prediction, font_style)

    wb.save(response)
    return response

def train_model(request):
    detection_accuracy.objects.all().delete()
    data = pd.read_csv("IOT_Datasets.csv")
    # data.replace([np.inf, -np.inf], np.nan, inplace=True)

    mapping = {'ham': 0,'spam': 1}
    data['Results'] = data['Label'].map(mapping)

    x = data['Message']
    y = data['Results']

    # data.drop(['Type_of_Breach'],axis = 1, inplace = True)
    cv = CountVectorizer()

    print(x)
    print(y)

    x = cv.fit_transform(data['Message'].astype(str))

    models = []
    from sklearn.model_selection import train_test_split
    X_train, X_test, y_train, y_test = train_test_split(x, y, test_size=0.20)
    X_train.shape, X_test.shape, y_train.shape

    print("Naive Bayes")

    from sklearn.naive_bayes import MultinomialNB

    NB = MultinomialNB()
    NB.fit(X_train, y_train)
    predict_nb = NB.predict(X_test)
    naivebayes = accuracy_score(y_test, predict_nb) * 100
    print("ACCURACY")
    print(naivebayes)
    print("CLASSIFICATION REPORT")
    print(classification_report(y_test, predict_nb))
    print("CONFUSION MATRIX")
    print(confusion_matrix(y_test, predict_nb))
    detection_accuracy.objects.create(names="Naive Bayes", ratio=naivebayes)


    # SVM Model
    print("SVM")
    from sklearn import svm

    lin_clf = svm.LinearSVC()
    lin_clf.fit(X_train, y_train)
    predict_svm = lin_clf.predict(X_test)
    svm_acc = accuracy_score(y_test, predict_svm) * 100
    print("ACCURACY")
    print(svm_acc)
    print("CLASSIFICATION REPORT")
    print(classification_report(y_test, predict_svm))
    print("CONFUSION MATRIX")
    print(confusion_matrix(y_test, predict_svm))
    detection_accuracy.objects.create(names="SVM", ratio=svm_acc)

    print("Logistic Regression")

    from sklearn.linear_model import LogisticRegression

    reg = LogisticRegression(random_state=0, solver='lbfgs').fit(X_train, y_train)
    y_pred = reg.predict(X_test)
    print("ACCURACY")
    print(accuracy_score(y_test, y_pred) * 100)
    print("CLASSIFICATION REPORT")
    print(classification_report(y_test, y_pred))
    print("CONFUSION MATRIX")
    print(confusion_matrix(y_test, y_pred))
    detection_accuracy.objects.create(names="Logistic Regression", ratio=accuracy_score(y_test, y_pred) * 100)

    print("Decision Tree Classifier")
    dtc = DecisionTreeClassifier()
    dtc.fit(X_train, y_train)
    dtcpredict = dtc.predict(X_test)
    print("ACCURACY")
    print(accuracy_score(y_test, dtcpredict) * 100)
    print("CLASSIFICATION REPORT")
    print(classification_report(y_test, dtcpredict))
    print("CONFUSION MATRIX")
    print(confusion_matrix(y_test, dtcpredict))
    detection_accuracy.objects.create(names="Decision Tree Classifier", ratio=accuracy_score(y_test, dtcpredict) * 100)

    print("SGD Classifier")
    from sklearn.linear_model import SGDClassifier
    sgd_clf = SGDClassifier(loss='hinge', penalty='l2', random_state=0)
    sgd_clf.fit(X_train, y_train)
    sgdpredict = sgd_clf.predict(X_test)
    print("ACCURACY")
    print(accuracy_score(y_test, sgdpredict) * 100)
    print("CLASSIFICATION REPORT")
    print(classification_report(y_test, sgdpredict))
    print("CONFUSION MATRIX")
    print(confusion_matrix(y_test, sgdpredict))
    detection_accuracy.objects.create(names="SGD Classifier", ratio=accuracy_score(y_test, sgdpredict) * 100)


    labeled = 'Processed_data.csv'
    data.to_csv(labeled, index=False)

    obj = detection_accuracy.objects.all()
    return render(request,'SProvider/train_model.html', {'objs': obj})