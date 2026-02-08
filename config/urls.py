from django.contrib import admin
from django.urls import path
from django.conf import settings
from django.conf.urls.static import static

from apps.remote_user import views as remoteuser
from apps.service_provider import views as serviceprovider


urlpatterns = [
    path('admin/', admin.site.urls),

    path('', remoteuser.login, name="login"),
    path('Register1/', remoteuser.Register1, name="Register1"),
    path('Predict_IOT_Message_Type/', remoteuser.Predict_IOT_Message_Type, name="Predict_IOT_Message_Type"),
    path('ViewYourProfile/', remoteuser.ViewYourProfile, name="ViewYourProfile"),
    path('Add_DataSet_Details/', remoteuser.Add_DataSet_Details, name="Add_DataSet_Details"),

    path('serviceproviderlogin/', serviceprovider.serviceproviderlogin, name="serviceproviderlogin"),
    path('View_Remote_Users/', serviceprovider.View_remote_users, name="View_remote_users"),

    path('charts/<str:chart_type>/', serviceprovider.charts, name="charts"),
    path('charts1/<str:chart_type>/', serviceprovider.charts1, name="charts1"),
    path('likeschart/<str:like_chart>/', serviceprovider.likeschart, name="likeschart"),

    path('View_IOTMessage_Type_Ratio/', serviceprovider.View_IOTMessage_Type_Ratio, name="View_IOTMessage_Type_Ratio"),
    path('train_model/', serviceprovider.train_model, name="train_model"),
    path('View_Prediction_Of_IOTMessage_Type/', serviceprovider.View_Prediction_Of_IOTMessage_Type, name="View_Prediction_Of_IOTMessage_Type"),
    path('Download_Trained_DataSets/', serviceprovider.Download_Trained_DataSets, name="Download_Trained_DataSets"),
]

if settings.DEBUG:
    urlpatterns += static(settings.MEDIA_URL, document_root=settings.MEDIA_ROOT)
