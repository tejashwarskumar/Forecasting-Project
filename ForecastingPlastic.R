library(readr)
plastic<-read.csv(file.choose())
View(plastic)

library(tseries)
salesamts<-ts(plastic$Sales,frequency = 12,start=c(1949))
plot(salesamts)

train<-salesamts[1:48]
test<-salesamts[49:60]
train<-ts(train,frequency = 12)
test<-ts(test,frequency = 12)

library(forecast)
library(fpp)
library(smooth)

hw_abg<-HoltWinters(train,alpha = 0.2,beta = 0.15,gamma = 0.05) #HoltWinters using specified alpha, beta and gamma values
hw_abg
hwabg_pred<-data.frame(predict(hw_abg,n.ahead = 12))
plot(forecast(hw_abg,h=12))
hwabg_mape<-MAPE(hwabg_pred$fit,test)*100

hw_a<-HoltWinters(train,alpha=0.2) #HoltWinters using specified alpha and optimal beta and gamma values
hw_a
hwa_pred<-data.frame(predict(hw_a,n.ahead = 12))
plot(forecast(hw_a,h=12))
hwa_mape<-MAPE(hwa_pred$fit,test)*100

hw_nabg<-HoltWinters(train) #HoltWinters using optimal alpha, beta and gamma values
hw_nabg
hwnabg_pred<-data.frame(predict(hw_nabg,n.ahead = 12))
plot(forecast(hw_nabg,h=12))
hwnabg_mape<-MAPE(hwnabg_pred$fit,test)*100

df_mape<-data.frame(c("hwabg_mape","hwa_mape","hwnabg_mape"),c(hwabg_mape,hwa_mape,hwnabg_mape))
colnames(df_mape)<-c("MAPE","VALUES")
View(df_mape)

new_model <- HoltWinters(salesamts,alpha = 0.2)
forecast_new <- data.frame(predict(new_model,n.ahead=12))
plot(forecast(new_model,n.ahead=12))

########## Model Based Forecasting ###########
X<- data.frame(outer(rep(month.abb,length = 60), month.abb,"==") + 0 )# Creating dummies for 12 months
colnames(X)<-month.abb # Assigning month names 
View(X)
plastic1<-cbind(plastic,X)

plastic1["t"]<- 1:60
plastic1["t_square"]<-plastic1["t"]*plastic1["t"]
attach(plastic1)

train<-plastic1[1:48,-c(1)]
test<-plastic1[49:60,-c(1)]

linear_model<-lm(Sales~t,data=train) #linear model
summary(linear_model)
linear_pred<-data.frame(predict(linear_model,interval='predict',newdata =test))
rmse_linear<-sqrt(mean((test$Sales-linear_pred$fit)^2,na.rm = T))
rmse_linear

Quad_model<-lm(Sales~t+t_square,data=train) #quadratic model
summary(Quad_model)
Quad_pred<-data.frame(predict(Quad_model,interval='predict',newdata=test))
rmse_Quad<-sqrt(mean((test$Sales-Quad_pred$fit)^2,na.rm=T))
rmse_Quad

Add_sea_Linear_model<-lm(Sales~t+Jan+Feb+Mar+Apr+May+Jun+Jul+Aug+Sep+Oct+Nov,data=train) # linear model with additive seasonality
summary(Add_sea_Linear_model)
Add_sea_Linear_pred<-data.frame(predict(Add_sea_Linear_model,interval='predict',newdata=test))
rmse_Add_sea_Linear<-sqrt(mean((test$Sales-Add_sea_Linear_pred$fit)^2,na.rm=T))
rmse_Add_sea_Linear

Add_sea_Quad_model<-lm(Sales~t+t_square+Jan+Feb+Mar+Apr+May+Jun+Jul+Aug+Sep+Oct+Nov,data=train) # quadratic model with additive seasonality
summary(Add_sea_Quad_model)
Add_sea_Quad_pred<-data.frame(predict(Add_sea_Quad_model,interval='predict',newdata=test))
rmse_Add_sea_Quad<-sqrt(mean((test$Sales-Add_sea_Quad_pred$fit)^2,na.rm=T))
rmse_Add_sea_Quad

table_rmse<-data.frame(c("rmse_linear","rmse_Quad","rmse_Add_sea_Linear","rmse_Add_sea_Quad"),c(rmse_linear,rmse_Quad,rmse_Add_sea_Linear,rmse_Add_sea_Quad))
colnames(table_rmse)<-c("model","RMSE")
View(table_rmse)

new_model <- lm(Sales~t+Jan+Feb+Mar+Apr+May+Jun+Jul+Aug+Sep+Oct+Nov,data=plastic1)
pred_new<-data.frame(predict(new_model,newdata=plastic1,interval = 'predict'))
View(pred_new)
plot(pred_new$fit,type = "o")
