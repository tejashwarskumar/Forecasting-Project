library(forecast) #for forecasting
library(fpp)     
library(smooth)   #for smoothing techniques
library(readxl)
airline<-read_xlsx(file.choose())
View(airline)

library(tseries)  #for converting the data to time series
amts<-ts(airline$Passengers,frequency = 12,start=c(1995))
View(amts)
plot(amts)

train<-amts[1:84]
test<-amts[85:96]
train<-ts(train,frequency = 12)
test<-ts(test,frequency = 12)

hw_abg<-HoltWinters(train,alpha = 0.2,beta = 0.15,gamma = 0.05) #HoltWinters with specified alpha, beta and gamma values
hw_abg
hwabg_pred<-data.frame(predict(hw_abg,n.ahead = 12))
plot(forecast(hw_abg,h=12))
hwabg_mape<-MAPE(hwabg_pred$fit,test)*100

hw_nabg<-HoltWinters(train) #HoltWinters with optimal alpha, beta and gamma values
hw_nabg
hwnabg_pred<-data.frame(predict(hw_nabg,n.ahead =12))
plot(forecast(hw_nabg,h=12))
hwnabg_mape<-MAPE(hwnabg_pred$fit,test)*100

df_mape<-data.frame(c("hwabg_mape","hwnabg_mape"),c(hwabg_mape,hwnabg_mape))
colnames(df_mape)<-c("MAPE","VALUES")
View(df_mape)

new_model <- HoltWinters(amts) #HoltWinters with optimal alpha, beta and gamma values has the least mean absolute percentage error
forecast_new <- data.frame(predict(new_model,n.ahead=12))
plot(forecast(new_model,n.ahead=12))

########## Model Based Forecasting ###########
X<- data.frame(outer(rep(month.abb,length = 96), month.abb,"==") + 0 )# Creating dummies for 12 months
View(X)
colnames(X)<-month.abb # Assigning month names 
View(X)
airline1<-cbind(airline,X)

airline1["t"]<- 1:96
airline1["t_square"]<-airline1["t"]*airline1["t"]
attach(airline1)

train<-airline1[1:84,-c(1)]
test<-airline1[85:96,-c(1)]

linear_model<-lm(Passengers~t,data=train) #linear model
summary(linear_model)
linear_pred<-data.frame(predict(linear_model,interval='predict',newdata =test))
rmse_linear<-sqrt(mean((test$Passengers-linear_pred$fit)^2,na.rm = T))
rmse_linear

Quad_model<-lm(Passengers~t+t_square,data=train) #quadratic model
summary(Quad_model)
Quad_pred<-data.frame(predict(Quad_model,interval='predict',newdata=test))
rmse_Quad<-sqrt(mean((test$Passengers-Quad_pred$fit)^2,na.rm=T))
rmse_Quad

Add_sea_Linear_model<-lm(Passengers~t+Jan+Feb+Mar+Apr+May+Jun+Jul+Aug+Sep+Oct+Nov,data=train) # linear model with additive seasonality
summary(Add_sea_Linear_model)
Add_sea_Linear_pred<-data.frame(predict(Add_sea_Linear_model,interval='predict',newdata=test))
rmse_Add_sea_Linear<-sqrt(mean((test$Passengers-Add_sea_Linear_pred$fit)^2,na.rm=T))
rmse_Add_sea_Linear

Add_sea_Quad_model<-lm(Passengers~t+t_square+Jan+Feb+Mar+Apr+May+Jun+Jul+Aug+Sep+Oct+Nov,data=train) # quadratic model with additive seasonality
summary(Add_sea_Quad_model)
Add_sea_Quad_pred<-data.frame(predict(Add_sea_Quad_model,interval='predict',newdata=test))
rmse_Add_sea_Quad<-sqrt(mean((test$Passengers-Add_sea_Quad_pred$fit)^2,na.rm=T))
rmse_Add_sea_Quad

table_rmse<-data.frame(c("rmse_linear","rmse_Quad","rmse_Add_sea_Linear","rmse_Add_sea_Quad"),c(rmse_linear,rmse_Quad,rmse_Add_sea_Linear,rmse_Add_sea_Quad))
colnames(table_rmse)<-c("model","RMSE")
View(table_rmse)

new_model <- lm(Passengers~t+t_square+Jan+Feb+Mar+Apr+May+Jun+Jul+Aug+Sep+Oct+Nov,data=airline1)
pred_new<-data.frame(predict(new_model,newdata=airline1,interval = 'predict'))
View(pred_new)
plot(pred_new$fit,type = "o")