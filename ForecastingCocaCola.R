library(readxl)
CocaCola<-read_xlsx(file.choose())
View(CocaCola)

library(tseries)
salesamts<-ts(CocaCola$Sales,frequency = 4,start=c(1986))
View(salesamts)
plot(salesamts) #The sales is plotted

train<-salesamts[1:38]
test<-salesamts[39:42]
train<-ts(train,frequency = 4)
test<-ts(test,frequency = 4)

library(forecast)
library(fpp)
library(smooth)

hw_abg<-HoltWinters(train,alpha = 0.2,beta = 0.15,gamma = 0.05) #HoltWinters using specified alpha, beta and gamma values
hw_abg
hwabg_pred<-data.frame(predict(hw_abg,n.ahead = 4))
plot(forecast(hw_abg,h=4))
hwabg_mape<-MAPE(hwabg_pred$fit,test)*100

hw_nabg<-HoltWinters(train) #HoltWinters using optimal alpha, beta and gamma values
hw_nabg
hwnabg_pred<-data.frame(predict(hw_nabg,n.ahead = 4))
plot(forecast(hw_nabg,h=4))
hwnabg_mape<-MAPE(hwnabg_pred$fit,test)*100

df_mape<-data.frame(c("hwabg_mape","hwnabg_mape"),c(hwabg_mape,hwnabg_mape))
colnames(df_mape)<-c("MAPE","VALUES")
View(df_mape)

new_model <- HoltWinters(salesamts) #HoltWinters with optimal alpha, beta and gamma values has least MAPE
forecast_new <- data.frame(predict(new_model,n.ahead=4))
plot(forecast(new_model,n.ahead=4))

########## Model Based Forecasting ###########
Q1 <-  ifelse(grepl("Q1",CocaCola$Quarter),'1','0')
Q2 <-  ifelse(grepl("Q2",CocaCola$Quarter),'1','0')
Q3 <-  ifelse(grepl("Q3",CocaCola$Quarter),'1','0')
Q4 <-  ifelse(grepl("Q4",CocaCola$Quarter),'1','0')
cocacola1<-cbind(CocaCola,Q1,Q2,Q3,Q4)
cocacola1["t"]<- 1:42
cocacola1["t_square"]<-cocacola1["t"]*cocacola1["t"]
attach(cocacola1)

train<-cocacola1[1:36,]
test<-cocacola1[37:40,]

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

Add_sea_Linear_model<-lm(Sales~t+Q1+Q2+Q3+Q4,data=train) # linear model with additive seasonality
summary(Add_sea_Linear_model)
Add_sea_Linear_pred<-data.frame(predict(Add_sea_Linear_model,interval='predict',newdata=test))
rmse_Add_sea_Linear<-sqrt(mean((test$Sales-Add_sea_Linear_pred$fit)^2,na.rm=T))
rmse_Add_sea_Linear

Add_sea_Quad_model<-lm(Sales~t+t_square+Q1+Q2+Q3+Q4,data=train) # quadratic model with additive seasonality
summary(Add_sea_Quad_model)
Add_sea_Quad_pred<-data.frame(predict(Add_sea_Quad_model,interval='predict',newdata=test))
rmse_Add_sea_Quad<-sqrt(mean((test$Sales-Add_sea_Quad_pred$fit)^2,na.rm=T))
rmse_Add_sea_Quad

table_rmse<-data.frame(c("rmse_linear","rmse_Quad","rmse_Add_sea_Linear","rmse_Add_sea_Quad"),c(rmse_linear,rmse_Quad,rmse_Add_sea_Linear,rmse_Add_sea_Quad))
colnames(table_rmse)<-c("model","RMSE")
View(table_rmse)

new_model <- lm(Sales~t+t_square+Q1+Q2+Q3+Q4,data=cocacola1)
pred_new<-data.frame(predict(new_model,newdata=cocacola1,interval = 'predict'))
View(pred_new)
plot(pred_new$fit,type = "o")
