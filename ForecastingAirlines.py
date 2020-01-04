import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from statsmodels.tsa.seasonal import seasonal_decompose
from statsmodels.tsa.holtwinters import SimpleExpSmoothing
from statsmodels.tsa.holtwinters import Holt
from statsmodels.tsa.holtwinters import ExponentialSmoothing
import statsmodels.graphics.tsaplots as tsa_plots

airlineData = pd.read_excel("C:/My Files/Excelr/17 - Forecasting/Assignment/Airlines+Data.xlsx")
airlineData.shape
airlineData.columns
airlineData.Month[0]
airlineData.Passengers.plot()

#################### MODEL BASED APPROACH ###################

pd.to_datetime(airlineData.Month[0],format="%m")
airlineData.Month[0].strftime("%b")
airlineData['month_name']= 0

#adding new column with Month Name for converting to dummy variables
for i in range(0,96):
    p = airlineData.Month[i]
    airlineData['month_name'][i]= airlineData.Month[i].strftime("%b")    

month_dummies = pd.get_dummies(airlineData['month_name'])
airlineData_dumm = pd.concat([airlineData,month_dummies],axis=1)

airlineData_dumm['t'] = np.arange(1,97)
airlineData_dumm["t_squared"] = airlineData_dumm["t"]*airlineData_dumm["t"]
airlineData_dumm["log_Passenger"] = np.log(airlineData_dumm["Passengers"])
airlineData_dumm = airlineData_dumm.drop('Month',axis=1)
airlineData_dumm.shape
airlineData_dumm.columns

Train = airlineData_dumm.head(84)
Test = airlineData_dumm.tail(12)

import statsmodels.formula.api as smf
linear_model = smf.ols('Passengers ~ t',data = Train).fit() #Linear
linear_model.summary()
pred_linear =  pd.Series(linear_model.predict(pd.DataFrame(Test['t'])))
rmse_linear = np.sqrt(np.mean((np.array(Test['Passengers'])-np.array(pred_linear))**2))
rmse_linear

Exp = smf.ols('log_Passenger~t',data=Train).fit() #Exponential
Exp.summary()
pred_Exp = pd.Series(Exp.predict(pd.DataFrame(Test['t'])))
rmse_Exp = np.sqrt(np.mean((np.array(Test['Passengers'])-np.array(np.exp(pred_Exp)))**2))
rmse_Exp

Quad = smf.ols('Passengers~t+t_squared',data=Train).fit() #Quadratic
pred_Quad = pd.Series(Quad.predict(Test[["t","t_squared"]]))
rmse_Quad = np.sqrt(np.mean((np.array(Test['Passengers'])-np.array(pred_Quad))**2))
rmse_Quad

add_sea = smf.ols('Passengers~Jan+Feb+Mar+Apr+May+Jun+Jul+Aug+Sep+Oct+Nov',data=Train).fit() #Additive Seasonality
pred_add_sea = pd.Series(add_sea.predict(Test[['Jan','Feb','Mar','Apr','May','Jun','Jul','Aug','Sep','Oct','Nov']]))
rmse_add_sea = np.sqrt(np.mean((np.array(Test['Passengers'])-np.array(pred_add_sea))**2))
rmse_add_sea

add_sea_Quad = smf.ols('Passengers~t+t_squared+Jan+Feb+Mar+Apr+May+Jun+Jul+Aug+Sep+Oct+Nov',data=Train).fit() #Quadratic with addititve seasonality
pred_add_sea_quad = pd.Series(add_sea_Quad.predict(Test[['Jan','Feb','Mar','Apr','May','Jun','Jul','Aug','Sep','Oct','Nov','t','t_squared']]))
rmse_add_sea_quad = np.sqrt(np.mean((np.array(Test['Passengers'])-np.array(pred_add_sea_quad))**2))
rmse_add_sea_quad

Mul_sea = smf.ols('log_Passenger~Jan+Feb+Mar+Apr+May+Jun+Jul+Aug+Sep+Oct+Nov',data = Train).fit() #Multiplicative Seasonality
pred_Mult_sea = pd.Series(Mul_sea.predict(Test))
rmse_Mult_sea = np.sqrt(np.mean((np.array(Test['Passengers'])-np.array(np.exp(pred_Mult_sea)))**2))
rmse_Mult_sea

Mul_Add_sea = smf.ols('log_Passenger~t+Jan+Feb+Mar+Apr+May+Jun+Jul+Aug+Sep+Oct+Nov',data = Train).fit() #Multiplicative Additive Seasonality
pred_Mult_add_sea = pd.Series(Mul_Add_sea.predict(Test))
rmse_Mult_add_sea = np.sqrt(np.mean((np.array(Test['Passengers'])-np.array(np.exp(pred_Mult_add_sea)))**2))
rmse_Mult_add_sea 

Mul_Add_qa_sea = smf.ols('log_Passenger~t+t_squared+Jan+Feb+Mar+Apr+May+Jun+Jul+Aug+Sep+Oct+Nov',data = Train).fit() #Multiplicative Additive Quadratic Seasonality
pred_Mult_add_qa_sea = pd.Series(Mul_Add_sea.predict(Test))
rmse_Mult_add_qa_sea = np.sqrt(np.mean((np.array(Test['Passengers'])-np.array(np.exp(pred_Mult_add_qa_sea)))**2))
rmse_Mult_add_qa_sea 

data = {"MODEL":pd.Series(["rmse_linear","rmse_Exp","rmse_Quad","rmse_add_sea","rmse_add_sea_quad","rmse_Mult_sea","rmse_Mult_add_sea","Mul_Add_qa_sea"]),"RMSE_Values":pd.Series([rmse_linear,rmse_Exp,rmse_Quad,rmse_add_sea,rmse_add_sea_quad,rmse_Mult_sea,rmse_Mult_add_sea,rmse_Mult_add_qa_sea])}
table_rmse=pd.DataFrame(data)
table_rmse

plt.plot(data['MODEL'],data['RMSE_Values'])
plt.scatter(data['MODEL'],data['RMSE_Values'])


#################### DATA DRIVEN APPROACH ######################

#Create new columns with month, year for better understaning of data
airlineData['month_name'] = airlineData.Month.dt.strftime("%b")
airlineData['year_name'] = airlineData.Month.dt.strftime("%Y")

heatmap_y_month = pd.pivot_table(data=airlineData,values="Passengers",index="year_name",columns="month_name",aggfunc="mean",fill_value=0)
sns.heatmap(heatmap_y_month,annot=True,fmt="g")

sns.boxplot(x="month_name",y="Passengers",data=airlineData)
sns.boxplot(x="year_name",y="Passengers",data=airlineData)
sns.lineplot(x="year_name",y="Passengers",hue="month_name",data=airlineData)

airlineData.Passengers.plot(label="Passenger")

for i in range(2,24,6):
    airlineData["Passengers"].rolling(i).mean().plot(label=str(i))
    
plt.legend(loc=3)

airlineData.index = pd.to_datetime(airlineData.Month,format="%b-%y")
decompose_ts_add = seasonal_decompose(airlineData.Passengers,model="additive")
decompose_ts_add.plot()
decompose_ts_mul = seasonal_decompose(airlineData.Passengers,model="multiplicative")
decompose_ts_mul.plot()

# ACF plots and PACF plots on Original data sets 
tsa_plots.plot_acf(airlineData.Passengers,lags=10)
tsa_plots.plot_pacf(airlineData.Passengers)

Train = airlineData.head(84)
Test = airlineData.tail(12)

def MAPE(pred,org):
    temp = np.abs((pred-org))*100/org
    return np.mean(temp)

# Simple Exponential Method
ses_model = SimpleExpSmoothing(Train["Passengers"]).fit()
pred_ses = ses_model.predict(start = Test.index[0],end = Test.index[-1])
MAPE(pred_ses,Test.Passengers)

# Holt method 
hw_model = Holt(Train["Passengers"]).fit()
pred_hw = hw_model.predict(start = Test.index[0],end = Test.index[-1])
MAPE(pred_hw,Test.Passengers)
tendArr = ["add", "mul", "additive", "multiplicative"]
sesonalArr = ["add", "mul", "additive", "multiplicative"]

mapeArr = [];
for i in tendArr:
    for j in sesonalArr:
        hwe_model_add_add = ExponentialSmoothing(Train["Passengers"],seasonal=j,trend=i,seasonal_periods=12,damped=True).fit()
        pred_hwe_add_add = hwe_model_add_add.predict(start = Test.index[0],end = Test.index[-1])
        mapeArr.append([i,j,MAPE(pred_hwe_add_add,Test.Passengers)])
