import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from statsmodels.tsa.holtwinters import SimpleExpSmoothing
from statsmodels.tsa.holtwinters import Holt
from statsmodels.tsa.holtwinters import ExponentialSmoothing
import statsmodels.graphics.tsaplots as tsa_plots

plasticData = pd.read_csv("C:/My Files/Excelr/17 - Forecasting/Assignment/PlasticSales.csv")
plasticData.shape
plasticData.head(13)
plasticData.columns

plasticData.Month[0].split('-')[0] +'-19'+ plasticData.Month[0].split('-')[1]
for i in range(0,60):
    p = plasticData.Month[i]
    plasticData['Month'][i]=p.split('-')[0] +'-19' + p.split('-')[1]
plasticData.Sales.plot()
plasticData.index = pd.to_datetime(plasticData.Month,format="%b-%Y")

######################## MODEL BASED APPROACH ##########################

plasticData['month_name']=0
for i in range(0,60):
    p = plasticData.Month[i]
    plasticData['month_name'][i]= p[0:3]
month_dummies = pd.get_dummies(plasticData['month_name'])
plasticData_dumm = pd.concat([plasticData,month_dummies],axis=1)

plasticData_dumm['t'] = np.arange(1,61)
plasticData_dumm["t_squared"] = plasticData_dumm["t"]*plasticData_dumm["t"]
plasticData_dumm["log_Sales"] = np.log(plasticData_dumm["Sales"])
plasticData_dumm = plasticData_dumm.drop('Month',axis=1)

Train = plasticData_dumm.head(48)
Test = plasticData_dumm.tail(12)

import statsmodels.formula.api as smf
linear_model = smf.ols('Sales ~ t',data = Train).fit() # Linear
linear_model.summary()
pred_linear =  pd.Series(linear_model.predict(pd.DataFrame(Test['t'])))
rmse_linear = np.sqrt(np.mean((np.array(Test['Sales'])-np.array(pred_linear))**2))
rmse_linear

Exp = smf.ols('log_Sales~t',data=Train).fit() # Exponential
Exp.summary()
pred_Exp = pd.Series(Exp.predict(pd.DataFrame(Test['t'])))
rmse_Exp = np.sqrt(np.mean((np.array(Test['Sales'])-np.array(np.exp(pred_Exp)))**2))
rmse_Exp

Quad = smf.ols('Sales~t+t_squared',data=Train).fit() # Quadratic
pred_Quad = pd.Series(Quad.predict(Test[["t","t_squared"]]))
rmse_Quad = np.sqrt(np.mean((np.array(Test['Sales'])-np.array(pred_Quad))**2))
rmse_Quad

add_sea = smf.ols('Sales~Jan+Feb+Mar+Apr+May+Jun+Jul+Aug+Sep+Oct+Nov',data=Train).fit() # Additive seasonality
pred_add_sea = pd.Series(add_sea.predict(Test[['Jan','Feb','Mar','Apr','May','Jun','Jul','Aug','Sep','Oct','Nov']]))
rmse_add_sea = np.sqrt(np.mean((np.array(Test['Sales'])-np.array(pred_add_sea))**2))
rmse_add_sea

add_sea_Quad = smf.ols('Sales~t+t_squared+Jan+Feb+Mar+Apr+May+Jun+Jul+Aug+Sep+Oct+Nov',data=Train).fit() # Additive Seasonality Quadratic
pred_add_sea_quad = pd.Series(add_sea_Quad.predict(Test[['Jan','Feb','Mar','Apr','May','Jun','Jul','Aug','Sep','Oct','Nov','t','t_squared']]))
rmse_add_sea_quad = np.sqrt(np.mean((np.array(Test['Sales'])-np.array(pred_add_sea_quad))**2))
rmse_add_sea_quad

Mul_sea = smf.ols('log_Sales~Jan+Feb+Mar+Apr+May+Jun+Jul+Aug+Sep+Oct+Nov',data = Train).fit() # Multiplicative Seasonality
pred_Mult_sea = pd.Series(Mul_sea.predict(Test))
rmse_Mult_sea = np.sqrt(np.mean((np.array(Test['Sales'])-np.array(np.exp(pred_Mult_sea)))**2))
rmse_Mult_sea

Mul_Add_sea = smf.ols('log_Sales~t+Jan+Feb+Mar+Apr+May+Jun+Jul+Aug+Sep+Oct+Nov',data = Train).fit() # Multiplicative Additive Seasonality
pred_Mult_add_sea = pd.Series(Mul_Add_sea.predict(Test))
rmse_Mult_add_sea = np.sqrt(np.mean((np.array(Test['Sales'])-np.array(np.exp(pred_Mult_add_sea)))**2))
rmse_Mult_add_sea

Mul_Add_qa_sea = smf.ols('log_Sales~t+t_squared+Jan+Feb+Mar+Apr+May+Jun+Jul+Aug+Sep+Oct+Nov',data = Train).fit() # Multiplicative Additive Quadratic Seasonality
pred_Mult_add_qa_sea = pd.Series(Mul_Add_sea.predict(Test))
rmse_Mult_add_qa_sea = np.sqrt(np.mean((np.array(Test['Sales'])-np.array(np.exp(pred_Mult_add_qa_sea)))**2))
rmse_Mult_add_qa_sea

data = {"MODEL":pd.Series(["rmse_linear","rmse_Exp","rmse_Quad","rmse_add_sea","rmse_add_sea_quad","rmse_Mult_sea","rmse_Mult_add_sea","Mul_Add_qa_sea"]),"RMSE_Values":pd.Series([rmse_linear,rmse_Exp,rmse_Quad,rmse_add_sea,rmse_add_sea_quad,rmse_Mult_sea,rmse_Mult_add_sea,rmse_Mult_add_qa_sea])}
table_rmse=pd.DataFrame(data)
table_rmse

plt.plot(data['MODEL'],data['RMSE_Values'])
plt.scatter(data['MODEL'],data['RMSE_Values'])

##################### DATA DRIVEN APPROACH #########################

plasticData["Date"] = pd.to_datetime(plasticData.Month,format="%b-%Y")
plasticData["month"] = plasticData.Date.dt.strftime("%b")
plasticData["year"] = plasticData.Date.dt.strftime("%Y")

heatmap_y_month = pd.pivot_table(data=plasticData,values="Sales",index="year",columns="month",aggfunc="mean",fill_value=0)
sns.heatmap(heatmap_y_month,annot=True,fmt="g")

sns.boxplot(x="month",y="Sales",data=plasticData)
sns.boxplot(x="year",y="Sales",data=plasticData)
sns.lineplot(x="year",y="Sales",hue="month",data=plasticData)

plasticData.Sales.plot(label="Sales")
for i in range(2,24,6):
    plasticData["Sales"].rolling(i).mean().plot(label=str(i))
plt.legend(loc=3)

tsa_plots.plot_acf(plasticData.Sales,lags=10)
tsa_plots.plot_pacf(plasticData.Sales)

Train = plasticData.head(48)
Test = plasticData.tail(12)

def MAPE(pred,org):
    temp = np.abs((pred-org))*100/org
    return np.mean(temp)

# Simple Exponential Method
ses_model = SimpleExpSmoothing(Train["Sales"]).fit()
pred_ses = ses_model.predict(start = Test.index[0],end = Test.index[-1])
MAPE(pred_ses,Test.Sales)

# Holt method 
hw_model = Holt(Train["Sales"]).fit()
pred_hw = hw_model.predict(start = Test.index[0],end = Test.index[-1])
MAPE(pred_hw,Test.Sales)

tendArr = ["add", "mul", "additive", "multiplicative"]
sesonalArr = ["add", "mul", "additive", "multiplicative"]
mapeArr = [];

for i in tendArr:
    for j in sesonalArr:
        hwe_model_add_add = ExponentialSmoothing(Train["Sales"],seasonal=j,trend=i,seasonal_periods=12,damped=True).fit()
        pred_hwe_add_add = hwe_model_add_add.predict(start = Test.index[0],end = Test.index[-1])
        mapeArr.append([i,j,MAPE(pred_hwe_add_add,Test.Sales)])
