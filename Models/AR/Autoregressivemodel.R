# Load necessary libraries
library(forecast)


ar_model <- auto.arima(pythondf$Load_DA[(35065):(43825)],d=0,max.q=0,max.p = 30*24,ic="aic")
forecast_values <- predict(ar_model, n.ahead = 192)$pred
pre_df <- data.frame("Day1" = forecast_values)
tru_df <- data.frame("Day1"=pythondf$Load_DA[43826:(43825+192)])
mean(abs(pre_df$Day1-tru_df$Day1))
for(d in 1:358){
  ar_model <- auto.arima(pythondf$Load_DA[(35065*24*d):(43825+24*d)],d=0,max.q=0,max.p = 60*24,ic="aic")
  # Forecast the next 192 hours (8 days)
  forecast_values <- predict(ar_model, n.ahead = 192)$pred
  pre_df <- cbind(pre_df,forecast_values)
  tru_df <- data.frame(tru_df,pythondf$Load_DA[(43826+24*d):(43825+192+24*d)])
  print(d)
}
mape <- c()
medae <- c()
for (i in 1:ncol(tru_df)) {
  a <- abs(tru_df[,i]-pre_df[,i])/tru_df[,i]
  mape[i] <- mean(a)
 # medae[i] <- median(abs(a))
  #mape <-  mean(a)* 100
}
