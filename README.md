# Reproduction of documented use cases of Convolutional Neural Networks in predictive modelling of financial markets

The following literature was used to identify different tecniques / methods :

[Algorithmic Financial Trading with Deep Convolutional Neural Networks: Time Series to Image Conversion Approach](https://www.researchgate.net/publication/324802031_Algorithmic_Financial_Trading_with_Deep_Convolutional_Neural_Networks_Time_Series_to_Image_Conversion_Approach)

[A deep learning based stock trading model with 2-D CNN trend detection](https://www.researchgate.net/publication/323131323_A_deep_learning_based_stock_trading_model_with_2-D_CNN_trend_detection)

[A quantitative trading method using deep convolution neural network](https://iopscience.iop.org/article/10.1088/1757-899X/490/4/042018/pdf)

[DeepLOB: Deep Convolutional Neural Networks for Limit Order Books](https://www.oxford-man.ox.ac.uk/wp-content/uploads/2020/03/DeepLOB-Deep-Convolutional-Neural-Networks-for-Limit-Order-Books.pdf)

[Forecasting Stock Prices from the Limit OrderBook using Convolutional Neural Networks](https://www.researchgate.net/publication/319220815_Forecasting_Stock_Prices_from_the_Limit_Order_Book_Using_Convolutional_Neural_Networks)

[Financial Trading Model with Stock Bar Chart Image Time Series with Deep Convolutional Neural Networks](https://arxiv.org/pdf/1903.04610)

Models to be used :

VGG-16
VGG-19
Resnet 18
Resnet 50
Resnet 101
Resnet 152

Tickers to be tested on :

'SPY', 'TLT', 'XLF', 'XLE', 'XLU', 'XLK', 'EXW1.DE', '^FTSE', '1329.T'

## Indicators considered :

Trend :

MACD : EMA12 - EMA26 based on 2 EMAs, with periods of 26 and 12
EMA : SMA used for 1st calc, for the rest we have EMA curr = Pcurr * α + (1-α) * EMA prev, where α in [0,1].
SMA : Pav / n, where Pav is the mean price of n previous price candles
WMA : ((P1*W1)+(P2*W2)+...+(Pn*Wn)) / (W1+W2+...+Wn), where Pi is price of previous ith day, and Wi is the weighting factor assigned to it

Momentum :

RSI : 100 - 100/(1+RS), where RS = mean gain / mean loss of previous n days

Volatility :

ATR : (PrevATR + True Range) / n, or (1/n) * Σ True Range i from 1/0 to n

Volume :

MFI : 100 - 100 / (1 + 4D Positive-Money-Fow / 14D Negative-Money-Flow)

Others (not implemented):

Sinewave Indicator ~ advanced turn point warning
MESA Adaptive Moving Average (MAMA) : haha