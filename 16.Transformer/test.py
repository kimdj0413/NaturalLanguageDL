import FinanceDataReader as fdr
import matplotlib.pyplot as plt
import seaborn as sns

kakao = '035720'
stock = fdr.DataReader(kakao)
print(stock)
plt.figure(figsize=(16,9))
sns.lineplot(y=stock['Close'], x=stock.index)
plt.xlabel('Time')
plt.ylabel('kakao')
plt.show()