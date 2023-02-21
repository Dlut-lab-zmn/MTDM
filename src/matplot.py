import matplotlib.pyplot as plt
allX = [i for i in range(5)]
37.11, 36.58,36.94
MRR_1 = [36.73, , , , 36.74]
28.23,27.68,
40.30,40.15
Hint_1 = [27.48, 28.05,,,27.97]
Hint_3 = [40.54,40.31,,,40.11]
54.36,54.28
Hint_10 = [54.99,54.78,,,54.40]
# 总体度的分布
plt.figure()
#plt.scatter(allX, Hint_1, label="MRR", linestyle=":")

# advisee度的折线图分布
#plt.plot(allX, Hint_3,label="advisee度的分布", linestyle="--")

# advisor度的折线图分布
plt.plot(allX, Hint_10, label="advisor度的分布", linestyle="-.")
plt.legend()
plt.title("")
plt.xlabel("Cycle : j")
plt.ylabel("频次")
plt.show()
