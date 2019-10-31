import numpy as np
import matplotlib.pyplot as plt

coefs2 = [-391.31065702248884,
          321.73537044851355,
          75.77307898929996]
coefs4 = [-223.92972608943492,
          -481.0127584074366,
          -485.02707252322165,
          884.206607293954,
          347.13589272897286]
coefs6 = [709.9802994997349,
          -122.38243480614074,
          -1037.6487525641326,
          -836.6217751283207,
          181.65062178074518,
          829.0039446975911,
          276.4022598776502]

def plot_coefs(coefs, col, lstyle, label):
    pc_range = np.linspace(0,1,500,endpoint=False)
    polynom = np.poly1d(coefs, r=False)
    vals = [polynom(x) for x in pc_range]
    vals /= np.sum(vals)
    plt.plot(pc_range, vals,
                color=col, linestyle=lstyle,
                label=label)

fig, ax = plt.subplots()
ax.spines['right'].set_visible(False)
ax.spines['top'].set_visible(False)
plot_coefs(coefs2, 'cyan', '-', 'deg 2')
plot_coefs(coefs4, 'deepskyblue', '--', 'deg 4')
plot_coefs(coefs6, 'cornflowerblue', '-.', 'deg 6')
plt.legend(frameon=False)
plt.ylim([0,0.003])
plt.xlabel("PRS percentile")
plt.ylabel("Individual correlation weight")
plt.savefig("learned_polynom_funcs.eps", format="eps", dpi=1000)
plt.show()

