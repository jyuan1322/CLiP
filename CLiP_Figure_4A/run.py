import numpy as np
import matplotlib.pyplot as plt

coefs2 = [-409.52448861055984,
          431.6513313140901,
          38.00764396532995]
coefs4 = [-540.0797504985338,
          -360.26557093065634,
          -44.21132533248787,
          732.5017720012906,
          234.54891527783096]
coefs6 = [-272.72479315758346,
          484.8418136747146,
          -270.8586221944605,
          -185.17269248431845,
          -426.3074761490136,
          582.3584611990045,
          88.95092357490671]

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

