import numpy as np
import matplotlib.pyplot as plt

adv_mask = np.load("final_output.npy").reshape(299,299,3)

plt.imshow(adv_mask)
plt.show()
