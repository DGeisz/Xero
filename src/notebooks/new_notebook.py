# Add cell to make it so that autoreloading works
# %%
%load_ext autoreload
%autoreload 2

# %%
import torch

# %%
# from src.master.BasicModel import BasicModel

from src.master import BasicModel


# from two import a

# print(a)

# %%

# %% [markdown]
# ## Hello there
# This is what I think the very best way to do things is $\frac{a}{b}$

# %%
import matplotlib.pyplot as plt
import numpy as np

# %%
# Create a plot of quadratic function
x = np.arange(-5, 5, 0.1)
y = x**2

plt.plot(x, y)
plt.show()


# %%
# Print python path
import sys

print(sys.path)

# %%
import os

print(os.getcwd())

# %%
from src.master import training_data

# %%

img_index = 2

data = training_data[img_index][0]

print(data.shape)

plt.imshow(data.reshape(28, 28, 1), cmap="Greys")

# %%

training_data[img_index][1]

# %%
from src.master import BasicModel

model = BasicModel()

model.draw_first_layer()

# %%
a = model.layer_1[0].weight.data

# %%
model.draw_first_layer_diff()





