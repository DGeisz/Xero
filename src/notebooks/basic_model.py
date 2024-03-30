# %%
%load_ext autoreload
%autoreload 2


# %%
import torch

from torch import nn
from src.master import (
    BasicModel,
    ModelTrainer,
    train_dataloader,
    test_dataloader,
    training_data,
)

# %%
model = BasicModel(10, 10)

# %%
optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)

trainer = ModelTrainer(
    model,
    loss_fn=nn.CrossEntropyLoss(),
    optimizer=optimizer,
    train_loader=train_dataloader,
    test_loader=test_dataloader,
)

trainer.train(10)

# %%
model.draw_data_overlay(training_data[2][0])


# %%
trainer.train(5)

# %%
trainer.model.draw_first_layer()

# %%
trainer.model.draw_first_layer_diff()

# %%
sgd_model = BasicModel(10, 10)

new_optim = torch.optim.SGD(sgd_model.parameters())

new_trainer = ModelTrainer(
    sgd_model,
    loss_fn=nn.CrossEntropyLoss(),
    optimizer=new_optim,
    train_loader=train_dataloader,
    test_loader=test_dataloader,
)

# %%
new_trainer.train(100)

# %%
sgd_model.draw_first_layer_diff()

sgd_model.draw_first_layer()

# %%
model.draw_first_layer()

# %%
torch.flatten(training_data[0][0]).shape


# %%
a = model.layer_1[0].weight * torch.flatten(training_data[0][0])


# %%
model.draw_data_overlay(training_data[0][0])


