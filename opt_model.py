import torch  # type:ignore 
import torch.nn as nn  # type: ignore
import torch.optim as optim  # type: ignore
import matplotlib.pyplot as plt  # type:ignore

class SimpleNet(nn.Module):
    def __init__(self, num_features):
        super(SimpleNet, self).__init__()
        self.fc1 = nn.Linear(num_features, 50)
        self.fc2 = nn.Linear(50, 20)
        self.fc3 = nn.Linear(20, 2)

    def forward(self, x):
        x = torch.relu(self.fc1(x))
        x = torch.relu(self.fc2(x))
        return self.fc3(x)

class Madam(torch.optim.Optimizer):
    def __init__(self, params, lr=0.01, beta1=0.9, beta2=0.999, eps=1e-8):
        defaults = dict(lr=lr, beta1=beta1, beta2=beta2, eps=eps)
        super(Madam, self).__init__(params, defaults)

    def step(self, closure=None):
        loss = None
        if closure is not None:
            loss = closure()

        for group in self.param_groups:
            for p in group['params']:
                if p.grad is None:
                    continue
                grad = p.grad.data
                state = self.state[p]

                if len(state) == 0:
                    state['m'] = torch.zeros_like(p.data)
                    state['k'] = torch.zeros_like(p.data)
                    state['v'] = torch.zeros_like(p.data)

                m, k, v = state['m'], state['k'], state['v']
                beta1, beta2, eps, lr = group['beta1'], group['beta2'], group['eps'], group['lr']

                m.mul_(beta1).add_(1 - beta1, grad)
                k.mul_(beta2).addcmul_(1 - beta2, grad, grad)
                v.mul_(beta1).addcdiv_(1 - beta1, grad / (torch.sqrt(k) + eps), v)

                look_ahead = p.data - lr * m / (torch.sqrt(k) + eps)
                p.data.add_(look_ahead - p.data)

        return loss

def train_model(model, optimizer, inputs, targets, num_epochs=50):
    criterion = nn.MSELoss()
    losses = []
    
    for epoch in range(num_epochs):
        optimizer.zero_grad()
        outputs = model(inputs)
        loss = criterion(outputs, targets)
        loss.backward()
        optimizer.step()
        losses.append(loss.item())
    
    return losses

num_samples = 10000
num_features = 1000

inputs = torch.randn(num_samples, num_features)
targets = torch.randn(num_samples, 2)
# print("targets: ", targets)

def create_optimizer(opt_name, model):
    if opt_name == "Madam":
        return Madam(model.parameters(), lr=0.001)
    elif opt_name == "SGD":
        return optim.SGD(model.parameters(), lr=0.001)
    elif opt_name == "Adam":
        return optim.Adam(model.parameters(), lr=0.001)
    elif opt_name == "RMSprop":
        return optim.RMSprop(model.parameters(), lr=0.001)
    elif opt_name == "Nadam":
        return optim.NAdam(model.parameters(), lr=0.001, betas=(0.9, 0.999), weight_decay=0.01)
    

optimizers = ["Madam", "Adam", "RMSprop"]

num_epochs = 100
results = {}

for opt_name in optimizers:
    model = SimpleNet(num_features)  
    optimizer = create_optimizer(opt_name, model) 
    losses = train_model(model, optimizer, inputs, targets, num_epochs=num_epochs)
    results[opt_name] = losses
    print(f"{opt_name} training completed.")

plt.figure(figsize=(10, 6))
for opt_name, losses in results.items():
    plt.plot(losses, label=opt_name)

print("----------------------------------")
print("number of samples: ",num_samples)
print("number of features: ", num_features)
print("Madam", min(results["Madam"]))
print("Adam", min(results["Adam"]))
print("RMSprop", min(results["RMSprop"]))

my_adam_loss = min(results["Madam"])
adam_loss = min(results["Adam"])
rms_prop_loss = min(results["RMSprop"])

# data = f'''
# number of samples: {num_samples}
# number of features:{num_features}
# Madam: {my_adam_loss}
# Adam: {adam_loss}
# RMSprop: {rms_prop_loss}
# '''
data = ''

plt.xlabel("Epoch")
plt.ylabel("Loss")
plt.title("Optimizer Comparison on SimpleNet")
plt.text(num_epochs - 20, max(max(losses) for losses in results.values()) * 0.8, data, fontsize=12, color='black', bbox=dict(facecolor='white', alpha=0.5))
plt.legend()
plt.show()
