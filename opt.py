from math import sqrt
from numpy import asarray, arange, meshgrid
from numpy.random import rand, seed
from matplotlib import pyplot

def objective(x, y):
    return x**2.0 + y**2.0 

def derivative(x, y):
    return asarray([x * 2.0, y * 2.0])

def adam(objective, derivative, bounds, n_iter, alpha, beta1, beta2, eps=1e-8):
    solutions, losses = [], []
    x = bounds[:, 0] + rand(len(bounds)) * (bounds[:, 1] - bounds[:, 0])
    m = [0.0 for _ in range(bounds.shape[0])]
    v = [0.0 for _ in range(bounds.shape[0])]

    for t in range(n_iter):
        g = derivative(x[0], x[1])
        for i in range(bounds.shape[0]):
            m[i] = beta1 * m[i] + (1.0 - beta1) * g[i]
            v[i] = beta2 * v[i] + (1.0 - beta2) * g[i]**2
            mhat = m[i] / (1.0 - beta1**(t + 1))
            vhat = v[i] / (1.0 - beta2**(t + 1))
            x[i] = x[i] - alpha * mhat / (sqrt(vhat) + eps)
        solutions.append(x.copy())
        losses.append(objective(x[0], x[1]))
    return solutions, losses

def modified_adam(objective, derivative, bounds, n_iter, alpha, beta1, beta2, eps=1e-8):
    solutions, losses = [], []
    x = bounds[:, 0] + rand(len(bounds)) * (bounds[:, 1] - bounds[:, 0])
    m = [0.0 for _ in range(bounds.shape[0])]
    k = [0.0 for _ in range(bounds.shape[0])]
    v = [0.0 for _ in range(bounds.shape[0])]

    for t in range(n_iter):
        delta_w = derivative(x[0], x[1])
        for i in range(bounds.shape[0]):
            m[i] = beta1 * m[i] + (1.0 - beta1) * delta_w[i]
            k[i] = beta2 * k[i] + (1.0 - beta2) * (delta_w[i]**2)
            v[i] = beta2 * v[i] + alpha * m[i] / (sqrt(k[i]) + eps)
            w_look_ahead = x[i] - m[i]
            x[i] = x[i] - alpha * delta_w[i] / (sqrt(k[i]) + eps)
        solutions.append(x.copy())
        losses.append(objective(x[0], x[1]))
    return solutions, losses

# seed(1)
bounds = asarray([[-1.0, 1.0], [-1.0, 1.0]])
n_iter = 60
alpha = 0.02
beta1, beta2 = 0.8, 0.999

solutions_adam, losses_adam = adam(objective, derivative, bounds, n_iter, alpha, beta1, beta2)
solutions_modified_adam, losses_modified_adam = modified_adam(objective, derivative, bounds, n_iter, alpha, beta1, beta2)

print(f"Final loss (Adam): {losses_adam[-1]:.5f}")
print(f"Final loss (Madam): {losses_modified_adam[-1]:.5f}")

xaxis = arange(bounds[0, 0], bounds[0, 1], 0.1)
yaxis = arange(bounds[1, 0], bounds[1, 1], 0.1)
x, y = meshgrid(xaxis, yaxis)
results = objective(x, y)

pyplot.figure(figsize=(12, 5))

pyplot.subplot(1, 2, 1)
pyplot.contourf(x, y, results, levels=50, cmap='jet')
solutions_adam = asarray(solutions_adam)
solutions_modified_adam = asarray(solutions_modified_adam)
pyplot.plot(solutions_adam[:, 0], solutions_adam[:, 1], 'o-', color='white', label='Original Adam')
pyplot.plot(solutions_modified_adam[:, 0], solutions_modified_adam[:, 1], 'o-', color='cyan', label='Mdam')
pyplot.title("Contour Plot with Optimizer Paths")
pyplot.legend()

pyplot.subplot(1, 2, 2)
pyplot.plot(range(n_iter), losses_adam, label="Adam", color="blue")
pyplot.plot(range(n_iter), losses_modified_adam, label="Madam", color="green")
pyplot.title("Loss vs Epochs")
pyplot.xlabel("Epochs")
pyplot.ylabel("Loss")
pyplot.legend()
pyplot.tight_layout()
pyplot.show()
