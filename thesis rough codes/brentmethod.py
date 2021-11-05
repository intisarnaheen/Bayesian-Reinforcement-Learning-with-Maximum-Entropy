from scipy import optimize
def f(x):
    return (x**3)+(x**2)+x-2
x_min = optimize.minimize_scalar(f, bounds=[-10, 10], method='bounded')
print(x_min)
