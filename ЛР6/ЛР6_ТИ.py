import numpy as np

SEED = 20
N = 10
EPS = 1e-06

INIT_RANGE = (1, 20)
U_RANGE = (0, 100)
V_RANGE = (-100, 0)

rng = np.random.default_rng(SEED)


def fmt(x, d=3):
    return "(" + "; ".join(f"{v:.{d}f}" for v in x) + ")"


def print_matrix(A, name="A", d=3):
    print(f"{name} =")
    for row in A:
        print("  " + " ".join(f"{v:8.{d}f}" for v in row))


def stochastic_matrix(n):
    A = rng.random((n, n))
    return A / A.sum(axis=1, keepdims=True)


def consensus(A, x0, eps=EPS, max_iter=100000):
    x = x0.astype(float).copy()
    for t in range(1, max_iter + 1):
        x_new = A @ x
        if x_new.max() - x_new.min() < eps:
            return x_new, t
        x = x_new
    raise RuntimeError("Сходимость не достигнута")


def influence_vector(A, eps=1e-12, max_iter=100000):
    r = np.ones(A.shape[0]) / A.shape[0]
    for _ in range(max_iter):
        r_new = r @ A
        if np.max(np.abs(r_new - r)) < eps:
            return r_new / r_new.sum()
        r = r_new
    raise RuntimeError("Вектор влияния не найден")


def choose_agents(n):
    agents = np.arange(n)
    k1 = rng.integers(1, n // 3 + 1)
    F = rng.choice(agents, size=k1, replace=False)
    rest = np.setdiff1d(agents, F)
    k2 = rng.integers(1, min(n // 3, len(rest)) + 1)
    S = rng.choice(rest, size=k2, replace=False)
    return np.sort(F), np.sort(S)


def agent_nums(a):
    return ", ".join(str(i + 1) for i in a)


A = stochastic_matrix(N)
print("Сгенерированная стохастическая матрица доверия")
print_matrix(A, "A", 3)
row_sums = A.sum(axis=1)
print("\nПроверка стохастичности:")
print("Суммы строк:", fmt(row_sums, 3))
print("Максимальное отклонение от 1:", f"{np.max(np.abs(row_sums - 1)):.3f}")
#print("Все элементы положительные:", np.all(A > 0))

x0 = rng.integers(INIT_RANGE[0], INIT_RANGE[1] + 1, size=N).astype(float)
x_final, it = consensus(A, x0)
r = influence_vector(A)
A_inf = np.tile(r, (N, 1))
X = r @ x0
print("\nРасчет итогового мнения без информационного влияния")
print("x(0) =", fmt(x0, 0))
print(f"x({it}) =", fmt(x_final, 3))
print("\nВектор итоговой влиятельности агентов:")
print("r =", fmt(r, 3))
print("\nПредельная матрица A^∞:")
print_matrix(A_inf, "A^∞", 3)
print("\nИтоговое мнение без влияния:")
print("X =", f"{X:.6f}")

F, S = choose_agents(N)
u = rng.integers(U_RANGE[0], U_RANGE[1] + 1)
v = rng.integers(V_RANGE[0], V_RANGE[1] + 1)
x_controlled = x0.copy()
x_controlled[F] = u
x_controlled[S] = v
x_controlled_final, it2 = consensus(A, x_controlled)
neutral = np.setdiff1d(np.arange(N), np.union1d(F, S))
r_F = r[F].sum()
r_S = r[S].sum()
X_0 = np.sum(r[neutral] * x0[neutral])
X_uv = r_F * u + r_S * v + X_0
print("\nМоделирование информационного влияния")
print("Агенты первого игрока:", agent_nums(F))
print("Агенты второго игрока:", agent_nums(S))
print("u =", u)
print("v =", v)
print("\nНачальные мнения с учетом влияния:")
print("x(0) =", fmt(x_controlled, 0))
print(f"\nРезультирующие мнения после {it2} итераций:")
print(f"x({it2}) =", fmt(x_controlled_final, 3))
print("\nРасчет по формуле X(u, v) = r_F u + r_S v + X^0:")
print("r_F =", f"{r_F:.3f}")
print("r_S =", f"{r_S:.3f}")
print("X^0 =", f"{X_0:.3f}")
print("X(u, v) =", f"{X_uv:.3f}")
print("\nПроверка:")
print("Итерационное итоговое мнение:", f"{x_controlled_final.mean():.3f}")
print("Итоговое мнение по формуле:", f"{X_uv:.3f}")
#print("Разница:", f"{abs(x_controlled_final.mean() - X_uv):.10f}")

target_1 = u
target_2 = v
dist_1 = abs(X_uv - target_1)
dist_2 = abs(X_uv - target_2)
print("\nОпределение победителя")
print("Желаемый результат первого игрока:", target_1)
print("Расстояние до цели первого игрока:", f"{dist_1:.3f}")
print("Желаемый результат второго игрока:", target_2)
print("Расстояние до цели второго игрока:", f"{dist_2:.3f}")
if dist_1 < dist_2:
    print("Выиграл первый игрок.")
elif dist_2 < dist_1:
    print("Выиграл второй игрок.")
else:
    print("Ничья.")
