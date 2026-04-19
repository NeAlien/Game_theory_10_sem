import numpy as np


def fmt4(value):
    return f"{float(value):.4f}"


def vector_decimal_str(values):
    return "(" + ", ".join(fmt4(v) for v in values) + ")"


def first_argmax(values):
    best_i = 0
    best_v = values[0]
    for i in range(1, len(values)):
        if values[i] > best_v:
            best_v = values[i]
            best_i = i
    return best_i


def first_argmin(values):
    best_i = 0
    best_v = values[0]
    for i in range(1, len(values)):
        if values[i] < best_v:
            best_v = values[i]
            best_i = i
    return best_i


a = -5.0
b = 5.0 / 12.0
c = 10.0 / 3.0
d = -2.0 / 3.0
e = -4.0 / 3.0

EPS = 0.01
K_STEPS = 4
MAX_N = 50

BR_EPS = 0.1
BR_MAX_STEPS = 10000

USE_COLOR = True
RESET = "\033[0m"
SADDLE_COLOR = "\033[30;102m"
TITLE_COLOR = "\033[1;36m"
INFO_COLOR = "\033[1;33m"


def colorize(text, color):
    if USE_COLOR:
        return f"{color}{text}{RESET}"
    return text


def H(x, y):
    return a * x * x + b * y * y + c * x * y + d * x + e * y


class ContinuousGameSolver:
    def __init__(self, a_coef, b_coef, c_coef, d_coef, e_coef):
        self.a = a_coef
        self.b = b_coef
        self.c = c_coef
        self.d = d_coef
        self.e = e_coef
        self.gridstep = None

    def setka_maker(self, N):
        x = np.linspace(0.0, 1.0, N + 1)
        y = np.linspace(0.0, 1.0, N + 1)
        self.gridstep = x

        X, Y = np.meshgrid(x, y)
        grid = np.column_stack([X.ravel(), Y.ravel()])
        return grid

    def solve(self, N):
        grid = self.setka_maker(N)
        x_vals = grid[:, 0]
        y_vals = grid[:, 1]

        z_vals = (
            self.a * x_vals ** 2
            + self.b * y_vals ** 2
            + self.c * x_vals * y_vals
            + self.d * x_vals
            + self.e * y_vals
        )

        size = len(self.gridstep)
        return z_vals.reshape(size, size).T


def find_saddle_points(matrix, tol=1e-10):
    matrix = np.array(matrix, dtype=float)

    row_mins = matrix.min(axis=1)
    col_maxs = matrix.max(axis=0)

    alpha = row_mins.max()
    beta = col_maxs.min()

    if not np.isclose(alpha, beta, atol=tol):
        return [], alpha, beta, row_mins, col_maxs

    saddle_points = []
    rows, cols = matrix.shape

    for i in range(rows):
        for j in range(cols):
            value = matrix[i, j]
            if np.isclose(value, row_mins[i], atol=tol) and np.isclose(value, col_maxs[j], atol=tol):
                saddle_points.append((i, j, value))

    return saddle_points, alpha, beta, row_mins, col_maxs


def analytical_solution():
    matrix_A = np.array([
        [2 * a, c],
        [c, 2 * b]
    ], dtype=float)

    vector_B = np.array([-d, -e], dtype=float)

    try:
        x_star, y_star = np.linalg.solve(matrix_A, vector_B)
        h_star = H(x_star, y_star)
        return x_star, y_star, h_star
    except np.linalg.LinAlgError:
        return None, None, None


def brown_robinson(matrix, eps=0.1, start_a=0, start_b=0, max_steps=10000):
    m, n = matrix.shape

    count_A = [0] * m
    count_B = [0] * n

    win_A = [0] * m
    loss_B = [0] * n

    a_choice = start_a
    b_choice = start_b

    best_upper = None
    best_lower = None
    current_eps = None

    table_rows = []

    for k in range(1, max_steps + 1):
        count_A[a_choice] += 1
        count_B[b_choice] += 1

        for i in range(m):
            win_A[i] += matrix[i, b_choice]

        for j in range(n):
            loss_B[j] += matrix[a_choice, j]

        current_upper = max(win_A) / k
        current_lower = min(loss_B) / k

        if best_upper is None or current_upper < best_upper:
            best_upper = current_upper

        if best_lower is None or current_lower > best_lower:
            best_lower = current_lower

        current_eps = best_upper - best_lower

        table_rows.append({
            "k": k,
            "A_choice": f"x{a_choice + 1}",
            "B_choice": f"y{b_choice + 1}",
            "win_A": win_A.copy(),
            "loss_B": loss_B.copy(),
            "upper": current_upper,
            "lower": current_lower,
            "eps": current_eps,
            "count_A": count_A.copy(),
            "count_B": count_B.copy(),
        })

        if current_eps <= eps:
            break

        a_choice = first_argmax(win_A)
        b_choice = first_argmin(loss_B)

    return {
        "rows": table_rows,
        "iterations": len(table_rows),
        "count_A": count_A,
        "count_B": count_B,
        "best_upper": best_upper,
        "best_lower": best_lower,
        "eps": current_eps,
    }


def print_pretty_matrix(matrix, saddle_points=None):
    matrix = np.array(matrix, dtype=float)
    rows, cols = matrix.shape

    saddle_set = set()
    if saddle_points:
        for i, j, _ in saddle_points:
            saddle_set.add((i, j))

    row_index_width = max(4, len(str(rows - 1)) + 1)
    max_val_len = max(len(f"{matrix[i, j]:.4f}") for i in range(rows) for j in range(cols))
    cell_width = max(9, max_val_len + 2)

    title = "Матрица игры"
    if saddle_points:
        title += " (выделена седловая точка)"
    title += ":"

    print()
    print(colorize(title, TITLE_COLOR))

    header = " " * row_index_width
    for j in range(cols):
        header += f"{j:>{cell_width}}"
    print(header)

    for i in range(rows):
        line = f"{i:>{row_index_width}}"
        for j in range(cols):
            value_text = f"{matrix[i, j]:>{cell_width}.4f}"
            if (i, j) in saddle_set:
                value_text = colorize(value_text, SADDLE_COLOR)
            line += value_text
        print(line)


def print_derivative_formulas():
    print("\nЧастные производные функции H(x,y):")
    print(f"  ∂H/∂x = {fmt4(2 * a)}x + {fmt4(c)}y {('- ' + fmt4(abs(d))) if d < 0 else ('+ ' + fmt4(d))}")
    print(f"  ∂H/∂y = {fmt4(c)}x + {fmt4(2 * b)}y {('- ' + fmt4(abs(e))) if e < 0 else ('+ ' + fmt4(e))}")
    print()


def print_brown_robinson_summary(result, grid):
    k = result["iterations"]

    x_br = np.array([cnt / k for cnt in result["count_A"]], dtype=float)
    y_br = np.array([cnt / k for cnt in result["count_B"]], dtype=float)

    x_mean = float(np.dot(x_br, grid))
    y_mean = float(np.dot(y_br, grid))

    lower = float(result["best_lower"])
    upper = float(result["best_upper"])
    middle = (lower + upper) / 2.0

    print("\nИтог метода Брауна—Робинсон:")
    print(f"Число итераций: {k}")
    print(f"Нижняя оценка: {fmt4(lower)}")
    print(f"Верхняя оценка: {fmt4(upper)}")
    print(f"Координаты по смешанным стратегиям: x={fmt4(x_mean)} y={fmt4(y_mean)}")
    print(f"Средняя оценка цены игры: H={fmt4(middle)}")

    return x_mean, y_mean, middle


def main():
    solver = ContinuousGameSolver(a, b, c, d, e)

    print("\n" + "=" * 60)
    print(colorize("АНАЛИТИЧЕСКОЕ РЕШЕНИЕ", TITLE_COLOR))
    print("=" * 60)

    print_derivative_formulas()

    x_analyt, y_analyt, h_analyt = analytical_solution()

    if x_analyt is not None:
        print(f"Стационарная точка: x* = {x_analyt:.4f}, y* = {y_analyt:.4f}")
        print(f"Значение функции: H* = {h_analyt:.4f}")
    else:
        print("Аналитическое решение недоступно.")
    print()

    print("=" * 60)
    print(colorize("РЕШЕНИЕ ИГРЫ ДЛЯ N ОТ 2 ДО 50", TITLE_COLOR))
    print("=" * 60)

    H_history = []
    last_numeric_x = None
    last_numeric_y = None
    last_numeric_h = None
    last_N = None

    for N in range(2, MAX_N + 1):
        print(f"\n{'=' * 60}")
        print(colorize(f"N = {N} (шаг сетки = 1/{N})", INFO_COLOR))
        print(f"{'=' * 60}")

        matrix = solver.solve(N)
        saddle_points, alpha, beta, row_mins, col_maxs = find_saddle_points(matrix)

        print_pretty_matrix(matrix, saddle_points)

        if saddle_points:
            row_idx, col_idx, _ = saddle_points[0]
            x_current = solver.gridstep[row_idx]
            y_current = solver.gridstep[col_idx]
            H_current = matrix[row_idx, col_idx]

            print("\nЕсть седловая точка:")
            print(f"x={x_current:.4f} y={y_current:.4f} H={H_current:.4f}")
        else:
            print("\nСедловой точки нет, решение методом Брауна-Робинсона:")

            result = brown_robinson(
                matrix=matrix,
                eps=BR_EPS,
                start_a=0,
                start_b=0,
                max_steps=BR_MAX_STEPS
            )

            x_current, y_current, H_current = print_brown_robinson_summary(result, solver.gridstep)

        H_history.append(float(H_current))

        last_numeric_x = x_current
        last_numeric_y = y_current
        last_numeric_h = H_current
        last_N = N

        if len(H_history) >= K_STEPS:
            recent_H = H_history[-K_STEPS:]
            h_spread = max(recent_H) - min(recent_H)

            if h_spread <= EPS:
                print()
                print(f"Разброс H за последние {K_STEPS} шагов: {h_spread:.4f} <= {EPS:.4f}")
                print("Условие завершения выполнено.")
                break

    print("\n" + "=" * 60)
    print(colorize("СРАВНЕНИЕ АНАЛИТИЧЕСКОГО И ЧИСЛЕННОГО РЕШЕНИЙ", TITLE_COLOR))
    print("=" * 60)

    if x_analyt is not None:
        print(f"Аналитическое решение: x={x_analyt:.4f} y={y_analyt:.4f} H={h_analyt:.4f}")

    if last_numeric_x is not None:
        print(f"Численное решение:    x={last_numeric_x:.4f} y={last_numeric_y:.4f} H={last_numeric_h:.4f}")
        print(f"Последнее N = {last_N}")

    if x_analyt is not None and last_numeric_x is not None:
        print(f"|Δx| = {abs(x_analyt - last_numeric_x):.4f}")
        print(f"|Δy| = {abs(y_analyt - last_numeric_y):.4f}")
        print(f"|ΔH| = {abs(h_analyt - last_numeric_h):.4f}")


if __name__ == "__main__":
    main()