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


# Проверка на седловую точку
def saddle_point(matrix):
    row_mins = matrix.min(axis=1)
    col_maxs = matrix.max(axis=0)

    alpha = row_mins.max()
    beta = col_maxs.min()

    return row_mins, col_maxs, alpha, beta


# Аналитическое решение
def analytical_solution(matrix):
    """
    x* = (u * C^-1) / (u * C^-1 * u^T)
    y* = (C^-1 * u^T) / (u * C^-1 * u^T)
    v  = 1 / (u * C^-1 * u^T)
    """
    n = matrix.shape[0]
    u_row = np.ones((1, n), dtype=float)
    u_col = np.ones((n, 1), dtype=float)

    det = np.linalg.det(matrix)
    if abs(det) < 1e-12:
        raise ValueError("Матрица вырождена, метод обратной матрицы неприменим.")

    inv_matrix = np.linalg.inv(matrix)
    denominator = float((u_row @ inv_matrix @ u_col)[0, 0])

    x = (u_row @ inv_matrix).flatten() / denominator
    y = (inv_matrix @ u_col).flatten() / denominator
    v = 1.0 / denominator

    return inv_matrix, x, y, v


def brown_robinson(matrix, eps=0.1, start_a=0, start_b=0, max_steps=10000):
    m, n = matrix.shape

    count_A = [0] * m
    count_B = [0] * n

    win_A = [0] * m
    loss_B = [0] * n

    a = start_a
    b = start_b

    best_upper = None
    best_lower = None
    current_eps = None

    table_rows = []

    for k in range(1, max_steps + 1):
        count_A[a] += 1
        count_B[b] += 1

        for i in range(m):
            win_A[i] += int(matrix[i, b])

        for j in range(n):
            loss_B[j] += int(matrix[a, j])

        current_upper = max(win_A) / k
        current_lower = min(loss_B) / k

        if best_upper is None or current_upper < best_upper:
            best_upper = current_upper

        if best_lower is None or current_lower > best_lower:
            best_lower = current_lower

        current_eps = best_upper - best_lower

        table_rows.append({
            "k": k,
            "A_choice": f"x{a + 1}",
            "B_choice": f"y{b + 1}",
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

        a = first_argmax(win_A)
        b = first_argmin(loss_B)

    return {
        "rows": table_rows,
        "iterations": len(table_rows),
        "count_A": count_A,
        "count_B": count_B,
        "best_upper": best_upper,
        "best_lower": best_lower,
        "eps": current_eps,
    }


def print_brown_robinson_table(result, first_n=10, last_n=5):
    rows = result["rows"]

    if len(rows) <= first_n + last_n:
        selected_rows = rows
    else:
        selected_rows = rows[:first_n] + [None] + rows[-last_n:]

    data_rows = []
    for r in selected_rows:
        if r is None:
            data_rows.append(["...", "...", "...", "...", "...", "...", "...", "...", "...", "...", "...", "..."])
        else:
            data_rows.append([
                str(r["k"]),
                r["A_choice"],
                r["B_choice"],
                str(r["win_A"][0]),
                str(r["win_A"][1]),
                str(r["win_A"][2]),
                str(r["loss_B"][0]),
                str(r["loss_B"][1]),
                str(r["loss_B"][2]),
                fmt4(r["upper"]),
                fmt4(r["lower"]),
                fmt4(r["eps"]),
            ])

    subheader = ["", "  A  ", "  B  ", "x1", "x2", "x3", "y1", "y2", "y3", "", "", ""]

    widths = [0] * 12
    for i in range(12):
        widths[i] = max(
            len(subheader[i]),
            *(len(row[i]) for row in data_rows)
        )

    widths[0] = max(widths[0], len("k"))
    widths[9] = max(widths[9], len("1/k v̄[k]"))
    widths[10] = max(widths[10], len("1/k v̲[k]"))
    widths[11] = max(widths[11], len("eps"))

    def cell(text, width):
        return text.center(width)

    def span_width(start, end):
        return sum(widths[start:end + 1]) + 3 * (end - start)

    def make_group_row():
        row = [
            cell("k", widths[0]),
            cell("Выбор игрока", span_width(1, 2)),
            cell("Выигрыш игрока A", span_width(3, 5)),
            cell("Проигрыш игрока B", span_width(6, 8)),
            cell(" 1/k v̄[k]", widths[9]),
            cell(" 1/k v̲[k]", widths[10]),
            cell("eps", widths[11]),
        ]
        return " | ".join(row)

    def make_subheader_row():
        return " | ".join(cell(subheader[i], widths[i]) for i in range(12))

    def make_data_row(items):
        return " | ".join(cell(items[i], widths[i]) for i in range(12))

    def make_separator():
        return "-+-".join("-" * w for w in widths)

    print("\nТаблица метода Брауна—Робинсон\n")
    print(make_group_row())
    print(make_subheader_row())
    print(make_separator())

    for row in data_rows:
        print(make_data_row(row))


def print_final_estimates(result):
    k = result["iterations"]
    count_A = result["count_A"]
    count_B = result["count_B"]

    x_br = [c / k for c in count_A]
    y_br = [c / k for c in count_B]

    print("\nИтог метода Брауна—Робинсон:")
    print(f"Число итераций: {k}")
    print(f"x~[{k}] = {vector_decimal_str(x_br)}")
    print(f"y~[{k}] = {vector_decimal_str(y_br)}")
    print(f"Нижняя оценка: {fmt4(result['best_lower'])}")
    print(f"Верхняя оценка: {fmt4(result['best_upper'])}")
    print(f"eps[{k}] = {fmt4(result['eps'])}")


def print_analytical_result(matrix):
    inv_matrix, x, y, v = analytical_solution(matrix)

    print("\nАналитическое решение")
    print("Обратная матрица C^-1:")
    print(np.round(inv_matrix, 4))

    print("\nОптимальная стратегия игрока A:")
    print(f"x* = {vector_decimal_str(x)}")

    print("\nОптимальная стратегия игрока B:")
    print(f"y* = {vector_decimal_str(y)}")

    print("\nЦена игры:")
    print(f"v = {fmt4(v)}")

    return inv_matrix, x, y, v


def print_comparison(x, y, v, result):
    k = result["iterations"]
    count_A = result["count_A"]
    count_B = result["count_B"]

    x_br = [c / k for c in count_A]
    y_br = [c / k for c in count_B]

    lower = float(result["best_lower"])
    upper = float(result["best_upper"])
    middle = (lower + upper) / 2.0

    print("\nСравнение результатов")
    print("Аналитический метод:")
    print(f"x* = ({fmt4(x[0])}, {fmt4(x[1])}, {fmt4(x[2])})")
    print(f"y* = ({fmt4(y[0])}, {fmt4(y[1])}, {fmt4(y[2])})")
    print(f"v  = {fmt4(v)}")

    print("\nМетод Брауна—Робинсон:")
    print(f"x~[{k}] = ({fmt4(x_br[0])}, {fmt4(x_br[1])}, {fmt4(x_br[2])})")
    print(f"y~[{k}] = ({fmt4(y_br[0])}, {fmt4(y_br[1])}, {fmt4(y_br[2])})")
    print(f"v ∈ [{fmt4(lower)}; {fmt4(upper)}]")
    print(f"Средняя оценка цены игры: {fmt4(middle)}")


def print_brown_robinson_checks(result, tol=1e-9):
    k = result["iterations"]
    x_br = np.array([c / k for c in result["count_A"]], dtype=float)
    y_br = np.array([c / k for c in result["count_B"]], dtype=float)

    lower = float(result["best_lower"])
    upper = float(result["best_upper"])
    v_br = (lower + upper) / 2.0

    print("\nПроверки:\n")

    print("Проверка неотрицательности вероятностей:")
    print("Игрок 1:")
    for p in x_br:
        sign = ">" if p >= -tol else "<"
        print(f"{fmt4(p)} {sign} 0")

    print("Игрок 2:")
    for p in y_br:
        sign = ">" if p >= -tol else "<"
        print(f"{fmt4(p)} {sign} 0")

    sx = x_br.sum()
    sy = y_br.sum()


    print("\nПроверка суммы вероятностей:")
    print(
        f"Игрок 1: "
        f"{fmt4(x_br[0])} + {fmt4(x_br[1])} + {fmt4(x_br[2])} = {sx:.4f}"
    )
    print(
        f"Игрок 2: "
        f"{fmt4(y_br[0])} + {fmt4(y_br[1])} + {fmt4(y_br[2])} = {sy:.4f}"
    )

    sign_v = ">" if lower > tol else "<="
    print("\nПроверка цены игры:")
    print(f"Нижняя оценка v = {fmt4(lower)} {sign_v} 0")


def main():
    # Вариант 1
    matrix = np.array([
        [1, 11, 11],
        [7,  5,  8],
        [16, 6,  2]
    ], dtype=int)

    print("Матрица игры:")
    print(matrix)

    row_mins, col_maxs, alpha, beta = saddle_point(matrix)

    print("\nПроверка на седловую точку:")
    print("Минимумы по строкам:", row_mins)
    print("Максимумы по столбцам:", col_maxs)
    print(f"alpha = {fmt4(alpha)}")
    print(f"beta  = {fmt4(beta)}")

    if alpha == beta:
        print(f"\nЕсть седловая точка. Цена игры v = {fmt4(alpha)}")
        return
    else:
        print("\nСедловой точки нет. Игра решается в смешанных стратегиях.")

    inv_matrix, x, y, v = print_analytical_result(matrix)

    result = brown_robinson(
        matrix=matrix,
        eps=0.1,
        start_a=0,
        start_b=0,
        max_steps=10000
    )

    print_brown_robinson_table(result, first_n=10, last_n=5)
    print_final_estimates(result)
    print_comparison(x, y, v, result)
    print_brown_robinson_checks(result)


if __name__ == "__main__":
    main()
