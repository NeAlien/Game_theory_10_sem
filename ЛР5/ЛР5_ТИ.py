from itertools import combinations
from fractions import Fraction
from math import factorial

I = (1, 2, 3, 4)


v0 = {
    frozenset(): 0,
    frozenset([1]): 4, frozenset([2]): 1,
    frozenset([3]): 3, frozenset([4]): 1,
    frozenset([1, 2]): 6, frozenset([1, 3]): 8,
    frozenset([1, 4]): 6, frozenset([2, 3]): 5,
    frozenset([2, 4]): 3, frozenset([3, 4]): 5,
    frozenset([1, 2, 3]): 9, frozenset([1, 2, 4]): 8,
    frozenset([1, 3, 4]): 10, frozenset([2, 3, 4]): 7,
    frozenset([1, 2, 3, 4]): 11,
}


def coals():
    return [frozenset(c) for r in range(len(I) + 1) for c in combinations(I, r)]


def cs(S):
    return "∅" if not S else "{" + ",".join(map(str, sorted(S))) + "}"


def fs(x):
    if isinstance(x, Fraction):
        return str(x.numerator) if x.denominator == 1 else f"{x.numerator}/{x.denominator}"
    return str(x)


def section(title):
    print("\n" + title)
    print("-" * len(title))


def print_v(v, title):
    section(title)
    for S in coals():
        print(f"  v({cs(S):>9}) -> {v[S]}")


def check_essential(v):
    N = frozenset(I)
    s = sum(v[frozenset([i])] for i in I)
    section("Проверка существенности")
    print(f"  v(I) = {v[N]}")
    print(f"  Сумма одиночных коалиций = {s}")
    if v[N] > s:
        print("  Вывод: игра существенная.")
    else:
        print("  Вывод: игра несущественная.")


def super_violations(v):
    res, used = [], set()
    for S in coals():
        for T in coals():
            if not S or not T or not S.isdisjoint(T):
                continue
            key = tuple(sorted([tuple(sorted(S)), tuple(sorted(T))]))
            if key in used:
                continue
            used.add(key)
            if v[S | T] < v[S] + v[T]:
                res.append((S, T))
    return res


def convex_violations(v):
    res, used = [], set()
    for S in coals():
        for T in coals():
            key = tuple(sorted([tuple(sorted(S)), tuple(sorted(T))]))
            if key in used:
                continue
            used.add(key)
            if v[S] + v[T] > v[S | T] + v[S & T]:
                res.append((S, T))
    return res


def print_super(v):
    section("Проверка супераддитивности")
    viol = super_violations(v)
    if not viol:
        print("  Нарушений не найдено.")
        print("  Вывод: игра супераддитивна.")
        return
    print(f"  Найдено нарушений: {len(viol)}")
    for num, (S, T) in enumerate(viol, 1):
        print(f"  {num}) v({cs(S | T)}) = {v[S | T]} < "
              f"v({cs(S)}) + v({cs(T)}) = {v[S]} + {v[T]} = {v[S] + v[T]}")


def print_convex(v):
    section("Проверка выпуклости")
    viol = convex_violations(v)
    if not viol:
        print("  Нарушений не найдено.")
        print("  Вывод: игра выпуклая.")
        return
    print(f"  Игра не является выпуклой. Нарушений: {len(viol)}")
    for num, (S, T) in enumerate(viol, 1):
        print(f"  {num}) v({cs(S)}) + v({cs(T)}) = {v[S]} + {v[T]} = {v[S] + v[T]} > "
              f"v({cs(S | T)}) + v({cs(S & T)}) = "
              f"{v[S | T]} + {v[S & T]} = {v[S | T] + v[S & T]}")


def make_super(v):
    v = v.copy()
    changes = []
    changed = True
    while changed:
        changed = False
        for S in coals():
            for T in coals():
                if not S or not T or not S.isdisjoint(T):
                    continue
                U = S | T
                need = v[S] + v[T]
                if v[U] < need:
                    old = v[U]
                    v[U] = need
                    changes.append((U, old, need, S, T))
                    changed = True
    section("Исправление характеристической функции")
    if not changes:
        print("  Исправления не требуются: исходная игра уже супераддитивна.")
    else:
        for num, (U, old, new, S, T) in enumerate(changes, 1):
            print(f"  {num}) v({cs(U)}): {old} -> {new}; "
                  f"основание: v({cs(S)}) + v({cs(T)}) = {new}")
    return v


def shapley(v):
    n = len(I)
    phi = {i: Fraction(0, 1) for i in I}
    details = {i: [] for i in I}
    for i in I:
        others = [j for j in I if j != i]
        for r in range(n):
            for S_tuple in combinations(others, r):
                S = frozenset(S_tuple)
                w = Fraction(factorial(r) * factorial(n - r - 1), factorial(n))
                inc = v[S | frozenset([i])] - v[S]
                add = w * inc
                phi[i] += add
                details[i].append((S, w, inc, add))
    section("Расчет вектора Шепли")
    for i in I:
        print(f"\nИгрок {i}")
        for S, w, inc, add in details[i]:
            print(f"  S = {cs(S):>7}; вес = {fs(w):>4}; "
                  f"прирост = {inc:>2}; вклад = {fs(add)}")
        print(f"  Итого: x{i}(v) = {fs(phi[i])} ≈ {float(phi[i]):.4f}")
    print("\nИтоговый вектор Шепли")
    print("  Точно:       (" + "; ".join(fs(phi[i]) for i in I) + ")")
    print("  Десятично:   (" + "; ".join(f"{float(phi[i]):.4f}" for i in I) + ")")
    return phi


def rationality(v, phi):
    N = frozenset(I)
    section("Проверка индивидуальной рационализации")
    for i in I:
        one = frozenset([i])
        sign = ">=" if phi[i] >= v[one] else "<"
        print(f"  x{i}(v) = {fs(phi[i])} {sign} v({cs(one)}) = {v[one]}")
    section("Проверка групповой рационализации")
    total = sum(phi.values())
    print(f"  Сумма компонент вектора Шепли = {fs(total)}")
    print(f"  v(I) = {v[N]}")
    if total == v[N]:
        print("  Вывод: групповая рационализация выполняется.")
    else:
        print("  Вывод: групповая рационализация нарушается.")


if __name__ == "__main__":
    print_v(v0, "Исходная характеристическая функция, вариант 1")
    check_essential(v0)
    print_super(v0)
    print_convex(v0)
    v = make_super(v0)
    print_v(v, "Характеристическая функция после проверки супераддитивности")
    print_super(v)
    phi = shapley(v)
    rationality(v, phi)
