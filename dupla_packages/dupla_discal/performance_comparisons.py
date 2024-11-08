import timeit
import numpy as np
import pandas as pd
from scipy import optimize
from numba import njit


def func1(
    freeParams: tuple[float],
    points: pd.DataFrame,
    polynomialN=3,
    S=0.9,
    R=np.array([[0.99, 0.01], [-0.01, 0.99]]),
    T=np.array([0.0, 0.1]),
    rigid=True,
):
    # create the powers for the polynomial as tuple, a generator would get used up by next comprehension
    powers = tuple(
        (i, j) for i in range(polynomialN + 1) for j in range(polynomialN + 1)
    )
    if rigid:
        points = points.apply(
            lambda row: pd.Series(S * R @ row + T, index=["x", "y"]),
            axis="columns",
            result_type="expand",
        )
    # first half coefficients belong to x
    xCalc = sum(
        freeParams[n] * pow(points.x, i) * pow(points.y, j)
        for n, (i, j) in enumerate(powers)
    )
    yCalc = sum(
        freeParams[polynomialN // 2 + n] * pow(points.x, i) * pow(points.y, j)
        for n, (i, j) in enumerate(powers)
    )
    # when naming it x and y we can directly subtract the corresponding true columns in loss()
    return pd.DataFrame({"x": xCalc, "y": yCalc})


def func2(
    freeParams: tuple[float],
    points: pd.DataFrame,
    powers,
    polynomialN=3,
    S=0.9,
    R=np.array([[0.99, 0.01], [-0.01, 0.99]]),
    T=np.array([0.0, 0.1]),
    rigid=True,
):
    """supply powers as argument"""
    if rigid:
        points = points.apply(
            lambda row: pd.Series(S * R @ row + T, index=["x", "y"]),
            axis="columns",
            result_type="expand",
        )
    xCalc = sum(
        freeParams[n] * pow(points.x, i) * pow(points.y, j)
        for n, (i, j) in enumerate(powers)
    )
    yCalc = sum(
        freeParams[polynomialN // 2 + n] * pow(points.x, i) * pow(points.y, j)
        for n, (i, j) in enumerate(powers)
    )
    return pd.DataFrame({"x": xCalc, "y": yCalc})


def func3(
    points: pd.DataFrame,
    cX,
    xY,
    S=0.9,
    R=np.array([[0.99, 0.01], [-0.01, 0.99]]),
    T=np.array([0.0, 0.1]),
    rigid=True,
):
    if rigid:
        points = points.apply(
            lambda row: pd.Series(S * R @ row + T, index=["x", "y"]),
            axis="columns",
            result_type="expand",
        )
    xCalc = np.polynomial.polynomial.polyval2d(points.x, points.y, cX)
    yCalc = np.polynomial.polynomial.polyval2d(points.x, points.y, cY)
    return pd.DataFrame({"x": xCalc, "y": yCalc})


def func1_loss(freeparams, points):
    """loss and jacobian"""
    res = func1(freeparams, points)
    res = ((res + np.random.rand(*res.shape) - res) ** 2).to_numpy().sum()
    return res


def gradient(freeparams, points, polynomialN=3):
    powers = tuple(
        (i, j) for i in range(polynomialN + 1) for j in range(polynomialN + 1)
    )
    pred = func1(freeparams, points)
    x = [pow(points.x, i) * pow(points.y, j) for n, (i, j) in enumerate(powers)]
    y = [pow(points.x, i) * pow(points.y, j) for n, (i, j) in enumerate(powers)]
    gradX = list(map(lambda a: pred.x * a, x))
    gradY = list(map(lambda a: pred.y * a, y))
    grad = np.array(gradX + gradY)
    return grad.sum(axis=1)


@njit
def func4(
    freeParams,
    powers,
    points,
    out,
    S=0.9,
    R=np.array([[0.99, 0.01], [-0.01, 0.99]]),
    T=np.array([0.0, 0.1]),
):
    """use jit, powers is now an array with each row being the power of x and the power y"""
    # apply rigid registration
    for n in range(points.shape[0]):
        # dumb doing matrix multiplication likes this
        out[n, 0] = R[0, 0] * points[n, 0] + R[0, 1] * points[n, 1] + T[0]
        out[n, 1] = R[1, 0] * points[n, 0] + R[1, 1] * points[n, 1] + T[1]
    for n in range(points.shape[0]):
        temp = np.zeros(powers.shape)
        for p in range(powers.shape[0]):
            temp[p, 0] = (
                freeParams[p]
                * pow(out[n, 0], powers[p, 0])
                * pow(out[n, 1], powers[p, 1])
            )
            temp[p, 1] = (
                freeParams[len(freeParams) // 2 + p]
                * pow(out[n, 0], powers[p, 0])
                * pow(out[n, 1], powers[p, 1])
            )
        out[n, 0] = np.sum(temp[:, 0])
        out[n, 1] = np.sum(temp[:, 1])
    return out


if __name__ == "__main__":
    params = np.random.rand(32)
    runs = 100
    points = pd.DataFrame(np.random.rand(1600, 2), columns=["x", "y"], dtype=np.float32)
    # powers inside the function
    # print(timeit.timeit("func1(params, points)", number=runs, globals=globals()))
    # print(timeit.timeit("func1(params, points, rigid=True)", number=runs, globals=globals()))
    # take powers out
    powers = tuple((i, j) for i in range(4) for j in range(4))
    # print(timeit.timeit("func2(params, points, powers)", number=runs, globals=globals()))
    # print(timeit.timeit("func2(params, points, powers, rigid=True)", number=runs, globals=globals()))
    powerArray = np.array(powers)
    breakpoint()
    res = func4(
        params, powerArray, points.values, np.zeros(points.shape, dtype=np.float32)
    )
    print(
        timeit.timeit(
            """func4(params, powerArray, points.values, np.zeros(points.shape, dtype=np.float32))""",
            number=runs,
            globals=globals(),
        )
    )
    # use polyval2d
    cX = np.random.rand(4, 4)
    cY = np.random.rand(4, 4)
    print(timeit.timeit("func3(points, cX, cY)", number=runs, globals=globals()))
    print(
        timeit.timeit(
            "func3(points, cX, cY, rigid=True)", number=runs, globals=globals()
        )
    )
    s = """optimize.minimize(
            func1_loss,
            x0=params,
            args=points,
            method="Nelder-Mead",
            options={"disp":4},
            jac=gradient,
            callback=lambda a: print("hi"))
        """
    print(timeit.timeit(s, number=1, globals=globals()))
