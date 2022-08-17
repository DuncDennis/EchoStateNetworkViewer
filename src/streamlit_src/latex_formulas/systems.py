"""Python file to get latex equation strings for all the systems. """


def _get_3d_equation(right_side_x: str, right_side_y: str, right_side_z: str,
                     differential: bool = True) -> str:
    """A utility function to get a 3d latex differential equation.

    # TODO: decide which one.

    Args:
        right_side_x: The string of the right hand side of the x variable: after dx/dt = {...}.
        right_side_y: The string of the right hand side of the y variable: after dy/dt = {...}.
        right_side_z: The string of the right hand side of the z variable: after dz/dt = {...}.
        differential: If true assume a differential equation, if False assume a discrete map.


    Returns:
        The latex string of the whole equation.
    """

    if differential:
        string = r"""\begin{aligned}
                    & \dot x=right_side_x \\
                    & \dot y=right_side_y \\
                    & \dot z=right_side_z
                    \end{aligned}
                    """
        # string = r"""\begin{aligned}
        #             &\frac{\mathrm{d} x}{\mathrm{~d} t}=right_side_x \\
        #             &\frac{\mathrm{d} y}{\mathrm{~d} t}=right_side_y \\
        #             &\frac{\mathrm{d} z}{\mathrm{~d} t}=right_side_z
        #             \end{aligned}
        #             """

    else:
        string = r"""\begin{aligned}
                    & x_{i+1}=right_side_x \\
                    & y_{i+1}=right_side_y \\
                    & z_{i+1}=right_side_z
                    \end{aligned}
                    """
    string = string.replace("right_side_x", right_side_x)
    string = string.replace("right_side_y", right_side_y)
    string = string.replace("right_side_z", right_side_z)
    return string


def _get_2d_equation(right_side_x: str, right_side_y: str, differential: bool = True) -> str:
    """A utility function to get a 2d latex differential equation.

    # TODO: decide which one.

    Args:
        right_side_x: The string of the right hand side of the x variable: after dx/dt = {...}.
        right_side_y: The string of the right hand side of the y variable: after dy/dt = {...}.
        differential: If true assume a differential equation, if False assume a discrete map.
    Returns:
        The latex string of the whole equation.
    """

    if differential:
        string = r"""\begin{aligned}
                    & \dot x=right_side_x \\
                    & \dot y=right_side_y \\
                    \end{aligned}
                    """

        # string = r"""\begin{aligned}
        #             &\frac{\mathrm{d} x}{\mathrm{~d} t}=right_side_x \\
        #             &\frac{\mathrm{d} y}{\mathrm{~d} t}=right_side_y
        #             \end{aligned}
        #             """

    else:
        string = r"""\begin{aligned}
                    & x_{i+1}=right_side_x \\
                    & y_{i+1}=right_side_y \\
                    \end{aligned}
                    """

    string = string.replace("right_side_x", right_side_x)
    string = string.replace("right_side_y", right_side_y)
    return string


Lorenz63 = _get_3d_equation(r"\sigma(y-x)",
                            r"x(\rho-z)-y",
                            r"x y-\beta z")

Roessler = _get_3d_equation(r"-y -z",
                            r"x + ay",
                            r"b + z(x-c)")

ComplexButterfly = _get_3d_equation(r"a(y-z)",
                                    r"-z\,\text{sgn}x",
                                    r"|x| - 1")

Chen = _get_3d_equation(r"a(y-x)",
                        r"(c-a)x - xz + cy",
                        r"xy - bz")

ChuaCircuit = _get_3d_equation(r"\alpha[y - x + bx + 0.5(a-b)(|x+1| - |x-1|)]",
                               r"x - y + z",
                               r"- \beta y")

Thomas = _get_3d_equation(r"-bx + \sin y",
                          r"-by + \sin z",
                          r"-bz + \sin x")

WindmiAttractor = _get_3d_equation(r"y",
                                   r"z",
                                   r"-az - y + b - \exp x")

Rucklidge = _get_3d_equation(r"\kappa x + \lambda y -yz",
                             r"x",
                             r"-z + y^2")

SimplestQuadraticChaotic = _get_3d_equation(r"y",
                                            r"z",
                                            r"-az + y^2 -x")

SimplestCubicChaotic = _get_3d_equation(r"y",
                                        r"z",
                                        r"-az + xy^2 -x")

SimplestPiecewiseLinearChaotic = _get_3d_equation(r"y",
                                                  r"z",
                                                  r"-az -y + |x| - 1")

DoubleScroll = _get_3d_equation(r"y",
                                r"z",
                                r"-a[z + y + x - \text{sgn} x]")

LotkaVolterra = _get_2d_equation(r"a x - b xy",
                                 r"-cy + dxy")

SimplestDrivenChaotic = _get_3d_equation(r"y",
                                         r"-x^3 + \sin\Omega z",
                                         r"1")

UedaOscillator = _get_3d_equation(r"y",
                                  r"-x^3 -by + A\sin\Omega z",
                                  r"1")

Henon = _get_2d_equation(r"1 - a x_i^2 + by_i",
                         r"x_i",
                         differential=False)

Logistic = r"""\begin{aligned}
                    & x_{i+1} = r x_i(1-x_i) 
                    \end{aligned}"""

KuramotoSivashinsky = r"""y_{t}=-y y_{x}-(1+\epsilon) y_{x x}-y_{x x x x}"""

Lorenz96 = r"""\dot x_{i}=\left(x_{i+1}-x_{i-2}\right) x_{i-1}-x_{i}+F"""

LinearSystem = r"""\dot x = A \cdot x"""

LATEX_DICT = {
    "Lorenz63": Lorenz63,
    "Roessler": Roessler,
    "ComplexButterfly": ComplexButterfly,
    "Chen": Chen,
    "ChuaCircuit": ChuaCircuit,
    "Thomas": Thomas,
    "WindmiAttractor": WindmiAttractor,
    "Rucklidge": Rucklidge,
    "SimplestQuadraticChaotic": SimplestQuadraticChaotic,
    "SimplestCubicChaotic": SimplestCubicChaotic,
    "SimplestPiecewiseLinearChaotic": SimplestPiecewiseLinearChaotic,
    "DoubleScroll": DoubleScroll,
    "LotkaVolterra": LotkaVolterra,
    "SimplestDrivenChaotic": SimplestDrivenChaotic,
    "UedaOscillator": UedaOscillator,
    "Henon": Henon,
    "Logistic": Logistic,
    "KuramotoSivashinsky": KuramotoSivashinsky,
    "Lorenz96": Lorenz96,
    "LinearSystem": LinearSystem,
}
