#!/usr/bin/env python3

import argparse
import math
import sys


ALLOWED_NAMES = {k: getattr(math, k) for k in dir(math) if not k.startswith("_")}  # Buduje słownik dozwolonych nazw z biblioteki math
ALLOWED_NAMES.update({"abs": abs, "min": min, "max": max})                         # Dodaje jeszcze 3 domyślne funkcje do słownika

DERIVATIVE_EPS = 1e-12  # Minimalna wartość pochodnej != 0


def positive_float(text):
    try:
        value = float(text)

    except ValueError:
        raise argparse.ArgumentTypeError("'%s' nie jest liczbą" % text)
    
    if value <= 0:
        raise argparse.ArgumentTypeError("wartość musi być > 0 (podano %s)" % text)
    
    return value


def positive_int(text):
    try:
        value = int(text)

    except ValueError:
        raise argparse.ArgumentTypeError("'%s' nie jest liczbą całkowitą" % text)
    
    if value <= 0:
        raise argparse.ArgumentTypeError("wartość musi być > 0 (podano %s)" % text)
    
    return value


def parse_args(argv=None):
    parser = argparse.ArgumentParser(
        prog="newton.py",
        description="Program poszukuje miejsca zerowego funkcji jednej zmiennej x metodą Newtona. "
                    "Pochodna liczona jest numerycznie.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        add_help=False,                                                   # add_help=False, bo -h jest użyte na krok pochodnej
    )
    parser.add_argument("function", metavar="FUNKCJA",
                        help="wzór funkcji zmiennej x"
                             "(dostępne: sin, cos, exp, log, sqrt, pi, e, ...; ^ = potęga)")
    parser.add_argument("-x", "--start", metavar="X0", type=float, default=1.0,
                        help="punkt startowy (domyślnie: %(default)s)")
    parser.add_argument("-h", "--step", metavar="H", type=positive_float, default=1e-5,
                        help="krok h w pochodnej numerycznej (domyślnie: %(default)s)")
    parser.add_argument("-n", "--iterations", metavar="N", type=positive_int, default=100,
                        help="maksymalna liczba kroków (domyślnie: %(default)s)")
    parser.add_argument("-e", "--eps", metavar="EPS", type=positive_float, default=1e-8,
                        help="dokładność: szukamy x, dla którego |f(x)| < EPS "
                             "(domyślnie: %(default)s)")
    parser.add_argument("-q", "--quiet", action="store_true",
                        help="nie wypisuj tabeli kroków, tylko wynik")
    parser.add_argument("--help", action="help", help="pokaż pomoc i zakończ")
    return parser.parse_args(argv)


# Zamienia string na funkcję
def build_function(expr):
    expr = expr.strip().replace("^", "**")
    if not expr:
        raise ValueError("pusty wzór funkcji")
    try:
        code = compile(expr, "<funkcja>", "eval")
    except SyntaxError as exc:
        raise ValueError("błąd składni we wzorze '%s': %s" % (expr, exc.msg))

    def f(x):
        return eval(code, {"__builtins__": {}}, dict(ALLOWED_NAMES, x=x))            # f(x) pamięta zmienną code, do słownika dodajemy x

    return f


# Metoda Newtona
def derivative(f, x, h):
    return (f(x + h) - f(x - h)) / (2 * h)

# Zwraca (x, zbieżność: bool, komunikat)
def newton(f, x0, h, max_iter, eps, verbose=True):
    x = x0
    if verbose:
        print("%5s  %22s  %22s" % ("krok", "x", "f(x)"))
    for i in range(max_iter + 1):
        fx = f(x)
        if verbose:
            print("%5d  %22.15g  %22.15g" % (i, x, fx))
        if abs(fx) < eps:
            return x, True, "znaleziono miejsce zerowe po %d krokach" % i
        if i == max_iter:
            break
        dfx = derivative(f, x, h)
        if abs(dfx) < DERIVATIVE_EPS:
            return x, False, "pochodna bliska zera w x=%g - metoda nie może być kontynuowana" % x
        x_new = x - fx / dfx
        if not math.isfinite(x_new):
            return x, False, "wynik kroku nie jest skończony"
        x = x_new
    return x, False, ("nie osiągnięto dokładności %g w %d krokach "
                      "(funkcja może nie mieć miejsc zerowych)" % (eps, max_iter))


def main():
    args = parse_args()

    try:
        f = build_function(args.function)
        f(args.start)  # Test wywołania, wychwytuje nieznane nazwy

    except ValueError as exc:
        print("Błąd: %s" % exc, file=sys.stderr)

    except NameError as exc:
        print("Błąd: nieznana nazwa we wzorze (%s)" % exc, file=sys.stderr)

    except (TypeError, ZeroDivisionError, OverflowError) as exc:
        print("Błąd przy obliczaniu funkcji w x=%g: %s" % (args.start, exc), file=sys.stderr)

    try:
        x, converged, message = newton(f, args.start, args.step, args.iterations, args.eps, verbose=not args.quiet)
        
    except (ValueError, ZeroDivisionError, OverflowError) as exc:
        print("Błąd numeryczny w trakcie iteracji: %s" % exc, file=sys.stderr)
        return 3

    print()
    print(message)
    if converged:
        print("x = %.12g,  f(x) = %.3g" % (x, f(x)))
        return 0
    print("ostatnie przybliżenie: x = %.12g" % x, file=sys.stderr)
    return 1


if __name__ == "__main__":
    sys.exit(main())
