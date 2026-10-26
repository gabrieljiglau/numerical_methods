from numpy import shape
from src.data import convert_input
from src.linear_models import linear_solver


if __name__ == '__main__':

    A, b = convert_input()
    print(f"A = {A} \n"
          f"b = {b} \n")
    print(f"b.shape = {shape(b)[0]}")

    linear_solver(A, b, w=1.99, ascending=False, algorithm='gauss_seidel')