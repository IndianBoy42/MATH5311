from numpy import *


def printM(M, out):
    savetxt(out, M, delimiter=',', newline=',//.\n')


N = 1000


def randomMatrix(N, out='inc/output.hmat'):
    M = random.randn(N, N)
    printM(M, out)


def bigIdentity(N, out='inc/eye.hmat'):
    M = eye(N)
    printM(M, out)


if __name__ == "__main__":
    bigIdentity(N)
    randomMatrix(N)
