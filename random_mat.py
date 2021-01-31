from numpy import *


def printRmat(M, out, suff='rmat'):
    savetxt(f'{out}.{suff}', M, delimiter=',', newline='\n')

def printHmat(M, out, suff='hmat'):
    savetxt(f'{out}.{suff}', M, delimiter=',', newline=',//.\n')


N = 1000


def randomMatrix(N, out='output'):
    M = random.randn(N, N)
    printHmat(M, out)
    printRmat(M, out)


def bigIdentity(N, out='eye'):
    M = eye(N)
    printHmat(M, out)
    printRmat(M, out)


if __name__ == "__main__":
    # bigIdentity(N)
    randomMatrix(N)
