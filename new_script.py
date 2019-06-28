from mpi4py import MPI

READY, START, DONE, EXIT, LOG, NONE = 0, 1, 2, 3, 4, 5
COMM = MPI.COMM_WORLD
SIZE = COMM.Get_size()
RANK = COMM.rank
STATUS = MPI.Status()

class Graph():
    # Constructs a graph
    def __init__(self, file_name):
        import csv
        import pandas as pd

        self.paths = {}
        self.vertexes = {}
        self.inverse_paths = {}
        csv = open(file_name)
        for line in csv:
            # If the start point has not been started
            row = line.split(' ')
            row[1] = row[1].rstrip("\n")
            if(self.paths.get(row[0], None) == None):
                self.paths[row[0]] = {}

            # Register path
            self.paths[row[0]][row[1]] = 1

            self.vertexes[row[0]] = 1
    def chunks(self, l, n):
        l = [*l]
        n = max(1, n)
        return (l[i:i+n] for i in range(0, len(l), n))

    def busca_em_largura(self):
        import time
        start_time = time.time()
        list = [*self.vertexes.keys()]
        index = 0
        if RANK == 0:
            data = [list[index]]
        else:
            data = None
            chunks = None

        while True:
            if RANK == 0:
                # dividing data into chunks
                chunks = [[] for _ in range(SIZE)]
                for i, chunk in enumerate(data):
                    chunks[i % SIZE].append(chunk)
            data = COMM.scatter(chunks, root=0)
            print(data, "original")
            new_list = []
            for i in data:
                if self.paths.get(i, None) != None:
                    for key in [*self.paths[i].keys()]:
                        new_list.append(key)
            
            data = COMM.gather(new_list, root = 0)
            print(new_list,"expanded queue")
            if RANK == 0:
                for i in data:
                    for j in i:
                        if self.inverse_paths.get(j, None) == None:
                            self.inverse_paths[j] = list[index]
                

                data = [item for sublist in data for item in sublist]
                print(self.inverse_paths, "--Inverted paths--")
                print()
                if len(data) == 0:
                    self.inverse_paths = {}
                    index = index + 1
            list = COMM.bcast(data,0)
            if list == []:
                print("Done")
                print("--- %s seconds ---" % (time.time() - start_time))
                return

    def count_all():
        counts = []
        for i in self.vertexes.keys():
            counts.append(self.count(self.inverse_paths, raiz, i))

        return counts

    def count(self, inverse_paths, origin, destination):
        c = 0
        while origin != destination:
            if inverse_paths.get(destination, None) == None:
                return -1
            destination = inverse_paths[destination]
            c = c+1
        return c


if __name__ == "__main__":
    if RANK == 0:
        file_name = 'test.txt'
        graph = Graph(file_name)
    else:
        graph = None

    graph = COMM.bcast(graph, root = 0)
    print("Results:")
    graph.busca_em_largura()