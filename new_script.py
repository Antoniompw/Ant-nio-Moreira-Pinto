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
        self.vertexes = []
        self.inverse_paths = {}
        csv = open(file_name)
        for line, _ in zip(csv, range(200000)):
            # If the start point has not been started
            row = line.split(' ')
            row[1] = row[1].rstrip("\n")
            if(self.paths.get(row[0], None) == None):
                self.paths[row[0]] = []

            # Register path
            self.paths[row[0]].append(row[1])

            if row[0] not in self.vertexes:
                self.vertexes.append(row[0])

    def chunker_list(self, seq, size):
        return (seq[i::size] for i in range(size))

    def busca_em_largura(self):
        index = 0
        if RANK == 0:
            import time
            start_time = time.time()
            data = [self.vertexes[index]]
        else:
            data = None
            scattered_data = None

        while True:
            if RANK == 0:
                scattered_data = self.chunker_list(data, SIZE)
            
            data = COMM.scatter(scattered_data, root=0)

            new_list = []

            self.inverse_paths = COMM.bcast(self.inverse_paths, root=0)
            for i in data:
                if self.paths.get(i, None) != None:
                    for j in self.paths[i]:
                        if self.inverse_paths.get(j, None) == None:
                            new_list.append(j)
            
            #print(new_list,"expanded queue", RANK)
            
            data = COMM.gather(new_list, root = 0)

            if RANK == 0:
                for i in data:
                    for j in i:
                        if self.inverse_paths.get(j, None) == None:
                            self.inverse_paths[j] = self.vertexes[index]
                data = [item for sublist in data for item in sublist]
                #print(self.inverse_paths, "--Inverted paths--")
                #print()
                if len(data) == 0:
                    #print("search n:"+str(index)+" ended")
                    #print("===========================")
                    #print("===========================")
                    #print("BFS N°"+str(index)+" result "+str(self.inverse_paths))
                    self.inverse_paths = {}
                    index = index + 1
                    if index < len(self.vertexes):
                        data = [self.vertexes[index]]

            index = COMM.bcast(index,0)
            if index == len(self.vertexes):
                if RANK == 0:
                    with open("result.txt", "a") as file:
                        file.write("How many processors "+str(SIZE)+"\n")
                        file.write("--- %s seconds ---" % (time.time() - start_time)+"\n")
                return

if __name__ == "__main__":
    #print(RANK)
    if RANK == 0:
        file_name = 'web-Google.txt'
        graph = Graph(file_name)
    else:
        graph = None

    graph = COMM.bcast(graph, root = 0)

    graph.busca_em_largura()