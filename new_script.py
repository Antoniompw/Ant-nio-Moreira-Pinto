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
        for line in csv:
            # If the start point has not been started
            row = line.split(' ')
            row[1] = row[1].rstrip("\n")
            if(self.paths.get(row[0], None) == None):
                self.paths[row[0]] = []

            # Register path
            self.paths[row[0]].append(row[1])

            if row[0] not in self.vertexes:
                self.vertexes.append(row[0])

    def chunks(self, l, n):
        l = [*l]
        n = max(1, n)
        return (l[i:i+n] for i in range(0, len(l), n))

    def busca_em_largura(self):
        index = 0
        if RANK == 0:
            data = [self.vertexes[index]]
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
            #print(data, "original", RANK)

            new_list = []
            for i in data:
                if self.paths.get(i, None) != None:
                    for j in self.paths[i]:
                        if self.inverse_paths.get(j, None) == None:
                            new_list.append(j)
            
            #print(new_list,"expanded queue", RANK)
            COMM.barrier()
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
                    self.inverse_paths = {}
                    index = index + 1
                    if index < len(self.vertexes):
                        data = [self.vertexes[index]]

            index = COMM.bcast(index,0)
            if index == len(self.vertexes):
                return

if __name__ == "__main__":
    #print(RANK)
    if RANK == 0:
        file_name = 'web-Google.txt'
        graph = Graph(file_name)
    else:
        graph = None

    graph = COMM.bcast(graph, root = 0)
    
    import time
    start_time = time.time()
    graph.busca_em_largura()
    COMM.barrier()
    if RANK == 0:
        with open("result.txt", "a") as file:
            file.write("How many processors"+SIZE)
            file.write("--- %s seconds ---" % (time.time() - start_time))