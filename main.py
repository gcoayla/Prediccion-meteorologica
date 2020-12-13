from datetime import datetime
import csv
import tkinter
import math

# Makes the KD-Tree for fast lookup
def make_kd_tree(points, dim, i=0):
    if len(points) > 1:
        points.sort(key=lambda x: x[i])
        i = (i + 1) % dim
        half = len(points) >> 1
        return [
            make_kd_tree(points[: half], dim, i),
            make_kd_tree(points[half + 1:], dim, i),
            points[half]
        ]
    elif len(points) == 1:
        return [None, None, points[0]]


# Adds a point to the kd-tree
def add_point(kd_node, point, dim, i=0):
    if kd_node is not None:
        dx = kd_node[2][i] - point[i]
        i = (i + 1) % dim
        for j, c in ((0, dx >= 0), (1, dx < 0)):
            if c and kd_node[j] is None:
                kd_node[j] = [None, None, point]
            elif c:
                add_point(kd_node[j], point, dim, i)

# k nearest neighbors
def get_knn(kd_node, point, k, dim, dist_func, return_distances=True, i=0, heap=None):
    import heapq
    is_root = not heap
    if is_root:
        heap = []
    if kd_node is not None:
        dist = dist_func(point, kd_node[2])
        dx = kd_node[2][i] - point[i]
        if len(heap) < k:
            heapq.heappush(heap, (-dist, kd_node[2]))
        elif dist < -heap[0][0]:
            heapq.heappushpop(heap, (-dist, kd_node[2]))
        i = (i + 1) % dim
        # Goes into the left branch, and then the right branch if needed
        for b in [dx < 0] + [dx >= 0] * (dx * dx < -heap[0][0]):
            get_knn(kd_node[b], point, k, dim, dist_func, return_distances, i, heap)
    if is_root:
        neighbors = sorted((-h[0], h[1]) for h in heap)
        return neighbors if return_distances else [n[1] for n in neighbors]

# For the closest neighbor
def get_nearest(kd_node, point, dim, dist_func, return_distances=True, i=0, best=None):
    if kd_node is not None:
        dist = dist_func(point, kd_node[2])
        dx = kd_node[2][i] - point[i]
        if not best:
            best = [dist, kd_node[2]]
        elif dist < best[0]:
            best[0], best[1] = dist, kd_node[2]
        i = (i + 1) % dim
        # Goes into the left branch, and then the right branch if needed
        for b in [dx < 0] + [dx >= 0] * (dx * dx < best[0]):
            get_nearest(kd_node[b], point, dim, dist_func, return_distances, i, best)
    return best if return_distances else best[1]



"""
If you want to attach other properties to your points, 
you can use this class or subclass it.
Usage:
point = PointContainer([1,2,3])
point.label = True  
print point         # [1,2,3]
print point.label   # True 
"""
class PointContainer(list):
    def __new__(self, value, name = None, values = None):
        s = super(PointContainer, self).__new__(self, value)
        return s


"""
Below is all the testing code
"""

import random, cProfile


def puts(l):
    for x in l:
        print(x)


def get_knn_naive(points, point, k, dist_func, return_distances=True):
    neighbors = []
    for i, pp in enumerate(points):
        dist = dist_func(point, pp)
        neighbors.append((dist, pp))
    neighbors = sorted(neighbors)[:k]
    return neighbors if return_distances else [n[1] for n in neighbors]

dim = 3

def rand_point(dim):
    return [random.uniform(-1, 1) for d in range(dim)]

def dist_sq(a, b, dim):
    return sum((a[i] - b[i]) ** 2 for i in range(dim))

def dist_sq_dim(a, b):
    return dist_sq(a, b, dim)

data = []

with open('datos/train3.csv', encoding="utf8") as csvfile:
    reader = csv.DictReader(csvfile)
    for row in reader:
        punto = []
        punto.append(float(row['lat']))
        punto.append(float(row['lon']))
        ano=row['ano']
        mes=row['mes']
        dia = row['dia']
        new_date = datetime(int(ano), int(mes), int(dia), 0, 0, 0, 0)
        punto.append(int(new_date.strftime("%j")))
        point = PointContainer(punto)
        punto2 = []
        punto2.append(row['pre'])
        punto2.append(row['max'])
        punto2.append(row['min'])
        point.label = punto2
        data.append(point)

kd_tree = make_kd_tree(data, dim)

ventana=tkinter.Tk()
ventana.geometry("800x480")

labelprueba=tkinter.Label(ventana,text="",width=5)

labellat=tkinter.Label(ventana,text="Latitud")
labellon=tkinter.Label(ventana,text="Longitud")
labelano=tkinter.Label(ventana,text="Año")
labelmes=tkinter.Label(ventana,text="Mes")
labeldia=tkinter.Label(ventana,text="Día")

cajalat=tkinter.Entry(ventana)
cajalon=tkinter.Entry(ventana)
cajaano=tkinter.Entry(ventana)
cajames=tkinter.Entry(ventana)
cajadia=tkinter.Entry(ventana)


labellat.grid(row=0,column=1)
labellon.grid(row=0,column=2)
labelano.grid(row=0,column=3)
labelmes.grid(row=0,column=4)
labeldia.grid(row=0,column=5)

labelprueba.grid(row=0,column=0)

cajalat.grid(row=1,column=1)
cajalon.grid(row=1,column=2)
cajaano.grid(row=1,column=3)
cajames.grid(row=1,column=4)
cajadia.grid(row=1,column=5)


labpremax=tkinter.Label(ventana,text="Precipitación máxima: ")
labpremmin=tkinter.Label(ventana,text="Precipitación mínima: ")
labpropre=tkinter.Label(ventana,text="Probabilidad de precipitación: ")
labpremed=tkinter.Label(ventana,text="Precipitación media: ")

reslabpremax=tkinter.Label(ventana,text="")
reslabpremmin=tkinter.Label(ventana,text="")
reslabpropre=tkinter.Label(ventana,text="")
reslabpremed=tkinter.Label(ventana,text="")

labpremax.grid(row=2,column=1)
labpremmin.grid(row=3,column=1)
labpropre.grid(row=4,column=1)
labpremed.grid(row=5,column=1)

reslabpremax.grid(row=2,column=2)
reslabpremmin.grid(row=3,column=2)
reslabpropre.grid(row=4,column=2)
reslabpremed.grid(row=5,column=2)

labtempmax=tkinter.Label(ventana,text="Temperatura máxima: ")
labtempmaxesp=tkinter.Label(ventana,text="Temperatura máxima esperada: ")
labtempmin=tkinter.Label(ventana,text="Temperatura mínima: ")
labtempminesp=tkinter.Label(ventana,text="Temperatura mínima esperada: ")

reslabtempmax=tkinter.Label(ventana,text="")
reslabtempmaxesp=tkinter.Label(ventana,text="")
reslabtempmin=tkinter.Label(ventana,text="")
reslabtempminesp=tkinter.Label(ventana,text="")

labtempmax.grid(row=2,column=3)
labtempmaxesp.grid(row=3,column=3)
labtempmin.grid(row=4,column=3)
labtempminesp.grid(row=5,column=3)

reslabtempmax.grid(row=2,column=4)
reslabtempmaxesp.grid(row=3,column=4)
reslabtempmin.grid(row=4,column=4)
reslabtempminesp.grid(row=5,column=4)

def consulta(lat,lon,di):
    result1 = []
    test = [lat, lon, di]
    vecinos = get_knn(kd_tree, test, 15, dim, dist_sq_dim)

    # Precipitacion
    preci = 0;
    npreci = 0;
    ppreci = 0;
    maxpreci = 0;
    minpreci = 20;

    # Temperaturamax
    tmax = 0
    ntmax = 0
    tmaxesp = 0

    # Temperaturamin
    tmin = 100
    ntmin = 0
    tminesp = 0
    '''
    for x in vecinos:
        print(x[1])
        print(x[1].label)
    '''
    for x in vecinos:
        if float(x[1].label[0]) != -99.9:
            if float(x[1].label[0]) > 0:
                ppreci += 1
            if float(x[1].label[0]) > maxpreci:
                maxpreci = float(x[1].label[0])
            if float(x[1].label[0]) <= minpreci:
                minpreci = float(x[1].label[0])
            preci += float(x[1].label[0])
            npreci += 1
        if float(x[1].label[1]) != -99.9:
            if float(x[1].label[1]) > tmax:
                tmax = float(x[1].label[1])
            ntmax += 1
            tmaxesp += float(x[1].label[1])
        if float(x[1].label[2]) != -99.9:
            if float(x[1].label[2]) < tmin:
                tmin = float(x[1].label[2])
            ntmin += 1
            tminesp += float(x[1].label[2])
    if ppreci==0:
        ppreci=1
    resultado = [maxpreci,minpreci,ppreci/npreci,preci/ppreci,tmax,tmaxesp/ntmax,tmin,tminesp/ntmin]
    return resultado

def ejecbtn():
    latitud=float(cajalat.get())
    longitud=float(cajalon.get())
    anoca = int(cajaano.get())
    mesca = int(cajames.get())
    diaca = int(cajadia.get())
    nueva = datetime(anoca,mesca,diaca, 0, 0, 0, 0)
    diaano=int(nueva.strftime("%j"))
    dev = consulta(latitud, longitud, diaano)
    reslabpremax["text"]=str(dev[0])
    reslabpremmin["text"]=str(dev[1])
    reslabpropre["text"]=str(dev[2])
    reslabpremed["text"]=str(dev[3])
    reslabtempmax["text"]=str(dev[4])
    reslabtempmaxesp["text"]=str(dev[5])
    reslabtempmin["text"]=str(dev[6])
    reslabtempminesp["text"]=str(dev[7])


botonbus=tkinter.Button(ventana, text="Buscar",command = ejecbtn)
botonbus.grid(row=1,column=6)

ventana.mainloop()

''' 
print("----------")
print("Precipitación máxima: "+str(dev[0]))
print("Precipitación mínima: "+ str(dev[1]))
print("Probabilidad de precipitación: " + str(dev[2]))
print("Precipitación media: "+ str(dev[3]))
print("----------")
print("Temperatura máxima: " + str(dev[4]))
print("Temperatura máxima esperada: " + str(dev[5]))
print("----------")
print("Temperatura mínima: " + str(dev[6]))
print("Temperatura mínima esperada: " + str(dev[7]))
'''
'''   
result1 = []
test = [-16.0576745834216, 	-72.67830488049958, 100]

print("Latitud: " + str(test[0]) + " Longitud: " + str(test[1]) + " Día: " + str(test[2]))

vecinos = get_knn(kd_tree, test, 20, dim, dist_sq_dim)

#Precipitacion
preci = 0;
npreci = 0;
ppreci = 0;
maxpreci = 0;
minpreci  = 20;

#Temperaturamax
tmax = 0
ntmax = 0
tmaxesp = 0

#Temperaturamin
tmin = 100
ntmin = 0
tminesp = 0

for x in vecinos:
    print(x[1].label)

for x in vecinos:
    if float(x[1].label[0]) != -99.9:
        if float(x[1].label[0]) >0:
            ppreci += 1
        if float(x[1].label[0]) >maxpreci:
            maxpreci=float(x[1].label[0])
        if float(x[1].label[0]) <= minpreci:
            minpreci=float(x[1].label[0])
        preci += float(x[1].label[0])
        npreci += 1
    if float(x[1].label[1]) != -99-0:
        if float(x[1].label[1]) > tmax:
            tmax=float(x[1].label[1])
        ntmax+=1
        tmaxesp+=float(x[1].label[1])
    if float(x[1].label[2]) != -99.9:
        if float(x[1].label[2]) < tmin:
            tmin = float(x[1].label[2])
        ntmin+=1
        tminesp+=float(x[1].label[2])

print("----------")
print("Precipitación máxima: "+str(maxpreci))
print("Precipitación mínima: "+ str(minpreci))
print("Probabilidad de precipitación: " + str(ppreci/npreci))
print("Precipitación media: "+ str(preci/ppreci))
print("----------")
print("Temperatura máxima: " + str(tmax))
print("Temperatura máxima esperada: " + str(tmaxesp/ntmax))
print("----------")
print("Temperatura mínima: " + str(tmin))
print("Temperatura mínima esperada: " + str(tminesp/ntmin))
'''


