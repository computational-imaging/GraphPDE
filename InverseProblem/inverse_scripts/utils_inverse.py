import sys
import os
import h5py
from dolfin import *
import dataio
sys.path.append("..")
import numpy as np
import torch
from glob import glob
import matplotlib.pyplot as plt
from torch_geometric.data import DataLoader
import gnn_module
from functools import partial
from scipy import spatial
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


def gen_mesh(vert,tri):
    mesh = Mesh()
    editor = MeshEditor()
    editor.open(mesh, 'triangle', 2, 2)
    editor.init_vertices(vert.shape[0])
    for idx, v in enumerate(vert):
        editor.add_vertex(idx, [v[0], v[1]])

    editor.init_cells(tri.shape[0])
    for idx, c in enumerate(tri):
        editor.add_cell(idx, [c[0], c[1], c[2]])
    editor.close()
    V=FunctionSpace(mesh, "Lagrange", 1)
    u = Function(V)
    return mesh,u,V


def solve_Eikonal(mesh):
    V=FunctionSpace(mesh, "Lagrange", 1)
    u_D = Expression('0', degree=1)

    def boundary(x, on_boundary):
        return on_boundary

    bc = DirichletBC(V, u_D, boundary)

    v = TestFunction(V)
    u = TrialFunction(V)
    f = Constant(1.0)
    y = Function(V)

    F1 = inner(grad(u), grad(v))*dx - f*v*dx
    solve(lhs(F1)==rhs(F1), y, bc)

    # Stabilized Eikonal equation 
    eps = Constant(mesh.hmax()/1)
    F = sqrt(inner(grad(y), grad(y)))*v*dx - f*v*dx + eps*inner(grad(y), grad(v))*dx
    solve(F==0, y, bc)
    return y

def if_continue(loss1,loss2,criteria=1e-10):
    if (loss1-loss2)**2/loss1**2<criteria:
        return False
    else:
        return True

def plot_input_graph(graph,extra=None,plot_fields=False,mask_idx=None,):
    if extra==None:
        num_fig = 4
        rownumber = 1
    else:
        rownumber = 1
        num_fig = 4+len(extra)
    if num_fig>5:
        rownumber = int(num_fig/5)+1
        num_fig = 5
   
    coords = graph.coords
    cells = graph.cell
    u = graph.x[:,0].detach().cpu().numpy()
    density = graph.x[:,2].detach().cpu().numpy()
    v = graph.x[:,1].detach().cpu().numpy()
    bdd = graph.x[:,3].detach().cpu().numpy()

    fig =  plt.figure(figsize=[4*num_fig,3*rownumber])
    fig.add_subplot(rownumber,num_fig,1)

    plt.tricontourf(coords[:,0],coords[:,1],cells,u, levels=50,cmap='jet',extend="max")
    plt.triplot(coords[:,0],coords[:,1],cells,linewidth=0.1)
    plt.colorbar()
    plt.title("input initial field")
    fig.add_subplot(rownumber,num_fig,2)
    plt.tricontourf(coords[:,0],coords[:,1],cells,v, levels=50,cmap='jet',extend="max")
    plt.triplot(coords[:,0],coords[:,1],cells,linewidth=0.1)
    plt.colorbar()
    plt.title("input initial velocity")
    fig.add_subplot(rownumber,num_fig,3)
    plt.tricontourf(coords[:,0],coords[:,1],cells,density, levels=50,extend="max")
    plt.triplot(coords[:,0],coords[:,1],cells,linewidth=0.1)
    plt.colorbar()
    plt.title("input density")
    fig.add_subplot(rownumber,num_fig,4)
    plt.scatter(coords[:,0],coords[:,1],s=(bdd*1>0)*10,c=bdd*1,cmap='jet')
    plt.colorbar()
    plt.title("input boundary nodes")
    
    for idx,element in enumerate(extra):
        element = element.detach().cpu().numpy()
        fig.add_subplot(rownumber,num_fig,4+idx+1)
        if idx==0:
            plt.tricontourf(coords[:,0],coords[:,1],cells,element, levels=50,extend="max")
            plt.triplot(coords[:,0],coords[:,1],cells,linewidth=0.1)
            plt.colorbar()
            plt.title("gt parameter")
        elif plot_fields:
            plt.tricontourf(coords[:,0],coords[:,1],cells,element, levels=50,cmap='jet',extend="max")
            plt.triplot(coords[:,0],coords[:,1],cells,linewidth=0.1)
            plt.colorbar()
            plt.title("field_{}".format(idx))
        else:
            s = np.zeros(coords.shape[0])
            s[mask_idx]=10
            plt.scatter(coords[:,0],coords[:,1],s=s,c=element,cmap='jet')
            plt.colorbar()
            plt.title("field_{}".format(idx))
        