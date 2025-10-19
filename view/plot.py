import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from mpl_toolkits.mplot3d.art3d import Poly3DCollection
from matplotlib.font_manager import FontProperties
import imageio
import shutil
from matplotlib import image
from pathlib import Path

class ResultViewer():
    def __init__(self, dim, data, ground_truth=None, initial_hyperplanes=None, convex_region_hyperplanes=None, X_limit=[-5., 5.], Y_limit=[-5., 5.], Z_limit=[-5., 5.]):
        
        self.dim = dim
        self.data = data
        self.ground_truth = ground_truth  # black line
        self.initial_hyperplanes = initial_hyperplanes  # blue line
        self.convex_region_hyperplanes = convex_region_hyperplanes   # red dot line
        
        self.A = None
        self.b = None
        self.A_bound = None
        self.b_bound = None
        
        self.xlim = X_limit
        self.ylim = Y_limit
        self.zlim = Z_limit
        
        self.boundary_hyperplanes()
        
        self.fig = plt.figure()
        if self.dim == 2:
            self.ax = self.fig.add_subplot(111)
            self.ax.set_xlim([self.xlim[0], self.xlim[1]])
            self.ax.set_ylim([self.ylim[0], self.ylim[1]])
        elif self.dim == 3:
            self.ax = self.fig.add_subplot(111, projection='3d')
            # self.ax = Axes3D(self.fig)
            self.ax.set_xlim([self.xlim[0], self.xlim[1]])
            self.ax.set_ylim([self.ylim[0], self.ylim[1]])
            self.ax.set_zlim([self.zlim[0], self.zlim[1]])
            
        
        
        
    
    def draw_result(self, ground_truth = True, initial_hyperplane = True, convex_region_hyperplane = True):
        
        if ground_truth and self.ground_truth is not None:
            # dark line
            self.hyperplane_plot(self.ground_truth, marker=",", markersize=0.5, linewidth = 0.5, linestyle = ':', color="black", label = 'ground truth')
        
        if initial_hyperplane and self.initial_hyperplanes is not None:
            # blue dotted line
            self.hyperplane_plot(self.initial_hyperplanes, marker=",", markersize=1, linewidth = 1, linestyle = '--', color="blue", label = 'intial hyperplanes')
        
        if convex_region_hyperplane and self.convex_region_hyperplanes is not None:
            # red dotted line
            # self.hyperplane_plot(self.convex_region_hyperplanes, marker=',', markersize=1, linewidth = 1, linestyle = '--', color="red", label = 'updated hyperplanes')
            # red solid line
            self.polygon_plot(label = 'collision-free convex region')
        self.data_plot(self.data)
        
    def boundary_hyperplanes(self):
        
        if self.dim == 2:
            A_bound = np.zeros((4, 2))
            b_bound = np.zeros((4,))
            A_bound[0] = np.array([1., 0.])
            A_bound[1] = np.array([-1., 0.])
            A_bound[2] = np.array([0., 1.])
            A_bound[3] = np.array([0., -1.])
            b_bound[0] = -self.xlim[1]
            b_bound[1] = self.xlim[0]
            b_bound[2] = -self.ylim[1]
            b_bound[3] = self.ylim[0]
        elif self.dim == 3:
            A_bound = np.zeros((6, 3))
            b_bound = np.zeros((6,))
            A_bound[0] = np.array([1., 0., 0.])
            A_bound[1] = np.array([-1., 0., 0.])
            A_bound[2] = np.array([0., 1., 0.])
            A_bound[3] = np.array([0., -1., 0.])
            A_bound[4] = np.array([0., 0., 1.])
            A_bound[5] = np.array([0., 0., -1.])
            b_bound[0] = -self.xlim[1]
            b_bound[1] = self.xlim[0]
            b_bound[2] = -self.ylim[1]
            b_bound[3] = self.ylim[0]
            b_bound[4] = -self.zlim[1]
            b_bound[5] = self.zlim[0]
        
        self.A_bound = A_bound
        self.b_bound = b_bound


    def get_intersections(self):
        A = self.convex_region_hyperplanes.A
        b = self.convex_region_hyperplanes.b
        A_num = len(A)
        self.A = np.vstack((A, self.A_bound))
        self.b = np.append(b, self.b_bound)
        
        intersections = []
        intersections_dict = {}   # {index_hyperplane: point}
        
        for i in range(len(self.A)):
            intersections_dict[i] = []
            
        if self.dim == 2:
            for i in range(len(self.A) - 1):
                for j in range(i+1, len(self.A)):
                    A_temp = self.A[[i, j], :] # np.array([self.A[i], self.A[j]])
                    b_temp = self.b[[i, j]] # np.array([self.b[i], self.b[j]])
                    if not (np.dot(A_temp[0], A_temp[1]) >= 1. or np.dot(A_temp[0], A_temp[1]) <= -1.):
                        intersection = -np.dot(np.linalg.inv(A_temp), b_temp)
                        intersections.append(intersection)
                        intersections_dict[i].append(intersection)
                        intersections_dict[j].append(intersection)
            for index, point in enumerate(intersections):
                if np.max(self.A.dot(point) + self.b) > 1e-2:
                    for key in intersections_dict:
                        for i in range(len(intersections_dict[key])):
                            if np.array_equal(intersections_dict[key][i], point):
                                intersections_dict[key].pop(i)
                                break
        elif self.dim == 3:
            for i in range(len(self.A) - 2):
                for j in range(i+1, len(self.A) - 1):
                    for k in range(j+1, len(self.A)):
                        A_temp = self.A[[i, j, k], :]
                        b_temp = self.b[[i, j, k]]
                        if (j == A_num and k == A_num+1) or (j == A_num+2 and k == A_num+3) or (j == A_num+4 and k == A_num+5):
                            continue
                        elif (np.dot(A_temp[0], A_temp[1]) >= 1. or np.dot(A_temp[0], A_temp[1]) <= -1.):
                            continue
                        elif (np.dot(A_temp[0], A_temp[2]) >= 1. or np.dot(A_temp[0], A_temp[2]) <= -1.):
                            continue
                        elif np.linalg.det(A_temp) == 0.:
                            continue
                        else:
                            intersection = -np.dot(np.linalg.inv(A_temp), b_temp)
                            intersections.append(intersection)
                            intersections_dict[i].append(intersection)
                            intersections_dict[j].append(intersection)
                            intersections_dict[k].append(intersection)
            for key in intersections_dict:
                for i in range(len(intersections_dict[key]) - 1, -1, -1):
                    point = intersections_dict[key][i]
                    if np.max(A.dot(point) + b) > 1e-2 or point[0] < self.xlim[0] or point[0] > self.xlim[1] or point[1] < self.ylim[0] or point[1] > self.ylim[1] or point[2] < self.zlim[0] or point[2] > self.zlim[1]:
                        intersections_dict[key].pop(i)
                        
        return intersections_dict, intersections
                        
    def sort_intersections(self, point_list, hyperplane_index, A):
        avg = np.array([0.]*self.dim)
        result = []
        if len(point_list) > 0:
            point_num = len(point_list)
            for i in range(point_num):
                avg[0] += point_list[i][0] / point_num
                avg[1] += point_list[i][1] / point_num
                if self.dim == 3:
                    avg[2] += point_list[i][2] / point_num
        thetas = []
        if len(point_list) < 2:
            return result
        
        if self.dim == 2:
            for point in point_list:
                thetas.append(np.atan2(point[1]-avg[1], point[0]-avg[0]))
        elif self.dim == 3:
            e1 = point_list[0] - avg
            normal = A[hyperplane_index]
            e2 = np.cross(e1, normal)
            e1_norm = np.linalg.norm(e1)
            e2_norm = np.linalg.norm(e2)
            e1 /= e1_norm
            e2 /= e2_norm
            a = np.zeros((2, 2))
            a[0][0] = e1[0]
            a[0][1] = e2[0]
            a[1][0] = e1[1]
            a[1][1] = e2[1]
            inv_A = np.linalg.pinv(a)
            b = np.array([0., 0., 0.])
            for point in point_list:
                b[0] = point[0] - avg[0]
                b[1] = point[1] - avg[1]
                b[2] = point[2] - avg[2]
                # x_on_hyperplane = inv_A.dot(b)
                proj = b - np.dot(b, normal) * normal
                x = np.dot(proj, e1)
                y = np.dot(proj, e2)
                theta = np.atan2(y, x)
                thetas.append(theta)
        index_list = np.argsort(thetas)
        for index in index_list:
            result.append(point_list[index])
        
        return result
        
    
    def intersections_with_boundary(self, hyperplanes):
        A = hyperplanes.A
        b = hyperplanes.b
        
        intersections = []
        intersections_dict = {}   # {index_hyperplane: point}
        
        for i in range(len(A)):
            intersections_dict[i] = []
            
        if self.dim == 2:
            for i in range(len(A)):
                for j in range(len(self.A_bound)):
                    A_temp = np.array([A[i], self.A_bound[j]])
                    b_temp = np.array([b[i], self.b_bound[j]])
                    if not (np.dot(A_temp[0], A_temp[1]) >= 1. or np.dot(A_temp[0], A_temp[1]) <= -1.):
                        intersection = -np.dot(np.linalg.inv(A_temp), b_temp)
                        intersections.append(intersection)
                        intersections_dict[i].append(intersection)
            for index, point in enumerate(intersections):
                if np.max(self.A_bound.dot(point) + self.b_bound) > 1e-2:
                    for key in intersections_dict:
                        for i in range(len(intersections_dict[key])):
                            if np.array_equal(intersections_dict[key][i], point):
                                intersections_dict[key].pop(i)
                                break
        elif self.dim == 3:
            for i in range(len(A)):
                for j in range(len(self.A_bound) - 1):
                    for k in range(j+1, len(self.A_bound)):
                        A_temp = np.array([A[i], self.A_bound[j], self.A_bound[k]])
                        b_temp = np.array([b[i], self.b_bound[j], self.b_bound[k]])
                        if (j == 0 and k == 1) or (j == 2 and k == 3) or (j == 4 and k == 5):
                            continue
                        elif (np.dot(A_temp[0], A_temp[1]) >= 1. or np.dot(A_temp[0], A_temp[1]) <= -1.):
                            continue
                        elif (np.dot(A_temp[0], A_temp[2]) >= 1. or np.dot(A_temp[0], A_temp[2]) <= -1.):
                            continue
                        elif np.linalg.det(A_temp) == 0.:
                            continue
                        else:
                            intersection = -np.dot(np.linalg.inv(A_temp), b_temp)
                            intersections.append(intersection)
                            intersections_dict[i].append(intersection)
            
            # for index, point in enumerate(intersections):
            #     if np.max(self.A_bound.dot(point) + self.b_bound) > 1e-2:
            for key in intersections_dict:
                for i in range(len(intersections_dict[key]) - 1, -1, -1):
                    point = intersections_dict[key][i]
                    if point[0] < self.xlim[0] or point[0] > self.xlim[1] or point[1] < self.ylim[0] or point[1] > self.ylim[1] or point[2] < self.zlim[0] or point[2] > self.zlim[1]:
                    # if np.array_equal(intersections_dict[key][i], point):
                        intersections_dict[key].pop(i)
                        
            
        return intersections_dict, intersections
    
    
    def cla(self):
        self.ax.cla()
        
    def show(self, flag=False, visible = True):
        if not visible:
            self.ax.grid(None)
            self.ax.axis("off")
        
        if flag:
            myfont = FontProperties(fname=r"c:\windows\fonts\times.ttf", size=14)
            
            # font.set_family('serif')
            # font.set_name('Times New Roman')  # Must be installed on your system
            # font.set_size(14)
            # font.set_weight('bold')

            self.ax.set_xlabel("x(m)", fontproperties = myfont)
            self.ax.set_ylabel("y(m)", fontproperties = myfont)
            if self.dim == 3:
                self.ax.set_zlabel("z(m)", fontproperties = myfont)
            """
            self.ax.set_xticks([-5, -4, -3, -2, -1, 0, 1, 2, 3, 4, 5], fontproperties = myfont)
            self.ax.set_yticks([-5, -4, -3, -2, -1, 0, 1, 2, 3, 4, 5], fontproperties = myfont)
            self.ax.set_xticklabels([-5, -4, -3, -2, -1, 0, 1, 2, 3, 4, 5], fontproperties = myfont)
            self.ax.set_yticklabels([-5, -4, -3, -2, -1, 0, 1, 2, 3, 4, 5], fontproperties = myfont)
            
                self.ax.set_zticks(np.linspace(self.zlim[0], self.zlim[1]), fontproperties = myfont)
                self.ax.set_zticklabels(np.linspace(self.zlim[0], self.zlim[1]), fontproperties = myfont)
            """
            
            # self.ax.lengend(prof = font)
            
            plt.savefig('../figures/result.pdf',dpi=300,bbox_inches = "tight")
            
        plt.tight_layout()
        plt.show()
        
        

    def line_plot(self, point1, point2, marker='o', markersize=8, linewidth=4, linestyle = '-', color="black", label = 'line'):
        if self.dim == 2:
            self.ax.plot([float(point1[0]), float(point2[0])], [float(point1[1]), float(point2[1])], marker=marker, markersize=markersize,  linewidth=linewidth, linestyle=linestyle, color=color, label = label)
        elif self.dim == 3:
            self.ax.plot([float(point1[0]), float(point2[0])], [float(point1[1]), float(point2[1])], [float(point1[2]), float(point2[2])], marker=marker, markersize=markersize,  linewidth=linewidth, linestyle=linestyle, color=color, label = label)
        
    
    def hyperplane_plot(self, hyperplanes, marker=",", markersize=1, linewidth=2, linestyle = '--', color="blue", label = 'line'):
        if self.dim == 3:
            intersection_Dict, intersection_list = self.intersections_with_boundary(hyperplanes)
            for key in intersection_Dict:
                point_polygon = self.sort_intersections(intersection_Dict[key], key, self.initial_hyperplanes.A)
                """
                x = []
                y = []
                z = []
                for item in point_polygon:
                    x.append(item[0])
                    y.append(item[1])
                    z.append(item[2])
                vert = [list(zip(x, y, z))]
                self.ax.add_collection3d(Poly3DCollection(vert))
                """
                for i in range(len(point_polygon)-1):
                    self.line_plot(point_polygon[i], point_polygon[i+1], marker=marker, markersize=markersize,  linewidth=linewidth, linestyle=linestyle, color=color, label = label)
                self.line_plot(point_polygon[0], point_polygon[-1], marker=marker, markersize=markersize,  linewidth=linewidth, linestyle=linestyle, color=color, label = label)
        elif self.dim == 2:
            intersection_Dict, intersection_list = self.intersections_with_boundary(hyperplanes)
            for key in intersection_Dict:
                for i in range(len(intersection_Dict[key]) - 1):
                    self.line_plot(intersection_Dict[key][i], intersection_Dict[key][i+1], marker=marker, markersize=markersize,  linewidth=linewidth, linestyle=linestyle, color=color, label = label)
    
    def polygon_plot(self, label = 'convex region'):
        if self.dim == 2:
            intersection_Dict, intersection_list = self.get_intersections()
            for key in intersection_Dict:
                for i in range(len(intersection_Dict[key]) - 1):
                    self.line_plot(intersection_Dict[key][i], intersection_Dict[key][i+1], color="red", label=label)
        elif self.dim == 3:
            intersection_Dict, intersection_list = self.get_intersections()
            for key in intersection_Dict:
                point_polygon = self.sort_intersections(intersection_Dict[key], key, self.A)
                print(len(point_polygon))
                if len(point_polygon) >= 2:
                    for i in range(len(point_polygon) - 1):
                        self.line_plot(point_polygon[i], point_polygon[i+1], color="red", label=label)
                    self.line_plot(point_polygon[0], point_polygon[-1], color="red", label=label)
                
    def data_plot(self, data, s=5, marker='.', c='black', label = 'data'):
        x = data[:, 0]
        y = data[:, 1]
        if self.dim == 2:
            self.ax.scatter(x, y, c=c, s =s, marker = marker, label = label)
        elif self.dim == 3:
            z = data[:, 2]
            self.ax.scatter(x, y, z, c=c, s =s, marker = marker, label = label)
          
