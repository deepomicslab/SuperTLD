# _*_ coding: utf-8 _*_
"""
Time:     2021/8/27 21:33
Author:   WANG Bingchen
Version:  V 0.1
File:     cell_circle.py
Describe: 
"""

import numpy as np
import umap
from .distance import distanceInOval





def angle2plane_coordinate(angle, a, b, k):
    xcor = a*np.cos(angle)
    ycor = b*np.sqrt(np.exp(k*a*np.cos(angle)))*np.sin(angle)

    plane_coordinate = np.concatenate((xcor, ycor), axis=1)
    return plane_coordinate





class CellCircle():

    def __init__(self, n_neighbors=20, min_dist=0.01):
        self.n_neighbors = n_neighbors
        self.min_dist = min_dist
        self.n_components = 1



    def cal_cell_circle(self, data, a=3, b=2, k=0.2):
        """

        :param data: input data array(n_features, n_cells)
        :param a: major axis length
        :param b: minor axie length
        :param k: Deformation parameter
        :return: {
                    angle: embeddings as angles(Unit: rad),
                    plane_embedding: embeddings in plane coordinate
                }
        """
        m, n = data.shape
        print("%d cells, %d features" % (n, m))
        data = data.transpose()

        mapper = umap.UMAP(n_components=self.n_components, n_neighbors=self.n_neighbors, min_dist=self.min_dist,
                           output_metric=distanceInOval, output_metric_kwds={"a": a, "b": b, "k": k}).fit(data)

        angle = mapper.transform(data)
        angle = angle % (2 * np.pi)

        plane_embedding = angle2plane_coordinate(angle, a=a, b=b, k=k)

        return {
            "angle": angle,
            "plane_embedding": plane_embedding
        }